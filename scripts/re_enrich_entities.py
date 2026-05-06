# -*- coding: utf-8 -*-
"""
re_enrich_entities.py
=====================
Bo sung Entities + Node Relations vao Neo4j da co san.

LOGIC NHAT QUAN VOI PIPELINE GOC:
  1. parse_unified_response() (entities.py):
     - _normalize_entity_type()  : PascalCase + alias (Signer->Person, Authority->Organization)
     - Filter structural labels  : bo Document/LegalArticle/Article/Clause/Chunk
     - _normalize_entity_name()  : expand abbreviations (ENTITY_NAME_ALIASES), filter garbage
     - Dedup exact (case-insensitive) per entity type
     - _dedup_entity_values()    : substring containment dedup + alias_map
     - _normalize_relationship() : comprehensive alias map + verb-root fuzzy match
     - Redirect node_relation endpoints via entity_alias_map

  2. Accumulate per doc (relations.py dong 669-696):
     - entities: merge by exact match per type
     - node_relations: dedup by tuple key (source_node, target_node, relationship)

  3. Flush via enrich_chunk_entities(use_apoc=False) -> _enrich_fallback():
     - MERGE entity nodes + HAS_ENTITY edges (Python-side dedup)
     - MERGE src -> MERGE tgt -> MERGE rel edge
     - SET enriched_v2 = true

FLUSH STRATEGY:
  Sau moi LLM response -> parse -> accumulate -> FLUSH NGAY vao Neo4j
  Khong doi buffer day. Index song song voi LLM processing.

CHECKPOINT:
  Luu done_ids (set of qdrant_id) sau MOI BATCH flush.
  Khi crash va chay lai, chi xu ly cac chunk chua co trong done_ids.

DB SAFETY:
  Chi dung MERGE / SET / MATCH. Khong co DELETE / DROP / REMOVE.

CHAY:
  cd /path/to/Legal-RAG
  python scripts/re_enrich_entities.py
"""

import sys
import os
import time
import datetime
import pickle
import threading

# =====================================================================
# CONFIGURATION
# =====================================================================
BATCH_SIZE_LLM   = 8      # so chunk / 1 LLM prompt (giong BATCH_SIZE trong relations.py)
DRY_RUN          = False   # True = khong ghi DB, chi in thong ke
RESUME           = True    # True = skip chunk da enriched_v2 hoac da co trong checkpoint
MAX_CHUNKS       = None    # None = tat ca. Dat so nguyen de test (VD: 100)

# Neo4j credentials override (None = doc tu .env / os.environ)
NEO4J_URI_OVERRIDE      = None
NEO4J_USERNAME_OVERRIDE = None
NEO4J_PASSWORD_OVERRIDE = None

CHECKPOINT_FILE = ".checkpoints/re_enrich_done_ids.pkl"
# =====================================================================


# ── Logging (UTF-8 safe cho ca Windows va Linux) ─────────────────────
os.makedirs(".debug", exist_ok=True)
os.makedirs(".checkpoints", exist_ok=True)

_LOG_FILE = os.path.join(
    ".debug",
    f"re_enrich_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)


class _DualWriter:
    """Ghi dong thoi ra terminal (UTF-8) va file log."""
    def __init__(self, logpath: str):
        self._term = open(
            sys.__stdout__.fileno(), mode="w",
            encoding="utf-8", buffering=1, closefd=False,
        )
        self._log = open(logpath, "a", encoding="utf-8")

    def write(self, s: str):
        try:
            self._term.write(s)
        except Exception:
            pass
        if "\r" not in s:
            self._log.write(s)
            self._log.flush()

    def flush(self):
        try:
            self._term.flush()
        except Exception:
            pass
        self._log.flush()


sys.stdout = _DualWriter(_LOG_FILE)
sys.stderr = sys.stdout


def _log(msg: str):
    print(msg)


# ── Banner ───────────────────────────────────────────────────────────
_log("=" * 65)
_log(f"  RE-ENRICH ENTITIES")
_log(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
_log("=" * 65)
_log(f"  Log            : {_LOG_FILE}")
_log(f"  Checkpoint     : {CHECKPOINT_FILE}")
_log(f"  DRY_RUN        : {DRY_RUN}")
_log(f"  RESUME         : {RESUME}")
_log(f"  BATCH_SIZE_LLM : {BATCH_SIZE_LLM} chunks/prompt")
_log(f"  MAX_CHUNKS     : {MAX_CHUNKS or 'ALL'}")
_log(f"  FLUSH          : Ngay sau moi LLM response (song song)")
_log("")


# ── sys.path + .env ──────────────────────────────────────────────────
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_repo, ".env"), override=False)
    _log("  .env           : loaded")
except ImportError:
    _log("  .env           : (dotenv not installed)")

if NEO4J_URI_OVERRIDE:
    os.environ["NEO4J_URI"] = NEO4J_URI_OVERRIDE
if NEO4J_USERNAME_OVERRIDE:
    os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME_OVERRIDE
if NEO4J_PASSWORD_OVERRIDE:
    os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD_OVERRIDE

_log(f"  NEO4J_URI      : {os.environ.get('NEO4J_URI', '(not set)')}")
_log("")


# ── Imports ──────────────────────────────────────────────────────────
from collections import Counter, defaultdict
from backend.database.neo4j_client import (
    get_neo4j_driver,
    enrich_chunk_entities,
)
from backend.ingestion.extractor.entities import (
    build_unified_prompt,
    parse_unified_response,
)
from backend.models.llm_factory import get_client
from backend.utils.text_utils import strip_thinking_tags

llm_client = get_client()
driver = get_neo4j_driver()

if not driver:
    _log("[FATAL] Khong ket noi duoc Neo4j.")
    sys.exit(1)

_log(f"[OK] Neo4j connected")
_log(f"[OK] DB Safety: MERGE/SET/MATCH only — no DELETE/DROP")
_log("")


# =====================================================================
# BUOC 1 — READ: Lay leaf nodes chua enrich tu Neo4j (read-only)
# =====================================================================
_FETCH_Q = """
MATCH (leaf)
WHERE leaf.qdrant_id IS NOT NULL
  AND leaf.text IS NOT NULL
  AND leaf.text <> ''
  AND ($resume = false OR leaf.enriched_v2 IS NULL OR leaf.enriched_v2 = false)
OPTIONAL MATCH (leaf)-[:PART_OF|BELONGS_TO*1..4]->(d:Document)
WITH leaf, d
ORDER BY leaf.qdrant_id
RETURN DISTINCT
  leaf.qdrant_id                          AS qdrant_id,
  leaf.text                               AS text,
  COALESCE(d.document_number, 'UNKNOWN')  AS doc_number,
  labels(leaf)[0]                         AS node_label
"""

_log("=" * 65)
_log("BUOC 1: Query Neo4j — lay leaf nodes chua enrich")
_log("=" * 65)

t0 = time.perf_counter()
with driver.session() as sess:
    _rows_raw = [dict(r) for r in sess.run(_FETCH_Q, resume=RESUME)]
_log(f"  Tim thay: {len(_rows_raw):,} nodes ({time.perf_counter()-t0:.1f}s)")

for lbl, cnt in Counter(r.get("node_label", "?") for r in _rows_raw).most_common():
    _log(f"    {lbl:20s}: {cnt:,}")

if not _rows_raw:
    _log("[DONE] Khong co node nao can enrich.")
    driver.close()
    sys.exit(0)

if MAX_CHUNKS:
    _rows_raw = _rows_raw[:MAX_CHUNKS]
    _log(f"  [TEST] Gioi han: {len(_rows_raw):,} chunks")


# =====================================================================
# BUOC 2 — CHECKPOINT: Load + filter done_ids
# =====================================================================
_log("")
_log("=" * 65)
_log("BUOC 2: Checkpoint — loc chunk da xu ly")
_log("=" * 65)

done_ids: set = set()
if RESUME and os.path.exists(CHECKPOINT_FILE):
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            done_ids = pickle.load(f)
        _log(f"  Checkpoint loaded: {len(done_ids):,} IDs tu truoc")
    except Exception as e:
        _log(f"  [WARN] Khong doc duoc checkpoint: {e}")

rows = [r for r in _rows_raw if r["qdrant_id"] not in done_ids]
_log(f"  Con lai can xu ly: {len(rows):,} chunks")

if not rows:
    _log("[DONE] Tat ca da enriched.")
    driver.close()
    sys.exit(0)


# =====================================================================
# BUOC 3 — GROUP theo doc_number
# =====================================================================
# Pipeline goc (relations.py 669-696) accumulate entities theo s_doc.
# Group de dam bao moi doc duoc xu ly rieng, entities khong bi tron.

_log("")
_log("=" * 65)
_log("BUOC 3: Group chunks theo doc_number")
_log("=" * 65)

docs_map: dict = defaultdict(list)
for r in rows:
    docs_map[r["doc_number"]].append(r)

doc_queue = list(docs_map.items())  # [(doc_number, [rows])]
_log(f"  {len(rows):,} chunks tu {len(doc_queue):,} documents")
_log("")


# =====================================================================
# HELPERS
# =====================================================================
_ckpt_lock = threading.Lock()


def _save_checkpoint():
    """Luu checkpoint an toan (thread-safe)."""
    with _ckpt_lock:
        try:
            with open(CHECKPOINT_FILE, "wb") as f:
                pickle.dump(done_ids, f)
        except Exception as e:
            _log(f"  [WARN] Checkpoint save failed: {e}")


def _flush_to_neo4j(params: list):
    """
    Ghi truc tiep vao Neo4j bang MERGE (khong DELETE).

    Goi enrich_chunk_entities(use_apoc=False) -> _enrich_fallback():
      - Per label:   MERGE (ent:Label {name}) + MERGE (leaf)-[:HAS_ENTITY]->(ent)
      - Per rel key: MERGE (src) + MERGE (tgt) + MERGE (src)-[r:REL]->(tgt)
      - _MARK_ENRICHED_QUERY: SET c.enriched_v2 = true

    Tat ca deu idempotent — chay nhieu lan khong duplicate.
    """
    if not params:
        return
    if DRY_RUN:
        n_e = sum(sum(len(v) for v in p.get("entities", {}).values()) for p in params)
        n_r = sum(len(p.get("node_relations", [])) for p in params)
        _log(f"    [DRY_RUN] skip {len(params)} chunks ({n_e} ents, {n_r} nrels)")
        return
    try:
        enrich_chunk_entities(driver, params, use_apoc=False)
    except Exception as e:
        _log(f"    [ERROR] enrich_chunk_entities: {e}")


def _accumulate_entities(accumulator: dict, new_entities: dict):
    """
    Merge new_entities vao accumulator, dedup EXACT (case-sensitive) per type.
    Giong y het relations.py dong 676-685:
        existing = set(entities_side_data[s_doc]["entities"][etype])
        for v in vals:
            if v not in existing:
                entities_side_data[s_doc]["entities"][etype].append(v)
                existing.add(v)
    """
    for etype, vals in new_entities.items():
        if not isinstance(vals, list):
            continue
        if etype not in accumulator:
            accumulator[etype] = []
        existing = set(accumulator[etype])
        for v in vals:
            if v not in existing:
                accumulator[etype].append(v)
                existing.add(v)


def _accumulate_nrels(accumulator: list, seen: set, new_nrels: list):
    """
    Merge new_nrels vao accumulator, dedup bang tuple key.
    Giong y het relations.py dong 687-696:
        existing_nrels = {(nr.source_node, nr.target_node, nr.relationship) for ...}
        if nr_key not in existing_nrels:
            entities_side_data[s_doc]["node_relations"].append(nr)
    """
    for nr in new_nrels:
        key = (
            nr.get("source_node", ""),
            nr.get("target_node", ""),
            nr.get("relationship", ""),
        )
        if key not in seen:
            accumulator.append(nr)
            seen.add(key)


# =====================================================================
# BUOC 4 — MAIN LOOP: LLM call -> parse -> accumulate -> flush ngay
# =====================================================================
# Flow cho MOI document:
#   doc_rows = [chunk1, chunk2, ..., chunkN]  (da group tu buoc 3)
#   doc_entities = {}   # accumulator entities (nhu entities_side_data[s_doc])
#   doc_nrels    = []   # accumulator node_relations
#
#   for batch in split(doc_rows, BATCH_SIZE_LLM):
#       prompt   = build_unified_prompt(batch_info)
#       response = llm_client.chat_completion(prompt)
#       parsed   = parse_unified_response(response)
#           -> _normalize_entity_type()
#           -> _normalize_entity_name()
#           -> _dedup_entity_values()
#           -> _normalize_relationship()
#           -> filter structural labels
#           -> redirect via entity_alias_map
#       _accumulate_entities(doc_entities, parsed["entities"])
#       _accumulate_nrels(doc_nrels, doc_nrel_seen, parsed["node_relations"])
#
#       >>> FLUSH NGAY <<<
#       enrich_params = [{qdrant_id, entities=doc_entities, node_relations=doc_nrels}
#                        for row in batch]
#       enrich_chunk_entities(driver, enrich_params)
#           -> _enrich_fallback():
#              MERGE entity nodes + HAS_ENTITY edges + node_relation edges
#              SET enriched_v2 = true
#
#       >>> CHECKPOINT <<<
#       done_ids.update(batch qdrant_ids)
#       _save_checkpoint()

_log("=" * 65)
_log("BUOC 4: LLM extraction -> flush Neo4j ngay sau moi batch")
_log("=" * 65)
_log("")

stats = {
    "docs_done"      : 0,
    "chunks_done"    : 0,
    "llm_calls"      : 0,
    "llm_errors"     : 0,
    "empty_responses": 0,
    "total_entities" : 0,
    "total_nrels"    : 0,
    "entity_counts"  : Counter(),
    "nrel_counts"    : Counter(),
}
t_global = time.perf_counter()


for doc_idx, (doc_number, doc_rows) in enumerate(doc_queue):

    # Chia chunks cua doc nay thanh LLM batches
    batches = [
        doc_rows[i : i + BATCH_SIZE_LLM]
        for i in range(0, len(doc_rows), BATCH_SIZE_LLM)
    ]

    # Accumulator per-doc (nhu entities_side_data[s_doc] trong relations.py)
    doc_entities  : dict = {}    # {EntityType: [name, ...]}
    doc_nrels     : list = []    # [{source_node, source_type, ...}]
    doc_nrel_seen : set  = set() # (src, tgt, rel) da co

    for b_idx, batch in enumerate(batches):

        # ── A. Build prompt (giong relations.py dong 554) ────────────
        batch_info = [
            {"s_doc": doc_number, "context": row["text"][:3000]}
            for row in batch
        ]
        prompt = build_unified_prompt(batch_info)
        stats["llm_calls"] += 1

        # ── B. Call LLM ──────────────────────────────────────────────
        try:
            resp = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=5000,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            _log(f"  [LLM ERROR] doc={doc_number} batch={b_idx+1}: {e}")
            stats["llm_errors"] += 1
            # Danh dau done de khong retry vo han
            for row in batch:
                done_ids.add(row["qdrant_id"])
            _save_checkpoint()
            continue

        if not resp or not resp.strip():
            stats["empty_responses"] += 1
            for row in batch:
                done_ids.add(row["qdrant_id"])
            _save_checkpoint()
            continue

        # ── C. Parse (giong relations.py dong 571) ───────────────────
        # parse_unified_response() da lam tat ca:
        #   _normalize_entity_type()
        #   _normalize_entity_name()
        #   _dedup_entity_values() + entity_alias_map
        #   _normalize_relationship()
        #   filter structural labels (Document/LegalArticle/...)
        #   redirect node_relation endpoints qua alias_map
        resp = strip_thinking_tags(resp)
        parsed = parse_unified_response(resp)
        new_ents  = parsed.get("entities", {})
        new_nrels = parsed.get("node_relations", [])

        # ── D. Accumulate per-doc (giong relations.py dong 676-696) ──
        _accumulate_entities(doc_entities, new_ents)
        _accumulate_nrels(doc_nrels, doc_nrel_seen, new_nrels)

        # ── E. FLUSH NGAY vao Neo4j (giong chunking_embedding.py 446-462) ──
        # Moi chunk trong batch nay nhan TOAN BO entities tich luy
        # cua doc tinh den thoi diem nay.
        # MERGE la idempotent nen chunk cu da co HAS_ENTITY se khong bi dup.
        enrich_params = []
        for row in batch:
            if doc_entities or doc_nrels:
                enrich_params.append({
                    "qdrant_id"     : row["qdrant_id"],
                    "entities"      : doc_entities,      # full accumulated
                    "node_relations": doc_nrels,          # full accumulated
                })

        _flush_to_neo4j(enrich_params)

        # ── F. CHECKPOINT per-batch ──────────────────────────────────
        for row in batch:
            done_ids.add(row["qdrant_id"])
            stats["chunks_done"] += 1

        _save_checkpoint()

    # ── Thong ke per-doc ─────────────────────────────────────────────
    stats["docs_done"] += 1
    n_ents_doc = sum(len(v) for v in doc_entities.values())
    for etype, vals in doc_entities.items():
        stats["entity_counts"][etype] += len(vals)
    stats["total_entities"] += n_ents_doc
    for nr in doc_nrels:
        stats["nrel_counts"][nr.get("relationship", "?")] += 1
    stats["total_nrels"] += len(doc_nrels)

    # ── Progress ─────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_global
    pct = stats["docs_done"] / len(doc_queue) * 100
    remain = len(doc_queue) - stats["docs_done"]
    eta_s = (elapsed / stats["docs_done"] * remain) if stats["docs_done"] > 0 else 0
    eta = str(datetime.timedelta(seconds=int(eta_s)))

    _log(
        f"  [{doc_idx+1:>5}/{len(doc_queue)}]"
        f"  {pct:5.1f}%"
        f"  doc={doc_number:<28s}"
        f"  chunks={len(doc_rows)}"
        f"  ents={n_ents_doc}"
        f"  nrels={len(doc_nrels)}"
        f"  ETA={eta}"
    )


# =====================================================================
# BUOC 5 — FINAL REPORT
# =====================================================================
driver.close()
t_total = time.perf_counter() - t_global

_log("")
_log("=" * 65)
_log(f"  HOAN TAT: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
_log("=" * 65)
_log(f"  Thoi gian tong    : {datetime.timedelta(seconds=int(t_total))}")
_log(f"  Documents         : {stats['docs_done']:,}")
_log(f"  Chunks            : {stats['chunks_done']:,}")
_log(f"  LLM calls         : {stats['llm_calls']:,}")
_log(f"  LLM errors        : {stats['llm_errors']:,}")
_log(f"  Empty responses   : {stats['empty_responses']:,}")
_log(f"  Total Entities    : {stats['total_entities']:,}")
_log(f"  Total Node Rels   : {stats['total_nrels']:,}")
_log(f"  DRY_RUN           : {DRY_RUN}")

_log("\n  Entity types (top 15):")
for et, cnt in stats["entity_counts"].most_common(15):
    _log(f"    {et:28s}: {cnt:,}")

_log("\n  Node Relations (top 15):")
for rel, cnt in stats["nrel_counts"].most_common(15):
    _log(f"    {rel:35s}: {cnt:,}")

_log(f"\n  Log       : {_LOG_FILE}")
_log(f"  Checkpoint: {CHECKPOINT_FILE}")

_log("""
=================================================================
CYPHER VERIFY — Chay trong Neo4j Browser:
=================================================================
// 1. Entity nodes
MATCH (n)
WHERE n:Organization OR n:Person OR n:Location
   OR n:Procedure OR n:Condition OR n:Fee
   OR n:Penalty OR n:Timeframe OR n:Role OR n:Concept
RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC;

// 2. HAS_ENTITY edges
MATCH ()-[r:HAS_ENTITY]->() RETURN count(r);

// 3. Node relation edges
MATCH ()-[r]->()
WHERE type(r) IN ['ISSUED_BY','SIGNED_BY','MANAGED_BY',
  'IMPLEMENTED_BY','REGULATED_BY','APPROVED_BY','AFFECTED_BY',
  'COMPLIES_WITH','SUBMITTED_TO','PERMITTED_TO','FUNDED_BY']
RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC;

// 4. enriched_v2 count
MATCH (n) WHERE n.enriched_v2 = true RETURN count(n);

// 5. Tong Document KHONG bi mat
MATCH (d:Document) RETURN count(d) AS total_docs;
""")
