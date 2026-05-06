# -*- coding: utf-8 -*-
"""
re_enrich_entities.py
=====================
Bo sung Entities + Node Relations vao Neo4j da co san.

TOAN BO CHI DUNG MERGE/SET/MATCH — KHONG CO DELETE/DROP.

FLUSH STRATEGY (v2 — BATCH):
  Gom SUPER_BATCH_DOCS doc mot luc, xay tat ca prompts → goi
  batch_chat_completion() (song song LLM_PARALLEL_WORKERS threads)
  → parse tung response → accumulate per-doc → flush Neo4j ngay.

  Ket qua: giam ETA tu ~8 ngay (tuan tu) xuong ~1 ngay (8 parallel).

CHECKPOINT:
  Luu done_ids sau moi SUPER_BATCH flush thanh cong.
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
BATCH_SIZE_LLM   = 8      # So chunk / 1 LLM prompt (giong pipeline goc)
SUPER_BATCH_DOCS = 8      # So doc gom vao 1 lan goi batch_chat_completion
PAGE_SIZE        = 50000  # So chunk moi lan fetch tu Neo4j (tranh day RAM)
DRY_RUN          = False   # True = khong ghi DB
RESUME           = True    # True = skip chunk da enriched_v2
MAX_CHUNKS       = None    # None = tat ca. Dat so de test (VD: 500)

NEO4J_URI_OVERRIDE      = None
NEO4J_USERNAME_OVERRIDE = None
NEO4J_PASSWORD_OVERRIDE = None

CHECKPOINT_FILE = ".checkpoints/re_enrich_done_ids.pkl"
# =====================================================================


# ── Logging UTF-8 ────────────────────────────────────────────────────
os.makedirs(".debug", exist_ok=True)
os.makedirs(".checkpoints", exist_ok=True)

_LOG_FILE = os.path.join(
    ".debug",
    f"re_enrich_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)


class _DualWriter:
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
_log("  RE-ENRICH ENTITIES v2 (Batch Parallel)")
_log(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
_log("=" * 65)
_log(f"  Log              : {_LOG_FILE}")
_log(f"  Checkpoint       : {CHECKPOINT_FILE}")
_log(f"  DRY_RUN          : {DRY_RUN}")
_log(f"  RESUME           : {RESUME}")
_log(f"  BATCH_SIZE_LLM   : {BATCH_SIZE_LLM} chunks/prompt")
_log(f"  SUPER_BATCH_DOCS : {SUPER_BATCH_DOCS} docs/parallel-call")
_log(f"  MAX_CHUNKS       : {MAX_CHUNKS or 'ALL'}")
_log("")


# ── sys.path + .env ──────────────────────────────────────────────────
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_repo, ".env"), override=False)
    _log("  .env             : loaded")
except ImportError:
    _log("  .env             : (dotenv not installed)")

if NEO4J_URI_OVERRIDE:
    os.environ["NEO4J_URI"] = NEO4J_URI_OVERRIDE
if NEO4J_USERNAME_OVERRIDE:
    os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME_OVERRIDE
if NEO4J_PASSWORD_OVERRIDE:
    os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD_OVERRIDE

_log(f"  NEO4J_URI        : {os.environ.get('NEO4J_URI', '(not set)')}")
_log(f"  LLM_PARALLEL_WORKERS: {os.environ.get('LLM_PARALLEL_WORKERS', '8 (default)')}")
_log("")


# ── Imports ──────────────────────────────────────────────────────────
from collections import Counter, defaultdict
from backend.database.neo4j_client import get_neo4j_driver, enrich_chunk_entities
from backend.ingestion.extractor.entities import build_unified_prompt, parse_unified_response
from backend.models.llm_factory import get_client
from backend.utils.text_utils import strip_thinking_tags

llm_client = get_client()
driver = get_neo4j_driver()

if not driver:
    _log("[FATAL] Khong ket noi duoc Neo4j.")
    sys.exit(1)

_log("[OK] Neo4j connected")
_log("[OK] DB Safety: MERGE/SET/MATCH only — no DELETE/DROP")
_log("")


# =====================================================================
# QUERY TEMPLATE — fetch theo trang, KHONG ORDER BY (nhanh hon 10x)
# Moi lan LIMIT PAGE_SIZE chunk, dung enriched_v2 lam con tro tu nhien:
#   Sau khi SET enriched_v2=true, lan query ke tiep tu dong bo qua.
# SKIP/LIMIT khong co ORDER BY tren Neo4j = streaming nhanh, it RAM.
# =====================================================================
# SKIP bi bo: enriched_v2 lam cursor tu nhien
# Moi trang: query tat ca node chua enrich → LIMIT PAGE_SIZE
# Sau khi flush, node co enriched_v2=true → trang sau tu dong bo qua
_FETCH_PAGE_Q = """
MATCH (leaf)
WHERE leaf.qdrant_id IS NOT NULL
  AND leaf.text IS NOT NULL
  AND leaf.text <> ''
  AND ($resume = false OR leaf.enriched_v2 IS NULL OR leaf.enriched_v2 = false)
OPTIONAL MATCH (leaf)-[:PART_OF|BELONGS_TO*1..4]->(d:Document)
WITH leaf, d
LIMIT $limit
RETURN DISTINCT
  leaf.qdrant_id                          AS qdrant_id,
  leaf.text                               AS text,
  COALESCE(d.document_number, 'UNKNOWN')  AS doc_number,
  labels(leaf)[0]                         AS node_label
"""

# =====================================================================
# BUOC 1+2 — CHECKPOINT load
# =====================================================================
_log("=" * 65)
_log("BUOC 1: Checkpoint load")
_log("=" * 65)

done_ids: set = set()
if RESUME and os.path.exists(CHECKPOINT_FILE):
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            done_ids = pickle.load(f)
        _log(f"  Checkpoint loaded: {len(done_ids):,} IDs tu truoc")
    except Exception as e:
        _log(f"  [WARN] Khong doc duoc checkpoint: {e}")
else:
    _log("  Checkpoint: trong (lan dau chay)")
_log("")



# =====================================================================
# HELPERS
# =====================================================================
_ckpt_lock = threading.Lock()


def _save_checkpoint():
    with _ckpt_lock:
        try:
            with open(CHECKPOINT_FILE, "wb") as f:
                pickle.dump(done_ids, f)
        except Exception as e:
            _log(f"  [WARN] Checkpoint save failed: {e}")


def _flush_to_neo4j(params: list):
    """MERGE entities + node_relations vao Neo4j (idempotent)."""
    if not params:
        return
    if DRY_RUN:
        n_e = sum(sum(len(v) for v in p.get("entities", {}).values()) for p in params)
        n_r = sum(len(p.get("node_relations", [])) for p in params)
        _log(f"    [DRY_RUN] skip {len(params)} params ({n_e} ents, {n_r} nrels)")
        return
    try:
        enrich_chunk_entities(driver, params, use_apoc=False)
    except Exception as e:
        _log(f"    [ERROR] enrich_chunk_entities: {e}")


def _accumulate_entities(accumulator: dict, new_entities: dict):
    """Dedup exact per entity type (giong relations.py dong 676-685)."""
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
    """Dedup bang tuple key (giong relations.py dong 687-696)."""
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
# BUOC 2: Super-Batch LLM (parallel) -> flush Neo4j
#
# Cau truc vong lap chinh (paginated):
#   PAGE_LOOP:
#     fetch PAGE_SIZE chunk tu Neo4j (SKIP offset, LIMIT PAGE_SIZE)
#     loc qua done_ids (checkpoint)
#     group theo doc_number
#     SUPER_BATCH_LOOP (8 doc / lan):
#       xay messages_list (tat ca prompts cua 8 doc)
#       batch_chat_completion() song song
#       parse + accumulate per-doc
#       flush Neo4j
#       checkpoint
#     offset += PAGE_SIZE
#   stop khi page tra ve 0 row
# =====================================================================
_log("=" * 65)
_log("BUOC 2: Paginated fetch + Super-Batch LLM -> flush Neo4j")
_log("=" * 65)
_log(f"  PAGE_SIZE        : {PAGE_SIZE:,} chunks/fetch")
_log(f"  SUPER_BATCH_DOCS : {SUPER_BATCH_DOCS} docs/parallel-call")
_log(f"  BATCH_SIZE_LLM   : {BATCH_SIZE_LLM} chunks/prompt")
_log("")

stats = {
    "pages_done"     : 0,
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

total_fetched = 0          # Tong so chunk da fetch
global_done   = False      # Co tat ca chua

while not global_done:

    # ── FETCH 1 TRANG tu Neo4j (LUON tu offset=0, WHERE lam cursor) ──
    fetch_limit = PAGE_SIZE
    if MAX_CHUNKS:
        remaining_budget = MAX_CHUNKS - total_fetched
        if remaining_budget <= 0:
            break
        fetch_limit = min(PAGE_SIZE, remaining_budget)

    t_fetch = time.perf_counter()
    with driver.session() as sess:
        page_rows = [
            dict(r) for r in sess.run(
                _FETCH_PAGE_Q,
                resume=RESUME,
                limit=fetch_limit,
            )
        ]

    if not page_rows:
        _log(f"  [PAGE {stats['pages_done']+1}] Khong con chunk nao. Hoan tat.")
        break

    # Loc checkpoint
    page_rows = [r for r in page_rows if r["qdrant_id"] not in done_ids]
    if not page_rows:
        stats["pages_done"] += 1
        _log(f"  [PAGE {stats['pages_done']}] Tat ca da trong checkpoint, skip.")
        continue

    total_fetched += len(page_rows)
    t_fetch_elapsed = time.perf_counter() - t_fetch

    # Group theo doc_number
    docs_map: dict = defaultdict(list)
    for r in page_rows:
        docs_map[r["doc_number"]].append(r)
    doc_queue = list(docs_map.items())

    _log(
        f"  [PAGE {stats['pages_done']+1}]"
        f"  fetch={len(page_rows):,} chunks"
        f"  ({t_fetch_elapsed:.1f}s)"
        f"  → {len(doc_queue)} docs"
        f"  RAM-safe: ~{len(page_rows)*1//1024}MB est."
    )

    # ── SUPER-BATCH: chia doc_queue thanh nhom SUPER_BATCH_DOCS ──────
    super_batches = [
        doc_queue[i : i + SUPER_BATCH_DOCS]
        for i in range(0, len(doc_queue), SUPER_BATCH_DOCS)
    ]

    for sb_idx, super_batch in enumerate(super_batches):

        # A. Xay messages_list cho ca super-batch
        messages_list  = []
        prompt_idx_map = {}
        doc_batch_plan = []

        for doc_i, (doc_number, doc_rows) in enumerate(super_batch):
            batches = [
                doc_rows[i : i + BATCH_SIZE_LLM]
                for i in range(0, len(doc_rows), BATCH_SIZE_LLM)
            ]
            doc_batch_plan.append((doc_number, doc_rows, batches))

            for batch_j, batch in enumerate(batches):
                batch_info = [
                    {"s_doc": doc_number, "context": row["text"][:3000]}
                    for row in batch
                ]
                prompt = build_unified_prompt(batch_info)
                idx = len(messages_list)
                messages_list.append([{"role": "user", "content": prompt}])
                prompt_idx_map[(doc_i, batch_j)] = idx

        n_prompts = len(messages_list)
        stats["llm_calls"] += n_prompts

        # B. Goi batch_chat_completion SONG SONG
        _log(
            f"    [LLM] P{stats['pages_done']+1} SB{sb_idx+1:>3}/{len(super_batches)}"
            f"  →  {n_prompts} prompts ({len(super_batch)} docs)  ..."
        )
        t_llm = time.perf_counter()
        try:
            responses = llm_client.batch_chat_completion(
                messages_list=messages_list,
                temperature=0.1,
                max_tokens=5000,
                response_format={"type": "json_object"},
            )
            t_llm_elapsed = time.perf_counter() - t_llm
            n_ok = sum(1 for r in responses if r and r.strip())
            _log(
                f"    [LLM] done  {t_llm_elapsed:.1f}s"
                f"  avg={t_llm_elapsed/max(1,n_prompts):.1f}s/prompt"
                f"  ok={n_ok}/{n_prompts}"
            )
        except Exception as e:
            _log(f"  [LLM BATCH ERROR] page={stats['pages_done']+1} sb={sb_idx+1}: {e}")
            stats["llm_errors"] += n_prompts
            for _, doc_rows, _ in doc_batch_plan:
                for row in doc_rows:
                    done_ids.add(row["qdrant_id"])
            _save_checkpoint()
            continue

        # C. Parse + accumulate + flush per-doc
        for doc_i, (doc_number, doc_rows, batches) in enumerate(doc_batch_plan):
            doc_entities  : dict = {}
            doc_nrels     : list = []
            doc_nrel_seen : set  = set()

            for batch_j, batch in enumerate(batches):
                idx  = prompt_idx_map.get((doc_i, batch_j))
                resp = responses[idx] if idx is not None and idx < len(responses) else ""

                if not resp or not resp.strip():
                    stats["empty_responses"] += 1
                    continue

                resp = strip_thinking_tags(resp)
                try:
                    parsed = parse_unified_response(resp)
                except Exception as e:
                    _log(f"    [PARSE ERR] doc={doc_number} b={batch_j}: {e}")
                    continue

                _accumulate_entities(doc_entities, parsed.get("entities", {}))
                _accumulate_nrels(doc_nrels, doc_nrel_seen, parsed.get("node_relations", []))

            # D. Flush Neo4j
            if doc_entities or doc_nrels:
                enrich_params = [
                    {
                        "qdrant_id"     : row["qdrant_id"],
                        "entities"      : doc_entities,
                        "node_relations": doc_nrels,
                    }
                    for row in doc_rows
                ]
                _flush_to_neo4j(enrich_params)
                _log(f"      [FLUSH] doc={doc_number} → {sum(len(v) for v in doc_entities.values())} ents, {len(doc_nrels)} rels")

            # E. Stats
            n_ents_doc = sum(len(v) for v in doc_entities.values())
            stats["total_entities"] += n_ents_doc
            stats["total_nrels"]    += len(doc_nrels)
            stats["docs_done"]      += 1
            stats["chunks_done"]    += len(doc_rows)

            for etype, vals in doc_entities.items():
                stats["entity_counts"][etype] += len(vals)
            for nr in doc_nrels:
                stats["nrel_counts"][nr.get("relationship", "?")] += 1

            for row in doc_rows:
                done_ids.add(row["qdrant_id"])

        # F. Checkpoint sau moi super-batch
        _save_checkpoint()

        # G. Progress
        elapsed = time.perf_counter() - t_global
        speed   = stats["docs_done"] / elapsed * 3600 if elapsed > 0 else 0
        # ETA dua tren toc do hien tai va so chunk con lai uoc tinh
        approx_remain_docs = max(0, (672138 - total_fetched) // max(1, len(page_rows) // len(doc_queue)))
        eta_s = (elapsed / stats["docs_done"] * (stats["docs_done"] + approx_remain_docs)) if stats["docs_done"] > 0 else 0
        eta   = str(datetime.timedelta(seconds=int(eta_s - elapsed))) if eta_s > elapsed else "?"

        _log(
            f"  P{stats['pages_done']+1}"
            f"  SB{sb_idx+1:>3}/{len(super_batches)}"
            f"  docs={stats['docs_done']:>6}"
            f"  ents={stats['total_entities']:>7}"
            f"  nrels={stats['total_nrels']:>6}"
            f"  ETA~{eta}"
            f"  ({speed:.0f} doc/h)"
        )

    # Ket thuc 1 trang
    stats["pages_done"] += 1
    # KHONG tang page_offset — enriched_v2 lam cursor tu nhien
    # Dung khi page tra ve 0 row (tat ca da enrich)
    if len(page_rows) < fetch_limit:
        _log(f"  [PAGE {stats['pages_done']}] Trang cuoi (< PAGE_SIZE), ket thuc.")
        break


# =====================================================================
# BUOC 3 — FINAL REPORT
# =====================================================================
driver.close()
t_total = time.perf_counter() - t_global

_log("")
_log("=" * 65)
_log(f"  HOAN TAT: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
_log("=" * 65)
_log(f"  Thoi gian tong    : {datetime.timedelta(seconds=int(t_total))}")
_log(f"  Pages fetched     : {stats['pages_done']:,}")
_log(f"  Documents         : {stats['docs_done']:,}")
_log(f"  Chunks            : {stats['chunks_done']:,}")
_log(f"  LLM prompts       : {stats['llm_calls']:,}")
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
CYPHER VERIFY:
  MATCH (n) WHERE n:Organization OR n:Person RETURN count(n);
  MATCH ()-[r:HAS_ENTITY]->() RETURN count(r);
  MATCH (n) WHERE n.enriched_v2 = true RETURN count(n);
  MATCH (d:Document) RETURN count(d) AS total_docs;
""")

_log("=" * 65)
_log("BUOC 4: Super-Batch LLM (parallel) -> flush Neo4j")
_log("=" * 65)
_log(f"  Song song: {SUPER_BATCH_DOCS} docs x ~{max(1, 672138//18140//BATCH_SIZE_LLM+1)} prompts/doc")
_log("")

from collections import Counter

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

# Chia doc_queue thanh cac super-batch
super_batches = [
    doc_queue[i : i + SUPER_BATCH_DOCS]
    for i in range(0, len(doc_queue), SUPER_BATCH_DOCS)
]

for sb_idx, super_batch in enumerate(super_batches):

    # ── A. Xay danh sach prompts cho toan bo super-batch ─────────────
    # Cau truc: prompt_index_map[(doc_i, batch_j)] = idx trong messages_list
    messages_list   = []   # list of [{"role":"user","content":prompt}]
    prompt_idx_map  = {}   # (doc_i, batch_j) → index trong messages_list
    doc_batch_plan  = []   # [(doc_number, doc_rows, [batch0, batch1, ...])]

    for doc_i, (doc_number, doc_rows) in enumerate(super_batch):
        batches = [
            doc_rows[i : i + BATCH_SIZE_LLM]
            for i in range(0, len(doc_rows), BATCH_SIZE_LLM)
        ]
        doc_batch_plan.append((doc_number, doc_rows, batches))

        for batch_j, batch in enumerate(batches):
            batch_info = [
                {"s_doc": doc_number, "context": row["text"][:3000]}
                for row in batch
            ]
            prompt = build_unified_prompt(batch_info)
            idx = len(messages_list)
            messages_list.append([{"role": "user", "content": prompt}])
            prompt_idx_map[(doc_i, batch_j)] = idx

    n_prompts = len(messages_list)
    stats["llm_calls"] += n_prompts

    # ── B. Goi batch_chat_completion (SONG SONG) ──────────────────────
    try:
        responses = llm_client.batch_chat_completion(
            messages_list=messages_list,
            temperature=0.1,
            max_tokens=5000,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        _log(f"  [LLM BATCH ERROR] super_batch={sb_idx+1}: {e}")
        stats["llm_errors"] += n_prompts
        # Danh dau tat ca chunk trong super-batch la done de khong retry
        for _, doc_rows, _ in doc_batch_plan:
            for row in doc_rows:
                done_ids.add(row["qdrant_id"])
        _save_checkpoint()
        continue

    # ── C. Phan phoi response, parse + accumulate per-doc ─────────────
    for doc_i, (doc_number, doc_rows, batches) in enumerate(doc_batch_plan):
        doc_entities  : dict = {}
        doc_nrels     : list = []
        doc_nrel_seen : set  = set()

        for batch_j, batch in enumerate(batches):
            idx = prompt_idx_map.get((doc_i, batch_j))
            resp = responses[idx] if idx is not None and idx < len(responses) else ""

            if not resp or not resp.strip():
                stats["empty_responses"] += 1
                continue

            resp = strip_thinking_tags(resp)
            try:
                parsed = parse_unified_response(resp)
            except Exception as e:
                _log(f"    [PARSE ERROR] doc={doc_number} batch={batch_j}: {e}")
                continue

            _accumulate_entities(doc_entities, parsed.get("entities", {}))
            _accumulate_nrels(doc_nrels, doc_nrel_seen, parsed.get("node_relations", []))

        # ── D. FLUSH neo4j cho tung doc (sau khi co du du lieu) ───────
        if doc_entities or doc_nrels:
            enrich_params = [
                {
                    "qdrant_id"     : row["qdrant_id"],
                    "entities"      : doc_entities,
                    "node_relations": doc_nrels,
                }
                for row in doc_rows
            ]
            _flush_to_neo4j(enrich_params)

        # ── E. Cap nhat stats ──────────────────────────────────────────
        n_ents_doc = sum(len(v) for v in doc_entities.values())
        stats["total_entities"] += n_ents_doc
        stats["total_nrels"]    += len(doc_nrels)
        stats["docs_done"]      += 1
        stats["chunks_done"]    += len(doc_rows)

        for etype, vals in doc_entities.items():
            stats["entity_counts"][etype] += len(vals)
        for nr in doc_nrels:
            stats["nrel_counts"][nr.get("relationship", "?")] += 1

        # Mark done
        for row in doc_rows:
            done_ids.add(row["qdrant_id"])

    # ── F. CHECKPOINT sau moi super-batch ────────────────────────────
    _save_checkpoint()

    # ── G. Progress ──────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_global
    pct     = stats["docs_done"] / len(doc_queue) * 100
    remain  = len(doc_queue) - stats["docs_done"]
    eta_s   = (elapsed / stats["docs_done"] * remain) if stats["docs_done"] > 0 else 0
    eta     = str(datetime.timedelta(seconds=int(eta_s)))
    speed   = stats["docs_done"] / elapsed * 3600 if elapsed > 0 else 0

    # In 1 dong sau moi super-batch (thay vi sau tung doc)
    last_doc = super_batch[-1][0]
    _log(
        f"  SB {sb_idx+1:>5}/{len(super_batches)}"
        f"  {pct:5.1f}%"
        f"  docs={stats['docs_done']:>6}/{len(doc_queue)}"
        f"  prompts={n_prompts}"
        f"  ents={stats['total_entities']:>7}"
        f"  ETA={eta}"
        f"  ({speed:.0f} docs/h)"
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
_log(f"  LLM prompts       : {stats['llm_calls']:,}")
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
CYPHER VERIFY:
  MATCH (n) WHERE n:Organization OR n:Person RETURN count(n);
  MATCH ()-[r:HAS_ENTITY]->() RETURN count(r);
  MATCH (n) WHERE n.enriched_v2 = true RETURN count(n);
  MATCH (d:Document) RETURN count(d) AS total_docs;
""")
