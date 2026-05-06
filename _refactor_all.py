"""
refactor_all.py — Refactor toàn bộ các file còn lại:
- backend/agent/utils_legal.py        : remove _VI_TRANSLATION_MAP, add import
- backend/models/llm_client.py        : remove _ICLLM_CONFIG + _JSON_ENFORCEMENT_PROMPT, add import
- backend/database/neo4j_client.py    : remove _ENTITY_LABELS, add import
- utils/migrate_neo4j_relations.py    : remove all constants, add sys.path + import
- utils/debug_neo4j_relations.py      : remove all constants, add sys.path + import
"""
import re
import sys

def refactor(path, replacements):
    """replacements: list of (old_str, new_str). old_str=None means append."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    original = text
    for old, new in replacements:
        if old is None:
            text = new + text
        else:
            if old not in text:
                print(f"  ⚠  Pattern not found in {path}: {old[:60]!r}...")
            text = text.replace(old, new, 1)
    if text == original:
        print(f"  ⏭  No changes: {path}")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  ✅  Written: {path}")


# ─────────────────────────────────────────────────────────────
# 1.  utils_legal.py  — remove _VI_TRANSLATION_MAP
# ─────────────────────────────────────────────────────────────
print("\n[1] utils_legal.py")
with open("backend/agent/utils_legal.py", "r", encoding="utf-8") as f:
    ul_text = f.read()

# Find the block from _VI_TRANSLATION_MAP = { to the closing }
vi_map_pat = re.compile(r"_VI_TRANSLATION_MAP = \{.*?\n\}\n", re.DOTALL)
match = vi_map_pat.search(ul_text)
if match:
    ul_text = ul_text[:match.start()] + "from backend.config import _VI_TRANSLATION_MAP\n" + ul_text[match.end():]
    with open("backend/agent/utils_legal.py", "w", encoding="utf-8") as f:
        f.write(ul_text)
    print("  ✅  _VI_TRANSLATION_MAP removed and import added")
else:
    print("  ⏭  _VI_TRANSLATION_MAP not found (already refactored?)")


# ─────────────────────────────────────────────────────────────
# 2.  llm_client.py  — remove _ICLLM_CONFIG + _JSON_ENFORCEMENT_PROMPT
# ─────────────────────────────────────────────────────────────
print("\n[2] llm_client.py")
with open("backend/models/llm_client.py", "r", encoding="utf-8") as f:
    lc_text = f.read()

# Remove the comment + _ICLLM_CONFIG block
icllm_pat = re.compile(
    r"# Thay đổi các thông số này.*?_ICLLM_CONFIG = \{.*?\}\n", re.DOTALL
)
enforcement_pat = re.compile(r"_JSON_ENFORCEMENT_PROMPT = \(.*?\)\n", re.DOTALL)

changed = False
if icllm_pat.search(lc_text):
    lc_text = icllm_pat.sub("", lc_text)
    changed = True
if enforcement_pat.search(lc_text):
    lc_text = enforcement_pat.sub("", lc_text)
    changed = True

if changed:
    # Insert import after the last existing import line block
    import_line = "from backend.config import _ICLLM_CONFIG, _JSON_ENFORCEMENT_PROMPT\n"
    # Find position after BaseLLMClient import
    insert_after = "from backend.models.interfaces import BaseLLMClient\n"
    lc_text = lc_text.replace(insert_after, insert_after + "\n" + import_line, 1)
    with open("backend/models/llm_client.py", "w", encoding="utf-8") as f:
        f.write(lc_text)
    print("  ✅  _ICLLM_CONFIG + _JSON_ENFORCEMENT_PROMPT removed and import added")
else:
    print("  ⏭  Nothing to change")


# ─────────────────────────────────────────────────────────────
# 3.  neo4j_client.py  — remove _ENTITY_LABELS
# ─────────────────────────────────────────────────────────────
print("\n[3] neo4j_client.py")
with open("backend/database/neo4j_client.py", "r", encoding="utf-8") as f:
    nc_text = f.read()

entity_labels_pat = re.compile(
    r"# Các nhãn entity hợp lệ.*?_ENTITY_LABELS = \[.*?\]\n", re.DOTALL
)
match = entity_labels_pat.search(nc_text)
if match:
    nc_text = nc_text[:match.start()] + "from backend.config import _ENTITY_LABELS\n" + nc_text[match.end():]
    with open("backend/database/neo4j_client.py", "w", encoding="utf-8") as f:
        f.write(nc_text)
    print("  ✅  _ENTITY_LABELS removed and import added")
else:
    print("  ⏭  _ENTITY_LABELS not found (already refactored?)")


# ─────────────────────────────────────────────────────────────
# 4.  migrate_neo4j_relations.py
# ─────────────────────────────────────────────────────────────
print("\n[4] migrate_neo4j_relations.py")
with open("utils/migrate_neo4j_relations.py", "r", encoding="utf-8") as f:
    mg_text = f.read()

# Remove blocks: CHUNKING_RELATIONS, FIXED_NODE_RELATIONS, BLACKLIST_RELATIONS, FIXED_DOC_RELATIONS, _VERB_ROOT_CANONICAL, _CROSS_VERB_MAPPING
blocks_to_remove = [
    re.compile(r"# ={20,}\n# DANH SÁCH RELATION TỪ CHUNKING.*?CHUNKING_RELATIONS = \{.*?\}\n", re.DOTALL),
    re.compile(r"# ={20,}\n# NORMALIZATION CONSTANTS.*?FIXED_NODE_RELATIONS = \{.*?\}\n", re.DOTALL),
    re.compile(r"BLACKLIST_RELATIONS = \{.*?\}\n# Regex.*?\n", re.DOTALL),
    re.compile(r"FIXED_DOC_RELATIONS = \{.*?\}\n", re.DOTALL),
    re.compile(r"# Bảng chuyển đổi verb-root.*?_VERB_ROOT_CANONICAL = \{.*?\}\n", re.DOTALL),
    re.compile(r"# Tập động.*?\n_CROSS_VERB_MAPPING = \{.*?\}\n", re.DOTALL),
]

n_removed = 0
for pat in blocks_to_remove:
    new_text, n = pat.subn("", mg_text)
    if n > 0:
        mg_text = new_text
        n_removed += n

if n_removed > 0:
    # Add sys.path + import at the top after existing imports
    sys_path_block = (
        "\nimport os as _os, sys as _sys\n"
        "_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))\n"
        "from backend.config import (\n"
        "    CHUNKING_RELATIONS, FIXED_NODE_RELATIONS, BLACKLIST_RELATIONS,\n"
        "    FIXED_DOC_RELATIONS, _VERB_ROOT_CANONICAL, _CROSS_VERB_MAPPING\n"
        ")\n"
    )
    # Insert just before the first function/class definition or after last import
    insert_after = "load_dotenv()\n"
    mg_text = mg_text.replace(insert_after, insert_after + sys_path_block, 1)
    with open("utils/migrate_neo4j_relations.py", "w", encoding="utf-8") as f:
        f.write(mg_text)
    print(f"  ✅  {n_removed} blocks removed, imports added")
else:
    print("  ⏭  Nothing removed")


# ─────────────────────────────────────────────────────────────
# 5.  debug_neo4j_relations.py
# ─────────────────────────────────────────────────────────────
print("\n[5] debug_neo4j_relations.py")
with open("utils/debug_neo4j_relations.py", "r", encoding="utf-8") as f:
    dg_text = f.read()

blocks_debug = [
    re.compile(r"FIXED_NODE_RELATIONS = \{.*?\}\n", re.DOTALL),
    re.compile(r"BLACKLIST_RELATIONS = \{.*?\}\n", re.DOTALL),
    re.compile(r"FIXED_DOC_RELATIONS = \{.*?\}\n", re.DOTALL),
    re.compile(r"# Bảng chuyển đổi verb-root.*?_VERB_ROOT_CANONICAL = \{.*?\}\n", re.DOTALL),
    re.compile(r"_CROSS_VERB_MAPPING = \{.*?\}\n", re.DOTALL),
]

n_removed = 0
for pat in blocks_debug:
    new_text, n = pat.subn("", dg_text)
    if n > 0:
        dg_text = new_text
        n_removed += n

if n_removed > 0:
    sys_path_block = (
        "\nimport os as _os, sys as _sys\n"
        "_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))\n"
        "from backend.config import (\n"
        "    FIXED_NODE_RELATIONS, BLACKLIST_RELATIONS, FIXED_DOC_RELATIONS,\n"
        "    _VERB_ROOT_CANONICAL, _CROSS_VERB_MAPPING\n"
        ")\n"
    )
    # Find a good insertion point — after dotenv load or after sys import
    insert_candidates = ["load_dotenv()\n", "import re\n"]
    inserted = False
    for anchor in insert_candidates:
        if anchor in dg_text:
            dg_text = dg_text.replace(anchor, anchor + sys_path_block, 1)
            inserted = True
            break
    if not inserted:
        dg_text = sys_path_block + dg_text

    with open("utils/debug_neo4j_relations.py", "w", encoding="utf-8") as f:
        f.write(dg_text)
    print(f"  ✅  {n_removed} blocks removed, imports added")
else:
    print("  ⏭  Nothing removed")

print("\nDone!")
