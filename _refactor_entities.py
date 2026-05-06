"""
Refactoring script: Remove constants from entities.py and replace with imports from backend.config
Run from repo root: python _refactor_entities.py
"""
import re

TARGET = "backend/ingestion/extractor/entities.py"

with open(TARGET, "r", encoding="utf-8") as f:
    text = f.read()

original = text  # keep backup

# ───────────────────────────────────────────────────────
# 1. Replace entire block from the comment to end of _EMPTY_ENTITIES
#    with the new import block
# ───────────────────────────────────────────────────────
IMPORT_BLOCK = (
    "from backend.config import (\n"
    "    FIXED_ENTITY_TYPES,\n"
    "    FIXED_NODE_RELATIONS,\n"
    "    BLACKLIST_RELATIONS,\n"
    "    FIXED_DOC_RELATIONS,\n"
    "    _VERB_ROOT_CANONICAL,\n"
    "    _CROSS_VERB_MAPPING,\n"
    "    DYNAMIC_ENTITY_TYPES,\n"
    "    DYNAMIC_NODE_RELATIONS,\n"
    "    DYNAMIC_DOC_RELATIONS,\n"
    "    _EMPTY_ENTITIES,\n"
    "    ENTITY_NAME_ALIASES\n"
    ")\n"
)

# Block 1: section header + all constant definitions up to (and including) _EMPTY_ENTITIES line
block1_pat = re.compile(
    r"# ={42,}\n# TẬP CỐ ĐỊNH.*?_EMPTY_ENTITIES: Dict\[str, List\[str\]\] = \{\}\n",
    re.DOTALL,
)
text, n1 = block1_pat.subn(IMPORT_BLOCK, text)
print(f"Block 1 (constants block) replacements: {n1}")

# ───────────────────────────────────────────────────────
# 2. Remove ENTITY_NAME_ALIASES dict (it's now in config.py)
# ───────────────────────────────────────────────────────
block2_pat = re.compile(
    r"ENTITY_NAME_ALIASES = \{.*?\n\}\n",
    re.DOTALL,
)
text, n2 = block2_pat.subn("", text)
print(f"Block 2 (ENTITY_NAME_ALIASES) removals: {n2}")

# ───────────────────────────────────────────────────────
# 3. Remove _CROSS_VERB_MAPPING dict (it's now in config.py)
# ───────────────────────────────────────────────────────
block3_pat = re.compile(
    r"_CROSS_VERB_MAPPING = \{.*?\"PROVIDED_WITH\": \"PROVIDED_BY\",\n\}\n",
    re.DOTALL,
)
text, n3 = block3_pat.subn("", text)
print(f"Block 3 (_CROSS_VERB_MAPPING) removals: {n3}")

if n1 == 0 and n2 == 0 and n3 == 0:
    print("WARNING: Nothing matched. File may already be refactored, or patterns are wrong.")
elif text == original:
    print("WARNING: File unchanged after substitution.")
else:
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Written to {TARGET}")
