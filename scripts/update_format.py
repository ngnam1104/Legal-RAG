import nbformat
import json

nb_path = 'd:/iCOMM/Legal-RAG/notebook/legal_rag_qdrant_colab.ipynb'
nb = nbformat.read(nb_path, as_version=4)

format_code = '''def format_result(result, rank):
    payload = result.payload
    out = f\"--- [Điểm {rank}] (ID: {result.id}) ---\\n\"
    out += json.dumps(payload, ensure_ascii=False, indent=4)
    out += \"\\n----------------------------------------\\n\"
    return out
'''

# Find the cell we appended and patch format_code, or just append again
found = False
for cell in nb.cells:
    if 'def format_result' in cell.source:
        import re
        cell.source = re.sub(r'def format_result.*?return out\n', format_code, cell.source, flags=re.DOTALL)
        found = True

nbformat.write(nb, nb_path)
print('Payload format updated.')
