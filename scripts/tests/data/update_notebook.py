import nbformat
import os
import glob

nb_path = 'd:/iCOMM/Legal-RAG/notebook/legal_rag_qdrant_colab.ipynb'
nb = nbformat.read(nb_path, as_version=4)

retrieval_files = [
    'd:/iCOMM/Legal-RAG/backend/retrieval/base.py',
    'd:/iCOMM/Legal-RAG/backend/retrieval/embedder.py',
    'd:/iCOMM/Legal-RAG/backend/retrieval/chunker.py',
    'd:/iCOMM/Legal-RAG/backend/retrieval/reranker.py',
    'd:/iCOMM/Legal-RAG/backend/retrieval/hybrid_search.py'
]

code = ''
for f in retrieval_files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
            # remove imports of local backend to avoid module errors in notebook
            content = content.replace('from backend.', '# from backend.')
            code += content + '\n\n'
    except Exception as e:
         pass

format_code = '''
def format_result(result):
    payload = result.payload
    out = f"--- [Điểm {result.id}] ---\n{{\\n"
    for k, v in payload.items():
        if isinstance(v, str):
            v_esc = v.replace('\\n', '\\\\n')
            out += f'    "{k}": "{v_esc}",\\n'
        elif isinstance(v, bool):
            out += f'    "{k}": {"true" if v else "false"},\\n'
        else:
            out += f'    "{k}": {v},\\n'
    out += "}\\n----------------------------------------\\n"
    return out
'''

for cell in nb.cells:
    if cell.cell_type == 'markdown':
        cell.source = cell.source.replace('Kaggle', 'Colab').replace('/kaggle/working', '/content')

# We can append it to the end or replace a specific cell.
# I'll just clear the notebook and keep it simple: insert standard setup + combined retrieval
# Or instead, just append a big cell for the synthesized code
nb.cells.append(nbformat.v4.new_markdown_cell('# Synthesized Retrieval Code & Formatting'))
nb.cells.append(nbformat.v4.new_code_cell(code + '\n' + format_code))

nbformat.write(nb, nb_path)
print('Notebook updated successfully.')
