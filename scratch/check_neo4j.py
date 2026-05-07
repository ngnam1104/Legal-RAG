import io
from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://10.9.2.57:7688', auth=('neo4j', 'u7aGQYEWeFJD-jyeHB4ATtoAud73PptW35M1RzFlT-0'))
with driver.session() as session:
    res = session.run('MATCH (d:Document) RETURN d.document_number LIMIT 10')
    docs = [r[0] for r in res]
    res2 = session.run('CALL db.relationshipTypes()')
    rels = [r[0] for r in res2]

with io.open('neo4j_info.txt', 'w', encoding='utf-8') as f:
    f.write(f'Docs: {docs}\n')
    f.write(f'Rels: {rels}\n')
driver.close()
