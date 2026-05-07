import io
from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://10.9.2.57:7688', auth=('neo4j', 'u7aGQYEWeFJD-jyeHB4ATtoAud73PptW35M1RzFlT-0'))
with driver.session() as session:
    # Check outbound relationships from a specific document
    res = session.run('MATCH (d:Document {document_number: "105/2016/QH13"})-[r]->(child) RETURN type(r) as rel, labels(child) as labels, count(*) as cnt')
    outbound = [dict(r) for r in res]
    
    # Check labels of nodes connected via PART_OF (if it's parent-child)
    res2 = session.run('MATCH (d:Document {document_number: "105/2016/QH13"})<-[r:PART_OF]-(child) RETURN labels(child) as labels, count(*) as cnt')
    inbound_part_of = [dict(r) for r in res2]

with io.open('neo4j_structure.txt', 'w', encoding='utf-8') as f:
    f.write(f'Outbound from 105/2016/QH13: {outbound}\n')
    f.write(f'Inbound PART_OF to 105/2016/QH13: {inbound_part_of}\n')
driver.close()
