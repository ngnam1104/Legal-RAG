# Neo4j Graph Database Schema & Queries

Tài liệu này mô tả cấu trúc các Node, Relationship và các mẫu truy vấn (Cypher) được sử dụng trong hệ thống Legal-RAG.

## 1. Node Labels

- **Document**: Văn bản luật chính (Luật, Thông tư, Nghị định...).
- **Article**: Điều khoản (Điều 1, Điều 2...).
- **Clause**: Khoản (Khoản 1, Khoản a...).
- **Chunk**: Mảnh văn bản nhỏ nhất (Lá), chứa `qdrant_id` để liên kết với Vector DB.
- **Authority**: Cơ quan ban hành (Chính phủ, Bộ Y tế...).
- **Signer**: Người ký văn bản.
- **LegalType**: Loại văn bản (Thông tư, Nghị định...).
- **Sector**: Lĩnh vực (Y tế, Thể thao...).

## 2. Relationship Types

### Cấp bậc văn bản (Hierarchy)
- `(Article)-[:BELONGS_TO]->(Document)`
- `(Clause)-[:PART_OF]->(Article)`
- `(Chunk)-[:PART_OF]->(Clause|Article)`
- `(Chunk)-[:BELONGS_TO]->(Document)` (Trường hợp Chunk mồ côi không thuộc Điều nào)

### Metadata & Thuộc tính
- `(Authority)-[:ISSUED]->(Document)`
- `(Signer)-[:SIGNED]->(Document)`
- `(Document)-[:HAS_TYPE]->(LegalType)`
- `(Document)-[:HAS_SECTOR]->(Sector)`

### Quan hệ chéo (Ontology - 10 Nhãn)
Các quan hệ này nối từ `Document` nguồn sang `Document` đích (hoặc Article/Clause đích thông qua Phantom Hierarchy).

| Nhãn | Ý nghĩa |
|---|---|
| `BASED_ON` | Căn cứ pháp lý |
| `AMENDS` | Sửa đổi văn bản |
| `REPEALS` | Bãi bỏ văn bản |
| `REPLACES` | Thay thế văn bản |
| `GUIDES` | Hướng dẫn thi hành |
| `APPLIES` | Áp dụng văn bản |
| `REFERENCES` | Trích dẫn, tham chiếu |
| `ISSUED_WITH` | Ban hành kèm theo |
| `ASSIGNS` | Giao nhiệm vụ/Uỷ quyền |
| `CORRECTS` | Đính chính văn bản |

## 3. Mẫu truy vấn (Cypher) phổ biến

### Kiểm tra số lượng Node
```cypher
MATCH (n) RETURN labels(n) as Label, count(*) as Count
```

### Kiểm tra số lượng Quan hệ
```cypher
MATCH ()-[r]->() RETURN type(r) as Type, count(*) as Count
```

### Xem toàn bộ cấu trúc của một văn bản (Ví dụ: 51/2025/TT-BYT)
```cypher
MATCH (d:Document {document_number: '51/2025/TT-BYT'})-[:BELONGS_TO|PART_OF*0..3]-(leaf)
RETURN d, leaf
```

### Tìm văn bản bị thay thế hoặc sửa đổi bởi văn bản hiện tại
```cypher
MATCH (d:Document)-[r:AMENDS|REPLACES]->(target:Document)
RETURN d.document_number as Source, type(r) as Type, target.document_number as Target, r.context as Context
```

### Kiểm tra Phantom Hierarchy (Điều/Khoản của văn bản đích)
```cypher
MATCH (d:Document {is_full_text: false})<-[:BELONGS_TO]-(art:Article)<-[:PART_OF]-(cl:Clause)
RETURN d.document_number, art.name, cl.name, art.text
```