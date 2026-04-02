# Lab 2：知識圖譜資料匯入（Graph Ingestion）

## 目標

將結構化的三元組文字解析並匯入 Neo4j 圖資料庫，建立知識圖譜。

## 核心概念

```
原始文字：  "Alice works_at Acme."
      ↓ 正則解析
三元組：    (Alice) -[:WORKS_AT]-> (Acme)
      ↓ Cypher MERGE
Neo4j 圖譜儲存
```

**三元組**是知識圖譜的基本單位，由 `(頭實體) -[關係]-> (尾實體)` 組成。

## 程式說明 — `ingest_graph.py`

| 步驟 | 程式碼區段 | 說明 |
|------|-----------|------|
| 1 | `PATTERNS` | 定義 5 種正則表達式，對應不同的關係類型 |
| 2 | `parse()` | 逐行比對正則，提取頭實體、關係、尾實體 |
| 3 | `upsert()` | 使用 Cypher `MERGE` 語句建立或更新節點與關係 |
| 4 | 主程式 | 先清空圖譜 (`DETACH DELETE`)，再逐行匯入 |

### 支援的關係類型

| 正則模式 | 頭節點類型 | 關係 | 尾節點類型 | 範例 |
|----------|-----------|------|-----------|------|
| `X works_at Y` | Person | WORKS_AT | Company | Alice works_at Acme |
| `X produces Y` | Company | PRODUCES | Product | Acme produces RocketSkates |
| `X partners_with Y` | Company | PARTNERS_WITH | Company | Acme partners_with BoltCorp |
| `X supplies Y to Z` | Company | SUPPLIES | Company | BoltCorp supplies TurboMotor to Acme |
| `X leads Y` | Person | LEADS | Product | Carol leads TurboMotor |

## 執行方式

```bash
cd lab2
python ingest_graph.py
```

程式會讀取 `docs/data.txt`，該檔案已包含 7 筆結構化三元組句子，與 Lab 1 的自然語言語料描述的是同一組實體與關係。

## 驗證結果

匯入完成後，開啟 Neo4j Browser（http://localhost:7474），執行以下 Cypher 查詢：

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m
```

應可看到完整的知識圖譜視覺化。

## 作業

1. 在 `docs/data.txt` 新增一種新關係（例如 `Dave manages Bob.`），並在 `PATTERNS` 中加入對應的正則表達式與標籤，驗證能否成功匯入。
2. 在 Neo4j Browser 中練習以下 Cypher 查詢，並記錄結果：
   - 查詢所有在 Acme 工作的人：`MATCH (p:Person)-[:WORKS_AT]->(c:Company {name:'Acme'}) RETURN p.name`
   - 查詢 BoltCorp 供應了什麼給 Acme：`MATCH (b:Company {name:'BoltCorp'})-[r:SUPPLIES]->(a:Company {name:'Acme'}) RETURN r.item`
   - 查詢 Carol 到 Acme 之間的所有路徑：`MATCH p=shortestPath((c:Person {name:'Carol'})-[*]-(a:Company {name:'Acme'})) RETURN p`
3. 思考題：目前的解析方式使用正則表達式，這有什麼限制？如果要處理更複雜的自然語言描述（例如「Alice 自 2023 年起在 Acme 的研發部門擔任工程師」），你會怎麼改進？
