# Lab 3：圖譜檢索問答（Graph Retrieval QA）

## 目標

利用 LLM 從使用者問題中**抽取實體**，再到 Neo4j 圖譜中擴展出相關的子圖（subgraph），最後將子圖作為上下文讓 LLM 生成答案。

## 核心概念

```
使用者問題："Carol 和 Acme 有什麼關係？"
      ↓ LLM 實體抽取
實體列表：["Carol", "Acme"]
      ↓ Cypher 子圖查詢 (max_hop=2)
子圖三元組：
  (Carol)-[:WORKS_AT]->(BoltCorp)
  (BoltCorp)-[:PARTNERS_WITH]->(Acme)
  (Carol)-[:LEADS]->(TurboMotor)
      ↓ 組成 Prompt
LLM 根據子圖回答問題
```

## 程式說明 — `graph_retrieval.py`

| 步驟 | 函式 | 說明 |
|------|------|------|
| 1 | `extract_entities()` | 送出 prompt 請 LLM 從問題中找出人名、公司名或產品名，回傳 JSON 格式 |
| 2 | `fetch_subgraph()` | 以抽取到的實體為起點，在圖譜中做 1~2 hop 的擴展查詢 |
| 3 | `qa_graph()` | 將子圖三元組組成上下文，交由 LLM 回答問題 |

### 關鍵 Cypher 查詢

```cypher
MATCH p=(n)-[*1..2]-(m)
WHERE n.name IN $ents
RETURN p LIMIT 50
```

- `[*1..2]`：表示沿著關係走 1 到 2 步（hop）
- 不限方向（`-` 而非 `->`），可涵蓋更多關聯

## 前置條件

- 已完成 **Lab 2** 的圖譜匯入（Neo4j 中有資料）

## 執行方式

```bash
cd lab3
python graph_retrieval.py
```

## 可嘗試的問題

| 問題 | 預期能檢索到的關係 |
|------|-------------------|
| Alice 在哪裡工作？ | (Alice)-[:WORKS_AT]->(Acme) |
| Acme 的合作夥伴是誰？ | (Acme)-[:PARTNERS_WITH]->(BoltCorp) |
| Carol 和 Acme 有什麼關係？ | 需要 2-hop：Carol → BoltCorp → Acme |
| 誰負責 TurboMotor？ | (Carol)-[:LEADS]->(TurboMotor) |

## 作業

1. 修改 `max_hop` 參數為 1，重新問「Carol 和 Acme 有什麼關係？」，觀察結果有何不同。為什麼 hop 數會影響回答能力？
2. 觀察 `extract_entities()` 的輸出：嘗試問一個不包含明確實體名稱的問題（例如「這家公司有什麼產品？」），看看 LLM 能否正確抽取實體。記錄失敗的案例。
3. 改進 `extract_entities()`：在 prompt 中加入 few-shot 範例，看看能否提升實體抽取的準確率。
4. 比較題：用同一個問題分別在 **Lab 1（向量 RAG）** 和 **Lab 3（圖譜 QA）** 中測試，比較兩者的回答。哪種方式在處理「多跳推理」問題時表現更好？為什麼？
