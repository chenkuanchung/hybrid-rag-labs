# Lab 0：環境驗證

## 目標

確認本機開發環境已正確安裝並啟動以下兩項核心服務：

1. **vLLM** — 以 OpenAI 相容 API 提供本地 LLM 推論服務
2. **Neo4j** — 圖資料庫，用於後續實驗的知識圖譜儲存與查詢

## 前置準備

| 服務 | 啟動方式 | 預設連線位址 |
|------|----------|-------------|
| Neo4j | `docker compose -f docker-compose-neo4j.yaml up -d` | bolt://localhost:7687 (帳號 `neo4j` / 密碼 `password123`) |
| vLLM | `docker compose -f docker-compose-vllm.yaml up -d` | http://localhost:8299/v1 |

> **提示**：請在專案根目錄執行上述指令。

## 程式說明 — `test.py`

| 步驟 | 說明 |
|------|------|
| 1 | 設定環境變數，將 `OPENAI_API_KEY` 與 `OPENAI_API_BASE` 指向本機 vLLM 端點 |
| 2 | 透過 `ChatOpenAI` 呼叫 LLM，請它用一句話解釋 RAG |
| 3 | 建立 Neo4j 連線，執行 `RETURN 1 AS ok` 驗證資料庫狀態 |

## 執行方式

```bash
cd lab0
python test.py
```

## 預期輸出

- 一段 LLM 產生的文字（說明 RAG 的定義）
- `Neo4j status: 1`（代表連線成功）

若出現連線錯誤，請確認 Docker 容器是否已啟動並正常運行。

## 作業

1. 修改 `test.py` 中的 prompt，改為請 LLM 解釋「什麼是 Knowledge Graph」，觀察回傳結果。
2. 在 Neo4j 驗證段落加入一個 Cypher 查詢 `CALL dbms.components() YIELD name, versions RETURN name, versions`，印出 Neo4j 的版本資訊。
3. 思考題：為什麼我們在本課程中選擇使用 vLLM 而不是直接呼叫 OpenAI API？這樣做有什麼優缺點？
