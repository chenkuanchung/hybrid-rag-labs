# Lab 1：向量 RAG（Vector RAG）

## 目標

建立一個基礎的 **Retrieval-Augmented Generation (RAG)** 系統，透過向量檢索從文件中找出與問題相關的段落，再交由 LLM 生成答案。

## 核心概念

```
使用者問題
    ↓
Embedding 模型 → 向量化
    ↓
向量資料庫 (Chroma) → 相似度搜尋 → 取出相關文件片段
    ↓
LLM 根據檢索到的片段生成答案
```

## 程式說明 — `vector_rag.py`

| 步驟 | 程式碼區段 | 說明 |
|------|-----------|------|
| 1 | `TextLoader` + `RecursiveCharacterTextSplitter` | 載入 `docs/data.txt` 並切分成小段落（chunk_size=100） |
| 2 | `HuggingFaceEmbeddings` + `Chroma` | 使用多語言模型將文字轉為向量，存入 Chroma 向量資料庫 |
| 3 | `RetrievalQA.from_chain_type` | 建立 RAG Chain：檢索器取出前 4 筆相關片段，LLM 根據這些片段回答問題 |
| 4 | `while True` 互動迴圈 | 使用者可持續提問，直到按 Enter 離開 |

## 資料檔案 — `docs/data.txt`

包含一組簡易的企業知識，例如人員任職、公司產品、合作關係等。

## 執行方式

```bash
cd lab1
python vector_rag.py
```

## 預期行為

程式啟動後會建立向量索引（首次較慢），之後進入互動問答模式。可嘗試的問題：

- `Alice 在哪裡工作？`
- `Acme 生產什麼？`
- `誰負責 TurboMotor？`

## 作業

1. 修改 `chunk_size` 參數（例如改為 50 或 200），觀察切分結果對回答品質的影響。嘗試回答：為什麼 chunk_size 的選擇很重要？
2. 在 `docs/data.txt` 新增 2-3 筆資料（例如 `Dave works_at BoltCorp.`），重新執行程式，確認新資料能被正確檢索。
3. 嘗試問一個**需要跨多筆資料推理**的問題（例如「Acme 的合作夥伴供應什麼零件？」），觀察純向量 RAG 是否能正確回答。記錄你的觀察，這將與後續 Graph RAG 做比較。
4. 思考題：向量 RAG 的檢索方式是基於「語意相似度」，這在什麼情境下可能會失敗？
