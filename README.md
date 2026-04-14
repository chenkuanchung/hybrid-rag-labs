# Enterprise-Grade Hybrid GraphRAG Implementation Labs
### 企業級混合式圖譜 RAG 技術實作：從向量檢索到圖譜防護

## 📖 專案簡介
本專案為參與 **「AI 新秀計畫 (AI Elites Program)」** 課程期間的深度實作紀錄。

本系列實驗旨在探討傳統向量 RAG (Retrieval-Augmented Generation) 在處理複雜關聯性問題（如多跳推理、全域理解）時的侷限性，並透過結合 **Neo4j 知識圖譜** 與 **Chroma 向量資料庫**，建構出一套具備「精準推理」能力與「生產環境安全防護 (Guardrails)」機制的混合式 GraphRAG 系統。

## 💡 核心技術概念：為什麼需要 GraphRAG？
傳統的向量 RAG 將文檔切分為獨立的文本塊（Chunks），雖然擅長精確定位局部資訊，但面對跨文件關聯或需要邏輯傳遞的問題時容易失效。

* **向量 RAG (Vector RAG)：** 基於語意相似度搜尋，適合回答「什麼是 X」。
* **圖譜 RAG (GraphRAG)：** 透過實體（Entities）與關係（Relationships）建立網狀連結，適合回答「X 與 Y 有什麼關聯」或「透過 Z 負責的項目有哪些」等複雜問題。

---

## 🛠️ 實驗路線圖 (Lab Roadmap)

| 階段 | 模組名稱 | 實作重點與技術挑戰 |
| :--- | :--- | :--- |
| **Lab 0** | **環境驗證** | 部署本地端 vLLM (Qwen2.5) 與 Docker 版 Neo4j 圖資料庫服務。 |
| **Lab 1** | **向量 RAG 基礎** | 實作 `SemanticChunker` 語意切分，建立 Chroma 向量索引管線。 |
| **Lab 2** | **圖譜資料匯入** | 撰寫 Regex 解析結構化三元組，利用 Cypher `MERGE` 語句構建知識圖譜。 |
| **Lab 3** | **圖譜檢索問答** | 實作 LLM 實體抽取與 K-Hop 子圖擴展，初步實現圖譜推理問答。 |
| **Lab 4** | **混合式 Graph RAG** | **核心架構實現**：結合向量搜尋（定位）與圖譜擴展（推理）的 Hybrid Pipeline。 |
| **Lab 5** | **實務語料萃取** | 處理非結構化企業語料（會議紀錄、Email、工單）。利用 LLM 自動化「萃取、清洗、匯入」流程。 |
| **Lab 6** | **Guardrails 防護機制** | **生產環境安全化**：加入注入偵測、主題過濾、證據充足性檢查與事實查核 (Grounding Check)。 |

---

## 🏗️ 技術棧 (Tech Stack)
* **LLM 推論引擎:** vLLM (Local Deployment)
* **模型:** Qwen2.5-3B-Instruct / sentence-transformers
* **圖形資料庫:** Neo4j (Cypher Query Language)
* **向量資料庫:** Chroma DB
* **開發框架:** LangChain / LangChain-Experimental

## 🧠 開發筆記與 AI 協作 (AI-Augmented Development)
在本專案的 `TODO` 完成過程中，我深度採用了 **AI 協作開發模式**：
* **複雜查詢優化：** 利用 LLM 協助生成與校對複雜的 Cypher 可變長度路徑查詢語句。
* **Prompt 迭代測試：** 在 Lab 5 與 Lab 6 中，透過與 AI 對話模擬攻擊路徑，反覆迭代事實查核 (Grounding) 與主題過濾的提示詞 (Prompts)。
* **Bug 排除：** 快速定位 Python 環境依賴與 Neo4j Bolt 連線協議的配置問題。

這種「人機協作」的開發流程，讓我能從繁瑣的語法糾錯中解放，將更多精力專注於 **RAG 架構設計** 與 **系統安全性測試**，這也是我認為未來 AI 工程師必備的核心能力。

## 🙏 致謝與聲明 (Credits)
本實驗框架與教材由 **國立陽明交通大學 AI 新秀計畫課程** 提供。
* **教材設計：** 陳界安 (Chieh-An Chen) 老師。
* **實作與優化：** [你的名字/GitHub ID]。

*註：本 Repository 僅用於展示課程練習成果與個人技術成長紀錄，尊重並維護原作者之智慧財產權。*

---

## 🚀 如何開始測試
1. 確認已啟動 Docker 版 Neo4j 與 vLLM 服務。
2. 依序執行各 Lab 腳本進行測試：
   ```bash
   # 範例：執行 Lab 6 安全防護版 RAG
   cd lab6
   python guardrailed_rag.py
