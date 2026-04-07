from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
import os

# LLM 仍走本機 vLLM
os.environ["OPENAI_API_KEY"]  = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
llm = ChatOpenAI(model="Qwen/Qwen1.5-7B-Chat", temperature=0.2)

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 1) 載入文件：以與索引相同的 embedding 做語意斷句（全系列 Lab 共用此策略）
docs = TextLoader("docs/data.txt", encoding="utf-8").load()
emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# TODO 1: 建立 SemanticChunker
# 使用 emb 作為 embedding，breakpoint_threshold_type="percentile"，breakpoint_threshold_amount=90
# 然後呼叫 split_documents(docs) 取得切分結果
splitter = None  # <-- 請替換這行
splits = []      # <-- 請替換這行
print(f"SemanticChunker：{len(splits)} 個 chunk")

# 2) Embedding + VectorStore
# TODO 2: 使用 Chroma.from_documents() 建立向量資料庫
# 參數：splits, emb, persist_directory="chroma_store"
vectordb = None  # <-- 請替換這行

# 3) 建立 RAG Chain
# TODO 3: 使用 RetrievalQA.from_chain_type() 建立 RAG Chain
# 參數：llm, retriever=vectordb.as_retriever(k=4), chain_type="stuff"
qa = None  # <-- 請替換這行

while True:
    q = input("提問 (enter 離開)：")
    if not q: break
    ans = qa.invoke({"query": q})
    print("Answer:", ans["result"])
