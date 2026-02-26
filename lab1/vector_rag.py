from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os

# LLM 仍走本機 vLLM
os.environ["OPENAI_API_KEY"]  = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
llm = ChatOpenAI(model="Qwen/Qwen1.5-7B-Chat", temperature=0.2)

# 1) 載入文件
docs = TextLoader("docs/data.txt").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
splits = splitter.split_documents(docs)

# 2) Embedding + VectorStore
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectordb = Chroma.from_documents(splits, emb, persist_directory="chroma_store")

# 3) 建立 RAG Chain
qa = RetrievalQA.from_chain_type(
    llm, retriever=vectordb.as_retriever(k=4),
    chain_type="stuff"
)

while True:
    q = input("提問 (enter 離開)：")
    if not q: break
    ans = qa.invoke({"query": q})
    print("Answer:", ans["result"])