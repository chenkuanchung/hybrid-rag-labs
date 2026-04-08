import os, re
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from neo4j import GraphDatabase

# ------- env & clients -------
os.environ["OPENAI_API_KEY"]="EMPTY"
os.environ["OPENAI_API_BASE"]="http://localhost:8299/v1"
LLM_MODEL="Qwen/Qwen1.5-7B-Chat"
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

# Vector store（與 Lab 1 相同索引：SemanticChunker 建於 lab1/chroma_store）
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_chroma_dir = Path(__file__).resolve().parent.parent / "lab1" / "chroma_store"
vectordb = Chroma(persist_directory=str(_chroma_dir), embedding_function=emb)

# Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","password123"))

# ------- functions ----------
def candidate_entities(question:str, k:int=4):
    # TODO 1: 用向量搜尋找出候選實體
    # 步驟：
    #   1. 使用 vectordb.similarity_search(question, k=k) 取得相關文件
    #   2. 用 re.findall(r'[A-Z][A-Za-z]+', d.page_content) 從每份文件中提取大寫開頭的詞
    #   3. 收集到 set 中去重，最後回傳前 5 個
    return []  # <-- 請替換這行

def graph_expand(ents, hop=2):
    if not ents: return []
    # TODO 2: 撰寫 Cypher 查詢並從結果中提取三元組
    # 步驟：
    #   1. 撰寫 Cypher：MATCH p=(n)-[*1..{hop}]-(m) WHERE n.name IN $ents RETURN p LIMIT 100
    #      用 f-string 嵌入 hop
    #   2. 用 driver.session() 執行查詢（參數 ents=ents）
    #   3. 從每個 record 的 r["p"].relationships 中提取三元組字串
    #      格式：f"({rel.start_node['name']})-[:{rel.type}]->({rel.end_node['name']})"
    #   4. 去重後回傳 list
    return []  # <-- 請替換這行

def answer_with_graph(question:str):
    ents=candidate_entities(question)
    triples=graph_expand(ents)
    context="\n".join(triples) if triples else "（圖譜中找不到相關關係）"
    # TODO 3: 撰寫 prompt，讓 LLM 根據圖譜三元組回答問題
    # 要求：
    #   - 角色設定為「企業知識專家」
    #   - 只能根據圖譜關係（context）回答
    #   - 若資訊不足回答「無足夠資訊」
    #   - 以繁體中文回答
    prompt = ""  # <-- 請撰寫你的 prompt（用 f-string 嵌入 context 和 question）
    return llm.invoke(prompt).content.strip(), triples, ents

if __name__=="__main__":
    while True:
        q=input("提問 (Enter 離開)：")
        if not q: break
        ans,triples,ents=answer_with_graph(q)
        print("候選實體：",ents)
        print("Evidence triples:",triples)
        print("Answer:",ans,"\n")