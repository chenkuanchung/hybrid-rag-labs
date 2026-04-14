import os, re
from pathlib import Path
import spacy

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from neo4j import GraphDatabase

# ------- env & clients -------
os.environ["OPENAI_API_KEY"]="EMPTY"
os.environ["OPENAI_API_BASE"]="http://localhost:8299/v1"
LLM_MODEL="Qwen2.5-3B-Instruct"
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

# Vector store（與 Lab 1 相同索引：SemanticChunker 建於 lab1/chroma_store）
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_chroma_dir = Path(__file__).resolve().parent.parent / "lab1" / "chroma_store"
vectordb = Chroma(persist_directory=str(_chroma_dir), embedding_function=emb)

# Neo4j
driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j","password123"))

# 載入 spaCy 模型
nlp = spacy.load("en_core_web_sm")

# ------- functions ----------
def candidate_entities(question:str, k:int=4):
    # TODO 1: 用向量搜尋找出候選實體
    # 步驟：
    #   1. 使用 vectordb.similarity_search(question, k=k) 取得相關文件
    #   2. 用 re.findall(r'[A-Z][A-Za-z]+', d.page_content) 從每份文件中提取大寫開頭的詞
    #   3. 收集到 set 中去重，最後回傳前 5 個
    docs = vectordb.similarity_search(question, k=k)
    entities = set()
    for d in docs:
        found = re.findall(r'[A-Z][A-Za-z]+', d.page_content)
        entities.update(found)
    return list(entities)[:5]

# def candidate_entities(question:str, k:int=4):
#     # 1. 向量搜尋
#     docs = vectordb.similarity_search(question, k=k)
#     entities = set()
    
#     for d in docs:
#         # 2. 讓 spaCy 處理文本，自動進行詞性標註與實體辨識 (NER)
#         doc = nlp(d.page_content)
        
#         # 3. 過濾並抓取特定標籤的實體
#         for ent in doc.ents:
#             # 只抓取：組織(ORG)、人名(PERSON)、產品(PRODUCT)、地理位置(GPE)
#             if ent.label_ in ['ORG', 'PERSON', 'PRODUCT', 'GPE']: 
#                 entities.add(ent.text)
                
#     return list(entities)[:5]

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
    query = f"""
    MATCH p=(n)-[*1..{hop}]-(m)
    WHERE n.name IN $ents
    RETURN p LIMIT 100
    """
    triples = set()
    with driver.session() as session:
        result = session.run(query, ents=ents)
        for record in result:
            path = record["p"]
            for rel in path.relationships:
                # 取得這條關係上的所有屬性 (轉成 dict)
                props = dict(rel)
                # 如果有屬性，就格式化成字串，否則留空
                prop_str = f" {props}" if props else ""
                
                # 組裝三元組字串時，把屬性加進去
                triple_str = f"({rel.start_node['name']})-[:{rel.type}{prop_str}]->({rel.end_node['name']})"
                triples.add(triple_str)
    return list(triples)

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

    prompt = f"""
    你是一位企業知識專家。請嚴格根據以下提供的「已知圖譜關係」來回答使用者的問題。
    如果已知關係中沒有足夠的資訊，請直接回答「無足夠資訊」。

    【已知圖譜關係】
    {context}

    【使用者問題】
    {question}

    請用繁體中文簡明扼要地回答：
    """
    return llm.invoke(prompt).content.strip(), triples, ents


if __name__=="__main__":
    while True:
        q=input("提問 (Enter 離開)：")
        if not q: break
        ans,triples,ents=answer_with_graph(q)
        print("候選實體：",ents)
        print("Evidence triples:",triples)
        print("Answer:",ans,"\n")
