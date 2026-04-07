import os, json
from typing import List
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"]="EMPTY"
os.environ["OPENAI_API_BASE"]="http://localhost:8000/v1"
llm = ChatOpenAI(model="Qwen/Qwen1.5-7B-Chat", temperature=0)

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","password123"))

def extract_entities(question:str)->List[str]:
    prompt = f"""
你是一位助手，請從使用者問題中找出人名、公司名或產品名，列成 Python list，不要包含重複或其他文字。
問題：{question}
回傳 JSON，例如：{{"entities":["Alice","Acme"]}}
"""
    resp = llm.invoke(prompt).content
    try:
        data=json.loads(resp.strip())
        return data.get("entities",[])
    except Exception:
        return []

def fetch_subgraph(entities:List[str], max_hop:int=2):
    if not entities: return []
    # TODO 1: 撰寫 Cypher 查詢，以 entities 為起點擴展 1~max_hop 步的子圖
    # 要求：
    #   - MATCH p=(n)-[*1..max_hop]-(m)   ← 用 .format(k=max_hop) 嵌入 hop 數
    #   - WHERE n.name IN $ents
    #   - RETURN p LIMIT 50
    query = ""  # <-- 請撰寫 Cypher 查詢
    with driver.session() as s:
        records=s.run(query,ents=entities)
        triples=[]
        for r in records:
            for rel in r["p"].relationships:
                triples.append(f"({rel.start_node['name']})-[:{rel.type}]->({rel.end_node['name']})")
        return list(set(triples))

def qa_graph(question:str):
    ents=extract_entities(question)
    triples=fetch_subgraph(ents)
    context="\n".join(triples) if triples else "（查無相關圖譜）"
    # TODO 2: 撰寫 prompt，將圖譜三元組（context）當作上下文，讓 LLM 根據這些關係回答問題
    # 提示：prompt 應包含：(1) 已知的圖譜關係 (2) 使用者的問題 (3) 要求 LLM 根據上下文回答
    prompt = ""  # <-- 請撰寫你的 prompt（用 f-string 嵌入 context 和 question）
    answer=llm.invoke(prompt).content.strip()
    return answer,triples

if __name__=="__main__":
    while True:
        q=input("提問(Enter 離開)：")
        if not q: break
        ans,ev=qa_graph(q)
        print("Answer:",ans)
        print("Evidence triples:",ev)