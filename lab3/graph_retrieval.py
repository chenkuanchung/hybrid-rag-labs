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
    query = """
    MATCH p=(n)-[*1..{k}]-(m)
    WHERE n.name IN $ents
    RETURN p LIMIT 50
    """.format(k=max_hop)
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
    prompt=f"""
已知下列關係：
{context}

根據以上資訊，回答使用者問題：
Q: {question}
A:"""
    answer=llm.invoke(prompt).content.strip()
    return answer,triples

if __name__=="__main__":
    while True:
        q=input("提問(Enter 離開)：")
        if not q: break
        ans,ev=qa_graph(q)
        print("Answer:",ans)
        print("Evidence triples:",ev)