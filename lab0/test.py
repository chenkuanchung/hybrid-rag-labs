import os
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI

# vLLM (OpenAI 端點)
os.environ["OPENAI_API_KEY"]  = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

llm = ChatOpenAI(model="Qwen/Qwen1.5-7B-Chat", temperature=0)
print(llm.invoke("用一句話解釋何謂 RAG").content)

# Neo4j 連線
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","password123"))
with driver.session() as s:
    result = s.run("RETURN 1 AS ok").single().value()
    print("Neo4j status:", result)
driver.close()