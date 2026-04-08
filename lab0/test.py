import os
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI

# vLLM (OpenAI 端點)
os.environ["OPENAI_API_KEY"]  = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8299/v1"
llm = ChatOpenAI(model="Qwen2.5-3B-Instruct", temperature=0.2)
print(llm.invoke("用一句話解釋何謂 RAG").content)

# Neo4j 連線
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","password123"))
with driver.session() as s:
    result = s.run("RETURN 1 AS ok").single().value()
    print("Neo4j status:", result)
driver.close()