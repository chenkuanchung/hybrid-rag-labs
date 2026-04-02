"""將 Lab 5 結構化三元組匯入 Neo4j（與 Lab 2 相同正則模式，資料來源為 docs/kg_triples.txt）。"""
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm

from triples_parse import parse

LAB5 = Path(__file__).resolve().parent
DATA_FILE = LAB5 / "docs" / "kg_triples.txt"

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))


def upsert(tx, h_l, h, rel, t_l, t, props):
    tx.run(
        f"""
        MERGE (h:{h_l} {{name: $h}})
        MERGE (t:{t_l} {{name: $t}})
        MERGE (h)-[r:{rel}]->(t)
        SET r += $props
        """,
        h=h,
        t=t,
        props=props,
    )


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"找不到三元組檔：{DATA_FILE}")

    lines = []
    for ln in DATA_FILE.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        n_ok = 0
        for line in tqdm(lines, desc="Neo4j ingest"):
            res = parse(line)
            if res:
                session.execute_write(upsert, *res)
                n_ok += 1
            else:
                print("略過無法解析的行：", line[:80])

    driver.close()
    if not lines:
        print("警告：沒有任何非註解行；Neo4j 已清空且未寫入三元組。請編輯 docs/kg_triples.txt。")
    else:
        print(f"Ingestion done. 成功匯入 {n_ok} / {len(lines)} 行。")


if __name__ == "__main__":
    main()
