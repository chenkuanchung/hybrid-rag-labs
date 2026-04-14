"""Lab 6：為混合式 Graph RAG 加入 Guardrails（防護機制）。

基於 Lab 4 的 hybrid Graph RAG pipeline，在 Input / Retrieval / Output
三個階段加入四道防護：

  1. 注入偵測（rule-based）
  2. 主題過濾（LLM-as-judge）
  3. 證據充足性（rule-based）
  4. 事實查核（LLM-as-judge）
"""
import argparse
import json
import os
import re
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from neo4j import GraphDatabase

# ------- env & clients -------
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8299/v1"
LLM_MODEL = "Qwen2.5-3B-Instruct"
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
_chroma_dir = Path(__file__).resolve().parent.parent / "lab1" / "chroma_store"
vectordb = Chroma(persist_directory=str(_chroma_dir), embedding_function=emb)

driver = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "password123"))

# ============================================================
# Guardrails
# ============================================================

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)",
    r"forget\s+(all\s+)?(previous|above|prior|your|instructions)",
    # --- 以下為新增的防護規則 ---
    # 英文攻擊：忽略前文指令
    r"disregard\s+(all\s+)?(previous|above|prior|instructions)",
    # 英文攻擊：企圖獲取或修改系統提示詞
    r"system\s+prompt",
    # 英文攻擊：越獄與開發者模式 (Jailbreak / DAN mode)
    r"(jailbreak|dan\s+mode|developer\s+mode)",
    # 中文攻擊：強制角色扮演
    r"你(現在)?是一個",
    r"(假裝|扮演)(你是)?",
    # 中文攻擊：要求忽略先前的設定
    r"忽略.*(指令|前文|提示|設定)"
]


def guard_injection(question: str) -> dict:
    """Input guardrail #1：規則比對偵測 prompt injection。"""
    for pat in INJECTION_PATTERNS:
        if re.search(pat, question, re.IGNORECASE):
            return {"pass": False, "reason": "偵測到可能的提示注入（prompt injection）"}
    return {"pass": True, "reason": "安全"}


def guard_topic(question: str) -> dict:
    """Input guardrail #2：LLM 判斷問題是否與企業知識相關。"""
    # TODO 2: 撰寫 prompt 讓 LLM 判斷問題是否與企業知識相關
    # 要求：
    #   - 定義「企業知識」的範疇（人員任職、公司資訊、產品、供應鏈、合作夥伴等）
    #   - 嵌入使用者的 question
    #   - 要求 LLM 僅回傳 JSON：{"relevant": true/false, "reason": "簡短理由"}
    prompt = f"""
    你是一位嚴格的企業內部知識守門員。
    請判斷以下的【使用者問題】是否與「企業內部知識」相關。
    
    「企業內部知識」的合法範疇包含：
    1. 人員任職狀況與負責產品
    2. 公司基本資訊與組織架構
    3. 公司產品資訊（如 RocketSkates, TurboMotor 等）
    4. 供應鏈關係（誰供應什麼給誰）
    5. 合作夥伴與策略聯盟關係
    
    如果問題涉及上述範疇，請判定為相關 (true)；如果問題是日常閒聊、天氣、與公司業務無關的常識，請判定為不相關 (false)。

    【使用者問題】：
    {question}

    請嚴格「只」回傳 JSON 格式的結果，絕對不要包含任何其他文字、Markdown 標記或解釋。
    JSON 格式如下：
    {{"relevant": true或false, "reason": "簡短的判斷理由（繁體中文）"}}
    """
    try:
        resp = llm.invoke(prompt).content.strip()
        match = re.search(r"\{.*?\}", resp, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if data.get("relevant", False):
                return {"pass": True, "reason": data.get("reason", "相關")}
            return {"pass": False, "reason": data.get("reason", "與企業知識無關")}
    except Exception:
        pass
    return {"pass": True, "reason": "（無法判斷，預設放行）"}


def guard_evidence(triples: list, min_count: int = 1) -> dict:
    """Retrieval guardrail：證據不足時拒答，避免憑空生成。"""
    # TODO 3: 實作證據充足性檢查
    # 邏輯：
    #   - 若 len(triples) < min_count → 回傳 {"pass": False, "reason": "僅檢索到 N 筆三元組，證據不足"}
    #   - 否則 → 回傳 {"pass": True, "reason": "檢索到 N 筆三元組"}
    n = len(triples)
    if n < min_count:
        return {"pass": False, "reason": f"僅檢索到 {n} 筆三元組，證據不足"}
    return {"pass": True, "reason": f"檢索到 {n} 筆三元組"}


def guard_grounding(answer: str, triples: list) -> dict:
    """Output guardrail：LLM 查核答案是否有圖譜證據支持。"""
    context = "\n".join(triples)
    # TODO 4: 撰寫 prompt 讓 LLM 做事實查核（grounding check）
    # 要求：
    #   - 角色設定為「事實查核助手」
    #   - 提供圖譜證據（context）和 LLM 生成的回答（answer）
    #   - 要求 LLM 判斷回答中的每項陳述是否都有圖譜依據
    #   - 僅回傳 JSON：{"grounded": true/false, "reason": "..."}
    prompt = f"""
    你是一位嚴格的「事實查核助手」。
    你的任務是比對下方的【圖譜證據】與【回答】，檢查回答中的「每一項陳述」是否都能在證據中找到完全的支持。
    
    【審查標準】：
    - 如果回答中包含了任何沒有出現在圖譜證據中的資訊（即使該資訊在現實世界中是正確的），或者有過度的邏輯推論，請判定為 false。
    - 只有當回答「完全」且「僅」基於圖譜證據時，才能判定為 true。

    【圖譜證據】：
    {context}

    【回答】：
    {answer}

    請嚴格「只」回傳 JSON 格式的結果，絕對不要包含任何其他文字、Markdown 標記或解釋。
    JSON 格式如下：
    {{"grounded": true或false, "reason": "簡短的查核理由（繁體中文）"}}
    """
    try:
        resp = llm.invoke(prompt).content.strip()
        match = re.search(r"\{.*?\}", resp, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if data.get("grounded", False):
                return {"pass": True, "reason": data.get("reason", "有根據")}
            return {"pass": False, "reason": data.get("reason", "答案可能包含幻覺")}
    except Exception:
        pass
    return {"pass": True, "reason": "（無法判斷，預設放行）"}


# ============================================================
# RAG Pipeline（與 Lab 4 相同邏輯）
# ============================================================

def candidate_entities(question: str, k: int = 4):
    docs = vectordb.similarity_search(question, k=k)
    ent = set()
    for d in docs:
        for tok in re.findall(r"[A-Z][A-Za-z]+", d.page_content):
            ent.add(tok)
    return list(ent)[:5]


def graph_expand(ents, hop=2):
    if not ents:
        return []
    query = f"""
    MATCH p=(n)-[*1..{hop}]-(m)
    WHERE n.name IN $ents
    RETURN p LIMIT 100
    """
    with driver.session() as s:
        recs = s.run(query, ents=ents)
        triples = set()
        for r in recs:
            for rel in r["p"].relationships:
                triples.add(
                    f"({rel.start_node['name']})-[:{rel.type}]->({rel.end_node['name']})"
                )
        return list(triples)


def generate_answer(question: str, triples: list) -> str:
    context = "\n".join(triples) if triples else "（圖譜中找不到相關關係）"
    prompt = (
        "你是一位企業知識專家，只能根據下列圖譜關係回答問題，"
        "若資訊不足請回答「無足夠資訊」。\n"
        f"圖譜：\n{context}\n\n"
        f"問題：{question}\n"
        "答案（繁體中文）："
    )
    return llm.invoke(prompt).content.strip()


# ============================================================
# Guardrailed Pipeline
# ============================================================

def _icon(passed: bool) -> str:
    return "v" if passed else "x"


def guardrailed_rag(question: str, enable_guards: bool = True):
    """完整 pipeline：Input → Retrieval → Generation → Output，每階段可加 guardrail。"""

    print(f"\n{'=' * 55}")
    print(f"  提問：{question}")
    print("=" * 55)

    # ---- Input Guards ----
    if enable_guards:
        print("── Input Guard ──")

        inj = guard_injection(question)
        print(f"  [{_icon(inj['pass'])}] 注入偵測：{inj['reason']}")
        if not inj["pass"]:
            print("── 最終回答 ──")
            print(f"  [已攔截] {inj['reason']}\n")
            return

        topic = guard_topic(question)
        print(f"  [{_icon(topic['pass'])}] 主題過濾：{topic['reason']}")
        if not topic["pass"]:
            print("── 最終回答 ──")
            print(f"  [已攔截] 此問題與企業知識無關，請換個問題。\n")
            return

    # ---- Retrieval ----
    print("── Retrieval ──")
    ents = candidate_entities(question)
    print(f"  候選實體：{ents}")
    triples = graph_expand(ents)
    print(f"  圖譜三元組：{len(triples)} 筆")
    if triples:
        for t in triples[:5]:
            print(f"    {t}")
        if len(triples) > 5:
            print(f"    ... 另有 {len(triples) - 5} 筆")

    # ---- Retrieval Guard ----
    if enable_guards:
        print("── Retrieval Guard ──")
        ev = guard_evidence(triples)
        print(f"  [{_icon(ev['pass'])}] 證據充足：{ev['reason']}")
        if not ev["pass"]:
            print("── 最終回答 ──")
            print("  無足夠證據回答此問題。\n")
            return

    # ---- Generation ----
    print("── LLM 回答 ──")
    answer = generate_answer(question, triples)
    print(f"  {answer}")

    # ---- Output Guard ----
    if enable_guards:
        print("── Output Guard ──")
        gr = guard_grounding(answer, triples)
        print(f"  [{_icon(gr['pass'])}] 事實查核：{gr['reason']}")
        if not gr["pass"]:
            print("── 最終回答 ──")
            print(f"  {answer}")
            print(f"  ⚠ 警告：{gr['reason']}\n")
            return

    print("── 最終回答 ──")
    print(f"  {answer}\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Lab 6：Guardrailed Graph RAG")
    ap.add_argument(
        "--no-guard",
        action="store_true",
        help="停用所有 guardrails，直接執行 RAG pipeline（方便對比）",
    )
    args = ap.parse_args()

    guards_on = not args.no_guard
    mode = "ON" if guards_on else "OFF"
    print(f"Guardrails: {mode}")
    print("輸入問題開始問答，按 Enter 離開。\n")

    while True:
        q = input("提問：")
        if not q:
            break
        guardrailed_rag(q, enable_guards=guards_on)

    driver.close()
