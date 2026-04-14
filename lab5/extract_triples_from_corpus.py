"""
讀取 docs/corpus 下所有 .txt，呼叫 LLM 抽取事實，寫入 docs/kg_triples.txt。
僅保留「與 Lab 2 ingest 相同文法」且可通過 triples_parse.parse() 的句子。
"""
import argparse
import os
import re
from pathlib import Path

from langchain_openai import ChatOpenAI

from triples_parse import parse

LAB5 = Path(__file__).resolve().parent
CORPUS_DIR = LAB5 / "docs" / "corpus"
DEFAULT_OUT = LAB5 / "docs" / "kg_triples.txt"

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8299/v1"
LLM_MODEL = "Qwen2.5-3B-Instruct"

# TODO 1: 撰寫 EXTRACTION_PROMPT — 讓 LLM 從語料中抽取三元組
# 這個 prompt 需要：
#   - 說明角色（知識圖譜抽取助手）
#   - 列出五種合法句式：works_at, produces, partners_with, supplies...to, leads
#   - 設定規則：只輸出純英文句子、不推測、不加編號說明
#   - 在適當位置放入 __CORPUS__（稍後會被語料內容取代）
#   - 提供中文關鍵詞到英文關係的對應提示（例如「策略聯盟」→ partners_with）
# 可參考 docs/kg_triples.template.txt 了解五種句式的格式
EXTRACTION_PROMPT = """
你是一位嚴格的「知識圖譜抽取助手」。請從以下語料抽取出事實，並轉換為嚴格的英文三元組句子。
警告：絕對不可以使用下方清單以外的動詞或句型！

【唯五合法的句式】：
1. [Person] works_at [Company]. 
2. [Company] produces [Product]. 
3. [Company] partners_with [Company]. 
4. [Company] supplies [Product] to [Company]. 
5. [Person] leads [Product]. 

【格式糾正範例（非常重要）】：
- 看到「A 和 B 是策略聯盟」，只能輸出：A partners_with B.
- 看到「C 是 D 的產品經理」，只能輸出：C leads D.
- 看到「E 提供 F 給 G」，只能輸出：E supplies F to G.

【語料內容】
__CORPUS__
"""


def load_corpus() -> str:
    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(f"找不到語料目錄：{CORPUS_DIR}")
    parts = []
    for p in sorted(CORPUS_DIR.glob("**/*.txt")):
        body = p.read_text(encoding="utf-8").strip()
        if body:
            parts.append(f"=== 檔案: {p.name} ===\n{body}")
    if not parts:
        raise RuntimeError(f"{CORPUS_DIR} 內沒有任何 .txt")
    return "\n\n".join(parts)


def normalize_line(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^[\-\*]\s+", "", s)
    s = re.sub(r"^\d+[\.\)、]\s*", "", s)
    s = s.strip("`").strip()
    return s


def extract_raw_lines(llm_text: str) -> list[str]:
    text = llm_text.strip()
    if "```" in text:
        m = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    out = []
    for line in text.splitlines():
        n = normalize_line(line)
        if n and not n.startswith("#"):
            out.append(n)
    return out


def filter_parsable(lines: list[str]) -> tuple[list[str], list[str]]:
    # TODO 2: 過濾出可被 parse() 解析的行，同時去重
    # 步驟：
    #   1. 對每一行呼叫 parse(line)（已從 triples_parse 匯入）
    #   2. 若 parse 回傳非 None 且該行尚未出現過 → 加入 good
    #   3. 若 parse 回傳 None → 加入 bad
    #   4. 回傳 (good, bad) 兩個 list
    good, bad = [], []
    seen = set()
    for line in lines:
        parsed = parse(line) # 使用 triples_parse.py 的正則來驗證
        if parsed:
            if line not in seen:
                good.append(line)
                seen.add(line)
        else:
            bad.append(line)
    return good, bad


def main() -> None:
    ap = argparse.ArgumentParser(description="從 corpus 經 LLM 產生 kg_triples.txt")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只印出結果，不寫入檔案",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"輸出路徑（預設：{DEFAULT_OUT}）",
    )
    args = ap.parse_args()

    corpus = load_corpus()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    prompt = EXTRACTION_PROMPT.replace("__CORPUS__", corpus)
    resp = llm.invoke(prompt).content
    raw = extract_raw_lines(resp)
    good, bad = filter_parsable(raw)

    if bad:
        print("以下行無法通過 ingest 文法，已捨棄：")
        for b in bad[:20]:
            print("  ", repr(b)[:120])
        if len(bad) > 20:
            print(f"  ... 另有 {len(bad) - 20} 行")

    header = (
        "# 本檔由 extract_triples_from_corpus.py 產生；可人工增刪後再執行 ingest_graph.py\n"
        f"# 模型：{LLM_MODEL}，temperature=0\n"
        "\n"
    )
    body = "\n".join(good) + ("\n" if good else "")

    if args.dry_run:
        print(header)
        print(body)
        print(f"# （共 {len(good)} 行可匯入）")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + body, encoding="utf-8")
    print(f"已寫入 {args.output} ，可匯入行數：{len(good)}")


if __name__ == "__main__":
    main()
