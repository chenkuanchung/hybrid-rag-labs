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
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
LLM_MODEL = "Qwen/Qwen1.5-7B-Chat"

EXTRACTION_PROMPT = """你是知識圖譜抽取助手。下列為企業內部多份文件（繁體中文為主，專有名詞為英文）。
請從中抽取**所有可確定**的人事、產品、供應與合作關係，輸出為**純英文句子**，每行一句，句尾必須是英文句點 . 

**只允許**以下五種句式（頭尾實體名稱需與原文一致，含大小寫）：

1. PersonName works_at CompanyName.
2. CompanyName produces ProductName.
3. CompanyName partners_with CompanyName.
4. CompanyName supplies ItemName to CompanyName.
5. PersonName leads ProductName.

規則：
- 不要輸出編號、說明、markdown、空行。
- 不要推測文件未明確支持的事實。
- 「策略聯盟／策略夥伴／策略合作」對應 partners_with（僅限**公司對公司**）。
- 「供應／出貨給／供應給」且涉及具體品項時，用 supplies 句式（供應方 supplies 品項 to 接收方）。
- 「負責某產品」對應 leads（人 leads 產品）。

--- 文件內容 ---
__CORPUS__
--- 結束 ---

請只輸出符合上述格式的句子，每行一句：
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
    good, bad = [], []
    seen = set()
    for line in lines:
        if parse(line):
            if line not in seen:
                seen.add(line)
                good.append(line)
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
