"""Lab 5 / ingest 共用的三元組行解析（與 Lab 2 正則一致，不依賴 Neo4j）。"""
import re
from typing import Any

PATTERNS = [
    (r"^(.+?) works_at (.+?)\.$", ("Person", "WORKS_AT", "Company")),
    (r"^(.+?) produces (.+?)\.$", ("Company", "PRODUCES", "Product")),
    (r"^(.+?) partners_with (.+?)\.$", ("Company", "PARTNERS_WITH", "Company")),
    (r"^(.+?) supplies (.+?) to (.+?)\.$", ("Company", "SUPPLIES", "Company")),
    (r"^(.+?) leads (.+?)\.$", ("Person", "LEADS", "Product")),
]


def parse(line: str) -> Any:
    for pat, (h_l, rel, t_l) in PATTERNS:
        m = re.match(pat, line)
        if m:
            if rel == "SUPPLIES":
                head, item, tail = m.groups()
                return (
                    h_l,
                    head.strip(),
                    rel,
                    t_l,
                    tail.strip(),
                    {"item": item.strip(), "source": line},
                )
            return (
                h_l,
                m.group(1).strip(),
                rel,
                t_l,
                m.group(2).strip(),
                {"source": line},
            )
    return None
