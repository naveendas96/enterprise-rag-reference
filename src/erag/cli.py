from __future__ import annotations

import argparse
import json

from .evals import run_eval
from .pipeline import RAG


def main() -> None:
    p = argparse.ArgumentParser("erag")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--role", default="employee")
    sub.add_parser("eval")
    args = p.parse_args()
    if args.cmd == "ask":
        ans = RAG().ask(args.question, args.role)
        print(ans.text)
        for c in ans.citations:
            print(f"  [{c['n']}] {c['doc']} > {c['section']}")
    else:
        r = run_eval()
        print("| metric | value |\n|---|---|")
        for k, v in r.items():
            print(f"| {k} | {v:.2f} |" if isinstance(v, float) else f"| {k} | {v} |")
        print(json.dumps(r))


if __name__ == "__main__":
    main()
