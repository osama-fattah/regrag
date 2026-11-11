"""Command line: regrag ingest | ask | eval | serve."""

from __future__ import annotations

import argparse
import json
import sys

from regrag.config import Settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="regrag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="chunk and index a folder of .md/.txt/.pdf files")
    pi.add_argument("--src", default="data/sample_corpus")

    pa = sub.add_parser("ask", help="ask one question")
    pa.add_argument("question")
    pa.add_argument("--json", action="store_true")

    pe = sub.add_parser("eval", help="run the gold set and write a report")
    pe.add_argument("--gold", default="data/eval/gold.jsonl")
    pe.add_argument("--out", default="reports")
    pe.add_argument("--gate", action="store_true", help="exit 1 if metrics fall below thresholds")

    ps = sub.add_parser("serve", help="run the API")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    s = Settings.from_env()

    if args.cmd == "ingest":
        from regrag.embeddings import get_embedder
        from regrag.index import HybridIndex
        from regrag.ingest import load_corpus

        chunks = load_corpus(args.src)
        HybridIndex.build(chunks, get_embedder(s)).save(s.index_dir)
        print(
            f"Indexed {len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents -> {s.index_dir}"
        )
        return 0

    if args.cmd == "ask":
        from regrag.pipeline import RegRAG

        a = RegRAG.from_settings(s).ask(args.question)
        if args.json:
            print(json.dumps(a.to_dict(), indent=2))
        else:
            print(a.answer)
            for c in a.citations:
                print(f"  [{c.n}] {c.title} > {c.section} ({c.chunk_id})")
        return 0

    if args.cmd == "eval":
        from regrag.eval.run import check_gates, evaluate, load_gold, write_report
        from regrag.pipeline import RegRAG

        rag = RegRAG.from_settings(s)
        rows, summary = evaluate(rag, load_gold(args.gold))
        path = write_report(rows, summary, args.out, f"{rag.reasoner.name} + {rag.index.embedder.name}")
        print(json.dumps(summary, indent=2))
        print(f"Report: {path}")
        if args.gate and (failures := check_gates(summary)):
            print("Quality gate FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
            return 1
        return 0

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("regrag.api:app", host=args.host, port=args.port)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
