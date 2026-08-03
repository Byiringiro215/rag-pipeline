"""CLI entrypoint for the custom RAG pipeline.

Usage:
    python main.py ingest file1.pdf file2.docx page.html
    python main.py query "What is the refund policy?"
    python main.py chat
    python main.py --mode dense --no-rerank query "..."
"""
import utils.session
import argparse
import sys

from rag.pipeline.rag import RAGPipeline


def cmd_ingest(args):
    pipeline = RAGPipeline(retrieval_mode=args.mode, use_reranker=not args.no_rerank)
    try:
        pipeline.load()
        print("Loaded existing index; adding new documents to it.")
    except FileNotFoundError:
        print("No existing index found; starting a fresh one.")

    count = pipeline.ingest(args.files)
    pipeline.save()
    print(f"Ingested {count} chunks from {len(args.files)} file(s). Index saved.")


def cmd_query(args):
    pipeline = RAGPipeline(retrieval_mode=args.mode, use_reranker=not args.no_rerank)
    pipeline.load()
    answer = pipeline.query(args.question)
    print("\n=== Answer ===\n")
    print(answer)


def cmd_chat(args):
    pipeline = RAGPipeline(retrieval_mode=args.mode, use_reranker=not args.no_rerank)
    pipeline.load()
    print("RAG chat — type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer = pipeline.query(question)
        print(f"\nAssistant: {answer}")


def build_parser():
    parser = argparse.ArgumentParser(description="Custom RAG pipeline CLI")
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load and index files")
    ingest_parser.add_argument("files", nargs="+", help="Paths to .pdf, .docx, or .html files")
    ingest_parser.set_defaults(func=cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Ask a single question")
    query_parser.add_argument("question")
    query_parser.set_defaults(func=cmd_query)

    chat_parser = subparsers.add_parser("chat", help="Interactive chat loop")
    chat_parser.set_defaults(func=cmd_chat)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    sys.exit(0)
