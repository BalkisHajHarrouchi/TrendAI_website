import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Lajavaness/bilingual-embedding-large")
PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore/db1")

def load_docs(path: str):
    if not os.path.exists(path):
        print(f"[ingest] WARN: {path} not found, skipping.")
        return []
    # Force UTF-8 and fall back to UTF-8 with BOM if needed
    try:
        return TextLoader(path, encoding="utf-8").load()
    except UnicodeDecodeError:
        return TextLoader(path, encoding="utf-8-sig").load()

def main():
    print("[ingest] loading docs...")

    # always resolve facts.txt relative to this file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    facts_path = os.path.join(base_dir, "facts.txt")

    docs = load_docs(facts_path)

    if not docs:
        print("[ingest] no docs loaded, exiting.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    print(f"[ingest] total chunks: {len(chunks)}")

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True},
    )

    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding,
    )

    db.add_documents(chunks)
    db.persist()
    print("[ingest] persisted to", PERSIST_DIR)


if __name__ == "__main__":
    main()
