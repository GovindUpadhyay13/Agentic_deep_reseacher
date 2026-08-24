# src/phase1_index.py
import os
import time
import urllib.request
from dotenv import load_dotenv

load_dotenv()

import arxiv
import fitz  # PyMuPDF
import chromadb

try:
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass

from chromadb.utils import embedding_functions
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

KEYWORDS = '(agent OR "agentic RAG" OR "tool use" OR "agent memory" OR "computer-use")'
SEARCH_QUERY = f'(cat:cs.CL OR cat:cs.AI OR cat:cs.LG) AND ti:{KEYWORDS} AND submittedDate:[202401010000 TO 202604302359]'
MAX_PAPERS = 700
DATA_DIR = "./data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
DB_DIR = os.path.join(DATA_DIR, "chroma_db")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(name="arxiv_papers", embedding_function=embed_fn)

def chunk_text(text, chunk_size=1200, overlap=200): 
    """Splits text into overlapping chunks to preserve context between paragraphs."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def run_pipeline():
    if collection.count() > 40000:
        print(f"Database already populated with {collection.count()} chunks. Skipping Phase 1.")
        return

    print("Fetching arXiv papers metadata...")
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=MAX_PAPERS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = list(client.results(search))
    print(f"Discovered {len(results)} papers matching search query criteria.")

    for i, paper in enumerate(tqdm(results, desc="Processing Papers")):
        paper_id = paper.get_short_id()
        pdf_path = os.path.join(PDF_DIR, f"{paper_id}.pdf")

        if not os.path.exists(pdf_path):
            try:
                paper.download_pdf(dirpath=PDF_DIR, filename=f"{paper_id}.pdf")
                time.sleep(1)
            except Exception as e:
                print(f"\n[Warning] Failed downloading {paper_id}: {e}")
                continue

        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()

            chunks = chunk_text(full_text)
            if not chunks:
                continue

            ids = [f"{paper_id}_chunk_{c_idx}" for c_idx in range(len(chunks))]
            metadatas = [{
                "arxiv_id": paper_id,
                "title": paper.title,
                "authors": ", ".join([a.name for a in paper.authors]),
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown",
                "chunk_index": c_idx,
                "total_chunks": len(chunks)
            } for c_idx in range(len(chunks))]

            collection.upsert(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"\n[Warning] Error indexing {paper_id}: {e}")
            continue

    print(f"\nPhase 1 Complete! Indexed {collection.count()} chunks into ChromaDB.")

if __name__ == "__main__":
    run_pipeline()
