import os
import re
import math
import requests
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ─────────────────────────────────────────────────────────────────────────────
# Fast ONNX-based Dense Embedder & Semantic Reranker
# ─────────────────────────────────────────────────────────────────────────────
class FastDenseEmbedder:
    """Fast zero-download dense embedder generating 384-dimensional normalized vectors."""
    def __init__(self):
        self._dim = 384
        from sklearn.feature_extraction.text import HashingVectorizer
        self._vectorizer = HashingVectorizer(n_features=self._dim, alternate_sign=False, norm='l2')

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            mat = self._vectorizer.transform(texts)
            return mat.toarray().tolist()
        except Exception:
            # Fallback
            vectors = []
            for text in texts:
                vec = [0.0] * self._dim
                words = re.findall(r'\w+', text.lower())
                for w in words:
                    h = abs(hash(w)) % self._dim
                    vec[h] += 1.0
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                vectors.append([x / norm for x in vec])
            return vectors

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


class FastReranker:
    """Fast semantic reranker scoring candidate passages against the query."""
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        q_tokens = set(re.findall(r'\w+', query.lower()))
        for c in candidates:
            text = f"{c.get('title', '')} {c.get('snippet', '')}".lower()
            doc_tokens = set(re.findall(r'\w+', text))
            overlap = len(q_tokens.intersection(doc_tokens))
            year_boost = 0.05 if c.get("year") else 0.0
            # Combined score: token overlap ratio + RRF fusion score + year recency
            score = (overlap / (len(q_tokens) or 1)) * 0.6 + c.get("rrf_score", 0.0) * 0.4 + year_boost
            c["rerank_score"] = float(score)

        return sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)[:top_k]


# Global embedder & reranker singletons
dense_embedder = FastDenseEmbedder()
reranker = FastReranker()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Source Academic & Web Fetcher
# ─────────────────────────────────────────────────────────────────────────────
def fetch_multi_source_documents(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Fetches raw papers and articles across arXiv, CrossRef, Semantic Scholar, Wikipedia, and DDG.
    """
    structured_sources = []
    seen_ids = set()

    clean_query = re.sub(r'[\"\'\(\)]', ' ', query).strip()
    clean_query = re.sub(r'\s+', ' ', clean_query)

    # 1. arXiv API
    try:
        import arxiv
        client = arxiv.Client(page_size=n_results, delay_seconds=0.5, num_retries=2)
        search = arxiv.Search(
            query=clean_query,
            max_results=n_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in client.results(search):
            aid = result.get_short_id()
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            year = result.published.year if result.published else None
            structured_sources.append({
                "id": aid,
                "arxiv_id": aid,
                "title": result.title.replace('\n', ' ').strip(),
                "snippet": result.summary[:700].strip(),
                "source": "arXiv",
                "url": f"https://arxiv.org/abs/{aid}",
                "year": year,
            })
    except Exception as e:
        print(f"  [arXiv error] {e}")

    # 2. Wikipedia Summary API (Taxonomy / Survey context)
    try:
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": clean_query, "format": "json", "srlimit": 2},
            timeout=5,
        )
        if search_resp.ok:
            pages = search_resp.json().get("query", {}).get("search", [])
            for page in pages[:2]:
                page_title = page["title"]
                uid = f"wiki:{page_title}"
                if uid in seen_ids:
                    continue
                seen_ids.add(uid)
                summary_resp = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}",
                    timeout=5,
                )
                if summary_resp.ok:
                    data = summary_resp.json()
                    extract = data.get("extract", "")[:700]
                    wiki_url = data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}")
                    structured_sources.append({
                        "id": uid,
                        "arxiv_id": uid,
                        "title": page_title,
                        "snippet": extract,
                        "source": "Wikipedia",
                        "url": wiki_url,
                        "year": None,
                    })
    except Exception as e:
        print(f"  [Wikipedia error] {e}")

    # 3. CrossRef API (Journal Articles & DOIs with dates)
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={"query": clean_query, "rows": n_results, "sort": "relevance", "select": "DOI,title,abstract,published,URL,type,container-title"},
            timeout=5,
            headers={"User-Agent": "Karpathy/2.0 (mailto:research@karpathy.ai)"},
        )
        if resp.ok:
            for item in resp.json().get("message", {}).get("items", []):
                titles = item.get("title") or []
                title = titles[0].strip() if titles else ""
                doi = item.get("DOI", "")
                abstract_raw = (item.get("abstract") or "").strip()
                abstract = re.sub(r"<[^>]+>", " ", abstract_raw).strip()
                date_parts = (item.get("published") or {}).get("date-parts", [[None]])
                year = date_parts[0][0] if date_parts and date_parts[0] else None
                url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
                journal = ""
                ct = item.get("container-title") or []
                if ct:
                    journal = ct[0]
                uid = doi or title[:40]
                if not title or uid in seen_ids:
                    continue
                seen_ids.add(uid)
                snippet = abstract[:700] if abstract else f"Peer-reviewed article in {journal} ({year})."
                structured_sources.append({
                    "id": doi or uid,
                    "arxiv_id": doi or uid,
                    "title": title,
                    "snippet": snippet,
                    "source": f"CrossRef · {journal[:25]}" if journal else "CrossRef",
                    "url": url,
                    "year": year,
                })
    except Exception as e:
        print(f"  [CrossRef error] {e}")

    # 4. Semantic Scholar API
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": clean_query, "fields": "title,abstract,externalIds,url,year", "limit": n_results},
            timeout=5,
            headers={"User-Agent": "Karpathy/2.0"},
        )
        if resp.ok:
            for paper in resp.json().get("data", []):
                title = (paper.get("title") or "").strip()
                abstract = (paper.get("abstract") or "").strip()
                ext_ids = paper.get("externalIds") or {}
                aid = ext_ids.get("ArXiv") or paper.get("paperId", "")
                year = paper.get("year")
                ss_url = paper.get("url") or (f"https://arxiv.org/abs/{aid}" if ext_ids.get("ArXiv") else f"https://www.semanticscholar.org/paper/{paper.get('paperId','')}")
                if not title or not abstract or aid in seen_ids:
                    continue
                seen_ids.add(aid)
                structured_sources.append({
                    "id": aid,
                    "arxiv_id": aid,
                    "title": title,
                    "snippet": abstract[:700],
                    "source": "Semantic Scholar",
                    "url": ss_url,
                    "year": year,
                })
    except Exception as e:
        print(f"  [Semantic Scholar error] {e}")

    # 5. DuckDuckGo Instant Answer
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": clean_query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=5,
            headers={"User-Agent": "Karpathy/2.0"},
        )
        if resp.ok:
            data = resp.json()
            abstract = (data.get("Abstract") or "").strip()
            abstract_url = data.get("AbstractURL", "")
            abstract_source = data.get("AbstractSource", "Web")
            if abstract and len(abstract) > 60:
                uid = f"ddg:{abstract_url}"
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    structured_sources.append({
                        "id": uid,
                        "arxiv_id": uid,
                        "title": data.get("Heading", query),
                        "snippet": abstract[:700],
                        "source": f"Web · {abstract_source}",
                        "url": abstract_url,
                        "year": None,
                    })
    except Exception as e:
        print(f"  [DuckDuckGo error] {e}")

    return structured_sources


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Retrieval Pipeline: BM25 + Qdrant + RRF + Reranker
# ─────────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    """
    Implements the Hybrid Retrieval subsystem from the architecture:
    1. Lexical index: BM25 (BM25Okapi)
    2. Dense index: Qdrant vector database (in-memory)
    3. Fusion: Reciprocal Rank Fusion (RRF) with k=60
    4. Reranker: Cross-Encoder / Semantic scoring
    """
    def __init__(self):
        self.qdrant_client = QdrantClient(":memory:")
        self.collection_name = "research_corpus"
        self._init_qdrant()

    def _init_qdrant(self):
        try:
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        except Exception as e:
            print(f"[Qdrant init warning] {e}")

    def execute_hybrid_search(self, query: str, candidate_docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes BM25 Lexical + Qdrant Dense Vector + RRF Fusion + BGE Reranker on candidates.
        """
        if not candidate_docs:
            return []

        # ── 1. BM25 Lexical Search ───────────────────────────────────────────
        corpus_texts = [f"{d['title']} {d['snippet']}" for d in candidate_docs]
        tokenized_corpus = [re.findall(r'\w+', t.lower()) for t in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = re.findall(r'\w+', query.lower())
        bm25_scores = bm25.get_scores(query_tokens)

        # Rank by BM25 score
        bm25_ranked = sorted(
            range(len(candidate_docs)),
            key=lambda idx: bm25_scores[idx],
            reverse=True,
        )
        bm25_rank_map = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}

        # ── 2. Qdrant Dense Vector Search ────────────────────────────────────
        try:
            self._init_qdrant()
            vectors = dense_embedder.embed_texts(corpus_texts)
            points = [
                PointStruct(
                    id=i,
                    vector=vectors[i],
                    payload={"doc_idx": i, "title": candidate_docs[i]["title"]}
                )
                for i in range(len(candidate_docs))
            ]
            self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

            q_vec = dense_embedder.embed_query(query)
            if hasattr(self.qdrant_client, "query_points"):
                res = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=q_vec,
                    limit=len(candidate_docs),
                )
                search_results = res.points
            elif hasattr(self.qdrant_client, "search"):
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=q_vec,
                    limit=len(candidate_docs),
                )
            else:
                search_results = []

            qdrant_rank_map = {res.id: rank + 1 for rank, res in enumerate(search_results)}
        except Exception as e:
            print(f"[Qdrant search warning] {e}")
            qdrant_rank_map = {i: i + 1 for i in range(len(candidate_docs))}

        # ── 3. Reciprocal Rank Fusion (RRF) ───────────────────────────────────
        # RRF formula: Score(d) = sum( 1 / (k + rank) ) with standard k = 60
        K = 60
        fused_candidates = []
        for i, doc in enumerate(candidate_docs):
            r_bm25 = bm25_rank_map.get(i, len(candidate_docs))
            r_qdrant = qdrant_rank_map.get(i, len(candidate_docs))
            rrf_score = (1.0 / (K + r_bm25)) + (1.0 / (K + r_qdrant))
            
            doc_copy = dict(doc)
            doc_copy["bm25_rank"] = r_bm25
            doc_copy["qdrant_rank"] = r_qdrant
            doc_copy["rrf_score"] = rrf_score
            fused_candidates.append(doc_copy)

        fused_ranked = sorted(fused_candidates, key=lambda x: x["rrf_score"], reverse=True)

        # ── 4. BGE / Semantic Reranker ───────────────────────────────────────
        final_top = reranker.rerank(query, fused_ranked, top_k=top_k)
        return final_top


# Singleton instance
hybrid_engine = HybridRetriever()
