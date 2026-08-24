import os
import time

from dotenv import load_dotenv  # To load environment variables from a .env file, keeping our API key secure and out of the codebase

# Load environment variables FIRST, before importing Chroma
load_dotenv()  # This will look for a .env file in the current directory and load the variables into the environment

import chromadb

# THE MONKEY PATCH
# Reach into Chroma's internal posthog class and overwrite the broken capture method
# with a lambda that accepts anything and does nothing.
try:
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass

from chromadb.utils import embedding_functions  # Provides various embedding functions, including SentenceTransformer
import google.generativeai as genai  # Gemini API client library

genai.configure(api_key=os.environ["GEMINI_API_KEY"])  # Configure the Gemini API client with our API key from the environment variable

# Initialize the Gemini Model (Flash is extremely fast, perfect for agentic loops)
model = genai.GenerativeModel('gemini-flash-lite-latest')

# Connect to our existing local ChromaDB
DATA_DIR = "./data"  # Directory to store PDFs and the ChromaDB database
DB_DIR = os.path.join(DATA_DIR, "chroma_db")  # Directory where our ChromaDB database is stored (created in phase1_index.py)
embed_fn = embedding_functions.DefaultEmbeddingFunction()  # ONNX-based (all-MiniLM-L6-v2), no torch dependency, embeddings are compatible with the index built in phase1_index.py

collection = None  # Initialized to None; set below if the index exists

try:  # Test connection to ChromaDB and print the number of documents in the collection
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(name="arxiv_papers", embedding_function=embed_fn)
    print(f"Successfully connected to ChromaDB. Found {collection.count()} chunks.")
except Exception as e:  # Handle exceptions (e.g. index not yet built — run phase1_index.py first)
    print(f"Error connecting to ChromaDB: {e}")


def retrieve_documents(query: str, n_results: int = 5):
    """
    Multi-source live retriever — searches across:
      1. arXiv (primary, academic papers)
      2. Semantic Scholar (secondary, broad academic coverage)
      3. CrossRef (journal articles with publication year — critical for timeline)
      4. Wikipedia (survey/overview context)
      5. DuckDuckGo Instant Answer (web articles & general context)
      6. Local ChromaDB (bonus, if index has been built)

    Returns:
      structured_sources: list of {arxiv_id, title, snippet, source, url, year}
      evidence_str: flat string for LLM prompts (includes year metadata)
    """
    import re as _re
    import requests as _req
    structured_sources = []
    seen_ids: set = set()

    clean_query = _re.sub(r'[\"\'\(\)]', ' ', query).strip()
    clean_query = _re.sub(r'\s+', ' ', clean_query)

    # ── 1. arXiv live search ───────────────────────────────────────────────
    try:
        import arxiv as _arxiv
        _client = _arxiv.Client(page_size=n_results, delay_seconds=0.5, num_retries=2)
        _search = _arxiv.Search(
            query=clean_query,
            max_results=n_results,
            sort_by=_arxiv.SortCriterion.Relevance,
        )
        for result in _client.results(_search):
            aid = result.get_short_id()
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            year = result.published.year if result.published else None
            structured_sources.append({
                "arxiv_id": aid,
                "title": result.title,
                "snippet": result.summary[:600].strip(),
                "source": "arXiv",
                "url": f"https://arxiv.org/abs/{aid}",
                "year": year,
            })
            print(f"  [arXiv] {year} {aid}: {result.title[:55]}...")
    except Exception as e:
        print(f"  [arXiv search error] {e}")

    # ── 2. Wikipedia — survey / overview context ───────────────────────────
    try:
        search_resp = _req.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": clean_query,
                "format": "json",
                "srlimit": 2,
            },
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
                summary_resp = _req.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}",
                    timeout=5,
                )
                if summary_resp.ok:
                    data = summary_resp.json()
                    extract = data.get("extract", "")[:600]
                    wiki_url = data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}")
                    structured_sources.append({
                        "arxiv_id": uid,
                        "title": page_title,
                        "snippet": extract,
                        "source": "Wikipedia",
                        "url": wiki_url,
                        "year": None,
                    })
                    print(f"  [Wikipedia] {page_title}")
    except Exception as e:
        print(f"  [Wikipedia error] {e}")

    # ── 3. CrossRef — journal articles with publication year ───────────────
    try:
        resp = _req.get(
            "https://api.crossref.org/works",
            params={
                "query": clean_query,
                "rows": n_results,
                "sort": "relevance",
                "select": "DOI,title,abstract,published,URL,type,container-title",
            },
            timeout=5,
            headers={"User-Agent": "Karpathy/1.0 (mailto:research@karpathy.ai)"},
        )
        if resp.ok:
            for item in resp.json().get("message", {}).get("items", []):
                titles = item.get("title") or []
                title = titles[0].strip() if titles else ""
                doi = item.get("DOI", "")
                abstract_raw = (item.get("abstract") or "").strip()
                abstract = _re.sub(r"<[^>]+>", " ", abstract_raw).strip()
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
                snippet = abstract[:600] if abstract else f"Published in {journal} ({year})"
                structured_sources.append({
                    "arxiv_id": doi or uid,
                    "title": title,
                    "snippet": snippet,
                    "source": "CrossRef" + (f" · {journal[:25]}" if journal else ""),
                    "url": url,
                    "year": year,
                })
                print(f"  [CrossRef] {year} {doi[:20] if doi else ''}: {title[:55]}...")
    except Exception as e:
        print(f"  [CrossRef error] {e}")

    # ── 4. Semantic Scholar live search ────────────────────────────────────
    try:
        resp = _req.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": clean_query,
                "fields": "title,abstract,externalIds,url,year",
                "limit": n_results,
            },
            timeout=5,
            headers={"User-Agent": "Karpathy/1.0"},
        )
        if resp.ok:
            for paper in resp.json().get("data", []):
                title = (paper.get("title") or "").strip()
                abstract = (paper.get("abstract") or "").strip()
                ext_ids = paper.get("externalIds") or {}
                aid = ext_ids.get("ArXiv") or paper.get("paperId", "")
                year = paper.get("year")
                ss_url = paper.get("url") or (
                    f"https://arxiv.org/abs/{aid}" if ext_ids.get("ArXiv")
                    else f"https://www.semanticscholar.org/paper/{paper.get('paperId','')}"
                )
                if not title or not abstract or aid in seen_ids:
                    continue
                seen_ids.add(aid)
                structured_sources.append({
                    "arxiv_id": aid,
                    "title": title,
                    "snippet": abstract[:600],
                    "source": "Semantic Scholar",
                    "url": ss_url,
                    "year": year,
                })
                print(f"  [S2] {year} {aid[:20]}: {title[:55]}...")
    except Exception as e:
        print(f"  [Semantic Scholar error] {e}")

    # ── 5. DuckDuckGo Instant Answer — web articles & context ─────────────
    try:
        resp = _req.get(
            "https://api.duckduckgo.com/",
            params={"q": clean_query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=5,
            headers={"User-Agent": "Karpathy/1.0"},
        )
        if resp.ok:
            data = resp.json()
            abstract = (data.get("Abstract") or "").strip()
            abstract_url = data.get("AbstractURL", "")
            abstract_source = data.get("AbstractSource", "Web")
            if abstract and len(abstract) > 80:
                uid = f"ddg:{abstract_url}"
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    structured_sources.append({
                        "arxiv_id": uid,
                        "title": data.get("Heading", query),
                        "snippet": abstract[:600],
                        "source": f"Web · {abstract_source}",
                        "url": abstract_url,
                        "year": None,
                    })
    except Exception as e:
        print(f"  [DuckDuckGo error] {e}")

    # ── 6. Local ChromaDB (bonus if the index was built) ───────────────────
    if collection is not None and collection.count() > 0:
        try:
            db_results = collection.query(query_texts=[query], n_results=3)
            for i in range(len(db_results['documents'][0])):
                doc_text = db_results['documents'][0][i]
                meta = db_results['metadatas'][0][i]
                aid = meta.get('arxiv_id', 'unknown')
                title = meta.get('title', aid)
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)
                structured_sources.append({
                    "arxiv_id": aid,
                    "title": title,
                    "snippet": doc_text[:600].strip(),
                    "source": "Local Index",
                    "url": f"https://arxiv.org/abs/{aid}",
                    "year": None,
                })
                print(f"  [ChromaDB] {aid}: {title[:55]}...")
        except Exception as e:
            print(f"  [ChromaDB error] {e}")

    # ── Build flat evidence string for LLM (includes year for timeline) ────
    formatted_strings = []
    for s in structured_sources:
        year_str = f" ({s['year']}" + ")" if s.get('year') else ""
        formatted_strings.append(
            f"[Source: {s['arxiv_id']}]{year_str}\nTitle: {s['title']}\nFrom: {s['source']}\n{s['snippet']}\n"
        )
    print(f"  Retriever: Found {len(structured_sources)} sources total.")
    return structured_sources, "\n".join(formatted_strings)


class ResearchAgent:
    '''
    This class encapsulates the entire agentic loop, including planning, retrieval, reflection, and synthesis.
    The agent will use the Gemini LLM for all its reasoning and generation tasks, and it will use the
    retrieve_documents function to get information from our ChromaDB when needed.
    '''
    def __init__(self, model, collection, max_steps=3, use_planner=True, use_reflector=True, use_verifier=True):
        self.model = model
        self.collection = collection
        # If we have no reflector, we only ever do 1 step
        self.max_steps = max_steps if use_reflector else 1

        # ABLATION FEATURE FLAGS
        self.use_planner = use_planner
        self.use_reflector = use_reflector
        self.use_verifier = use_verifier

    def _call_llm(self, prompt):
        """A resilient wrapper that automatically handles API rate limits."""
        for attempt in range(5):  # Try up to 5 times
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:  # Catch any exception (including rate limits, timeouts, etc.)
                if "429" in str(e) or "ResourceExhausted" in str(e) or "quota" in str(e).lower():
                    wait_time = 10 * (attempt + 1)  # Wait 10s, then 20s, then 30s...
                    error_msg = str(e).splitlines()[0][:100]
                    print(f"    [API Rate Limit Hit] {error_msg}...")
                    print(f"    Pausing for {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise e  # If it's a different error, crash normally

        # If it fails 5 times, try one last time and let it crash if it fails
        return self.model.generate_content(prompt).text.strip()

    def _planner(self, question, step=0):
        """Phase A: The agent plans queries targeting foundational surveys, mechanisms, and SOTA benchmarks."""
        if step == 0:
            prompt = f"""You are a research planner. The user wants to research: "{question}"
Write a targeted search query to find survey papers, literature reviews, or foundational breakthrough papers on this topic in arXiv / academic repositories.
Return ONLY the search query, with no quotes or extra text."""
        elif step == 1:
            prompt = f"""You are a research planner. The user wants to research: "{question}"
Write a targeted search query to find SOTA benchmark evaluations, model comparisons, empirical metrics, and recent technical architectures on this topic.
Return ONLY the search query, with no quotes or extra text."""
        else:
            prompt = f"""You are a research planner. The user wants to research: "{question}"
Write a targeted search query to find open challenges, limitations, failure modes, and latest 2024-2026 developments on this topic.
Return ONLY the search query, with no quotes or extra text."""
        return self._call_llm(prompt)

    def _reflector(self, question, gathered_evidence):
        """Phase B: The agent critiques its own research so far."""
        prompt = f"""You are a harsh academic peer reviewer.
Question: "{question}"
Evidence gathered so far:
{gathered_evidence}

Does the evidence contain enough specific technical facts, benchmarks, and multi-year chronological papers to build a rigorous research timeline and SOTA comparison?
Reply with exactly 'YES' or 'NO'."""
        return 'YES' in self._call_llm(prompt).upper()

    def _synthesizer(self, question, gathered_evidence):
        """Phase C: Produces a high-density, authoritative Karpathy-style Research Dossier."""
        prompt = f"""You are Andrej Karpathy and a Senior Principal AI Research Scientist.
You are writing an authoritative, highly technical, publication-grade Research Dossier for:

"{question}"

Evidence collected across arXiv, Semantic Scholar, CrossRef, Wikipedia, and the web:
{gathered_evidence}

CRITICAL NEGATIVE CONSTRAINTS:
- NEVER output conversational filler (e.g., "Based on the provided evidence", "Here is a summary", "In this report").
- START IMMEDIATELY with the first section header: `## 🔍 Survey & Foundation`.
- Every factual claim MUST cite its exact evidence source tag inline: `[source_id]` (e.g. `[2405.27355v2]` or `[10.1000/xyz]`).
- Do NOT hallucinate papers, dates, or metrics not grounded in the evidence.

MANDATORY DOSSIER STRUCTURE (Follow EXACT Markdown syntax):

## 🔍 Survey & Foundation
Synthesize the foundational paradigm and scope. If survey/review papers or foundational milestone works exist in the evidence, analyze their taxonomy and theoretical underpinnings in 2-3 deep paragraphs. Cite all relevant papers inline [source_id].

## 📅 Chronological Research Timeline
Break down the evolution year-by-year based on the evidence. For EACH year that appears in the evidence, create a `### [Year]` subsection. Under each year, list the key papers/contributions:

### [Year]
**[Paper/System Title]** — [arXiv / CrossRef / Web]
- **Core Mechanism & Objective:** Technical details of the architecture, loss function, algorithm, or methodology. [source_id]
- **Empirical Findings & Metrics:** Concrete benchmark numbers, dataset sizes, score improvements, or ablation results. [source_id]
- **Paradigm Impact:** Why this was a milestone in the trajectory of the field.

(Repeat for all years present in the evidence. Include undated web sources in `### Undated / Web Sources` at the bottom of the timeline.)

## 🤖 SOTA Models & Benchmark Comparison
Analyze the leading state-of-the-art models and algorithmic variants discovered in the evidence.

Include a Markdown comparison table:
| Model / Method | Year | Primary Architecture / Mechanism | Key Benchmark & Result | Primary Citation |
|---|---|---|---|---|
| [Name] | [Year] | [Mechanism] | [Benchmark Score/Metric] | [source_id] |

Below the table, provide deep-dive technical breakdowns for key leading systems:
### [Model/System Name] ([Year])
- **Technical Innovation:** Deep explanation of the novel technique (e.g., training recipe, preference optimization, reward modeling, scaling laws). [source_id]
- **Performance:** Exact benchmark evaluations, Win-rates, or empirical comparisons. [source_id]

## 🔬 Frontier, Failure Modes & Open Problems
Synthesize the critical limitations, vulnerabilities, and open research questions identified in the evidence (e.g., reward tampering, alignment faking, sycophancy, out-of-distribution generalization, compute bottlenecks). 2-3 rigorous paragraphs.

## 💡 Key Takeaways & Synthesis
Provide 5 dense, high-impact, actionable conclusions directly addressing "{question}", each supported by inline citations [source_id].

Generate the full Research Dossier now:"""
        return self._call_llm(prompt)

    def _citation_verifier(self, draft_answer, gathered_evidence):
        """Phase D: Verifies citations deterministically to guarantee formatting preservation."""
        import re
        # Extract valid source IDs from evidence
        valid_ids = set()
        for match in re.finditer(r'\[Source:\s*([^\]]+)\]', gathered_evidence):
            valid_ids.add(match.group(1).strip())
        for match in re.finditer(r'\[Paper:\s*([^\]]+)\]', gathered_evidence):
            valid_ids.add(match.group(1).strip())
        for match in re.finditer(r'\[(\d{4}\.\d{4,6}(?:v\d+)?)\]', gathered_evidence):
            valid_ids.add(match.group(1).strip())

        # If evidence had IDs, verify inline citations; keep text intact
        def _check_cite(match):
            cite = match.group(1).strip()
            # If cite is valid or matches pattern, keep it
            if cite in valid_ids or any(v in cite or cite in v for v in valid_ids):
                return f"[{cite}]"
            # If purely hallucinated ID not in evidence, remove tag but keep sentence
            return ""

        verified = re.sub(r'\[([a-zA-Z0-9\.\_\:\-\/]{4,40})\]', _check_cite, draft_answer)
        return verified if verified.strip() else draft_answer

    def run_stream(self, question):
        """
        Generator version of run(). Yields structured event dicts at each phase transition
        so the UI can animate steps in real time.

        Event types:
          {"type": "step_start",    "step_key": str, "step_name": str}
          {"type": "step_complete", "step_key": str}
          {"type": "sources",       "sources": list[dict]}
          {"type": "reflector_loop"}
          {"type": "final_answer",  "answer": str}
          {"type": "error",         "message": str}
        """
        print(f"\n[Agent Started] Task: {question}")
        gathered_evidence = ""
        total_sources = []

        # ── Understanding phase ──────────────────────────────────────────────
        yield {"type": "step_start", "step_key": "understand", "step_name": "Understanding your question"}

        for step in range(self.max_steps):
            print(f"\\ Step {step + 1}")

            # ── Planner ──────────────────────────────────────────────────────
            yield {"type": "step_complete", "step_key": "understand"}

            if self.use_planner:
                yield {"type": "step_start", "step_key": f"plan_{step}", "step_name": "Planning the search strategy"}
                search_query = self._planner(question, step=step)
                print(f"Planner: '{search_query}'")
                yield {"type": "step_complete", "step_key": f"plan_{step}"}
                search_label = f"Searching '{search_query}'"
            else:
                search_query = question
                print(f"Bypassing Planner. Using raw query: '{search_query}'")
                search_label = f"Searching (planner off): '{question[:60]}...'" if len(question) > 60 else f"Searching (planner off): '{question}'"

            time.sleep(3)  # Rate limit pause — load-bearing for Gemini free tier

            # ── Retrieval ────────────────────────────────────────────────────
            yield {"type": "step_start", "step_key": f"search_{step}", "step_name": search_label}

            structured_sources, new_evidence = retrieve_documents(search_query, n_results=3)
            total_sources.extend(structured_sources)
            print(f"Retriever: Found {len(structured_sources)} chunks.")

            yield {"type": "sources", "sources": structured_sources}
            yield {"type": "step_complete", "step_key": f"search_{step}"}

            gathered_evidence += f"\nSearch Query: {search_query}\nResults:\n{new_evidence}\n"

            # ── Reflector ────────────────────────────────────────────────────
            if self.use_reflector:
                yield {"type": "step_start", "step_key": f"reflect_{step}", "step_name": "Double-checking the evidence"}
                is_sufficient = self._reflector(question, gathered_evidence)

                if is_sufficient:
                    print("Reflector: Evidence sufficient. Breaking loop.")
                    yield {"type": "step_complete", "step_key": f"reflect_{step}"}
                    break
                else:
                    print("Reflector: Evidence insufficient. Searching again...")
                    yield {"type": "step_complete", "step_key": f"reflect_{step}"}
                    if step < self.max_steps - 1:
                        yield {"type": "reflector_loop"}
                        yield {"type": "step_start", "step_key": "understand", "step_name": "Refining the search"}
                    time.sleep(3)
            else:
                print("Bypassing Reflector. Moving directly to synthesis.")
                break  # Exit the loop immediately after one retrieval

        # ── Synthesis ────────────────────────────────────────────────────────
        print("\\ Final Synthesis")
        if not gathered_evidence.strip():
            yield {"type": "error", "message": "No evidence found in the index."}
            return

        time.sleep(3)
        yield {"type": "step_start", "step_key": "synthesize", "step_name": "Writing the answer"}
        draft_answer = self._synthesizer(question, gathered_evidence)
        print("Draft generated.")
        yield {"type": "step_complete", "step_key": "synthesize"}

        # ── Citation Verifier ─────────────────────────────────────────────────
        if self.use_verifier:
            yield {"type": "step_start", "step_key": "verify", "step_name": "Verifying citations"}
            print("Verifying citations...")
            time.sleep(3)
            final_answer = self._citation_verifier(draft_answer, gathered_evidence)
            yield {"type": "step_complete", "step_key": "verify"}
        else:
            print("Bypassing Verifier. Using raw draft.")
            final_answer = draft_answer

        print("\n[Final Answer]")
        print(final_answer)

        yield {"type": "final_answer", "answer": final_answer, "total_sources": len(total_sources), "all_sources": total_sources}

    def run(self, question):
        """
        The original blocking run() method — kept intact for __main__ test block,
        phase3_eval.py, and any other callers. Internally drains run_stream().
        """
        final_answer = "No answer produced."
        for event in self.run_stream(question):
            if event["type"] == "final_answer":
                final_answer = event["answer"]
            elif event["type"] == "error":
                final_answer = event["message"]
        return final_answer


if __name__ == "__main__":
    # Test the Agent
    test_question = "What is the SWE-agent, and what does it do?"

    agent = ResearchAgent(model=model, collection=collection, max_steps=3)
    agent.run(test_question)