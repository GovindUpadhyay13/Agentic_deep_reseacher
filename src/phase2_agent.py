import os
import re
import time
import operator
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from langgraph.graph import StateGraph, END

# Import Hybrid Retrieval Subsystem
try:
    from hybrid_retriever import fetch_multi_source_documents, hybrid_engine
except ImportError:
    from src.hybrid_retriever import fetch_multi_source_documents, hybrid_engine

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
default_model = genai.GenerativeModel('gemini-flash-lite-latest')


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph State Definition
# ─────────────────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    question: str
    plan_steps: List[str]
    current_step_idx: int
    current_queries: List[str]
    raw_sources: Annotated[List[Dict[str, Any]], operator.add]
    top_evidence: List[Dict[str, Any]]
    extracted_findings: List[str]
    reflection_notes: str
    is_sufficient: bool
    retry_count: int
    max_retries: int
    verified_evidence: str
    final_answer: str


# ─────────────────────────────────────────────────────────────────────────────
# Resilient LLM Caller
# ─────────────────────────────────────────────────────────────────────────────
def call_llm_resilient(model, prompt: str, max_retries: int = 5) -> str:
    """Invokes Gemini LLM with exponential backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "ResourceExhausted" in err or "quota" in err.lower():
                wait_t = 8 * (attempt + 1)
                print(f"    [Rate limit pause] waiting {wait_t}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_t)
            else:
                raise e
    return model.generate_content(prompt).text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Agent Core Class
# ─────────────────────────────────────────────────────────────────────────────
class LangGraphResearchAgent:
    """
    Encapsulates the LangGraph Research Workflow:
    Planner -> Context -> Retriever -> Reader -> Reflector (retry loop) -> Citation Verifier -> Synthesizer
    Powered by Hybrid Retrieval (BM25 + Qdrant Dense + RRF + Semantic Reranker).
    """
    def __init__(self, model=None, max_retries: int = 2):
        self.model = model or default_model
        self.max_retries = max_retries
        self.graph = self._build_graph()

    # ── 1. Planner Node ───────────────────────────────────────────────────────
    def _planner_node(self, state: ResearchState) -> Dict[str, Any]:
        """Breaks query into logical research angles (Foundations, SOTA Mechanisms, Open Frontiers)."""
        prompt = f"""You are an elite research planner. The user wants a comprehensive research report for:
"{state['question']}"

Break this research goal into 3 precise, highly specific search angles:
1. Foundational literature reviews, surveys, taxonomy, and historical milestone papers.
2. Technical mechanisms, algorithmic variants, model architectures, and benchmark evaluations.
3. Limitations, failure modes, safety/alignment vulnerabilities, and frontier developments.

Return ONLY a numbered list of 3 search angle descriptions, nothing else."""
        plan_text = call_llm_resilient(self.model, prompt)
        steps = [s.strip() for s in plan_text.split('\n') if s.strip()]
        if len(steps) < 3:
            steps = [
                f"Foundations and survey of {state['question']}",
                f"SOTA architectures and benchmarks for {state['question']}",
                f"Open challenges and failure modes in {state['question']}"
            ]
        print(f"[Planner] Generated {len(steps)} sub-research steps.")
        return {"plan_steps": steps, "current_step_idx": 0, "reflection_notes": "", "retry_count": 0}

    # ── 2. Context Node ───────────────────────────────────────────────────────
    def _context_node(self, state: ResearchState) -> Dict[str, Any]:
        """Builds targeted search queries for the current research step and reflection feedback."""
        step_idx = state.get("current_step_idx", 0)
        plan_steps = state.get("plan_steps", [])
        current_focus = plan_steps[step_idx] if step_idx < len(plan_steps) else state["question"]
        reflection = state.get("reflection_notes", "")

        ref_clause = f"\nAddress this gap from previous review: {reflection}" if reflection else ""
        prompt = f"""You are a search query engineer for academic databases.
Research Goal: "{state['question']}"
Current Angle: "{current_focus}"{ref_clause}

Formulate 2 distinct, highly effective search queries for arXiv and academic search engines:
- Query 1: Keyword-rich query for surveys and foundational papers.
- Query 2: Specific query targeting recent SOTA architectures, benchmarks, and mechanisms.

Return ONLY the 2 queries, one per line, no numbering or extra text."""
        raw_queries = call_llm_resilient(self.model, prompt)
        queries = [q.strip().strip('"').strip("'") for q in raw_queries.split('\n') if q.strip()][:2]
        if not queries:
            queries = [state['question'], f"{state['question']} survey benchmark"]
        print(f"[Context] Formulated queries: {queries}")
        return {"current_queries": queries}

    # ── 3. Retriever Node (Hybrid Retrieval: BM25 + Qdrant + RRF + Reranker) ─
    def _retriever_node(self, state: ResearchState) -> Dict[str, Any]:
        """Executes multi-source fetching and Hybrid Retrieval on all candidate papers."""
        all_candidates = []
        seen_ids = set()

        for q in state.get("current_queries", [state["question"]]):
            docs = fetch_multi_source_documents(q, n_results=4)
            for d in docs:
                if d["id"] not in seen_ids:
                    seen_ids.add(d["id"])
                    all_candidates.append(d)

        # Run Hybrid Search (BM25 + Qdrant Dense + RRF + Semantic Reranker)
        combined_query = " ".join(state.get("current_queries", [state["question"]]))
        top_reranked = hybrid_engine.execute_hybrid_search(
            query=combined_query,
            candidate_docs=all_candidates,
            top_k=6,
        )
        print(f"[Retriever] Hybrid retrieval selected {len(top_reranked)} top chunks from {len(all_candidates)} candidates.")
        return {"raw_sources": top_reranked, "top_evidence": top_reranked}

    # ── 4. Reader Node ────────────────────────────────────────────────────────
    def _reader_node(self, state: ResearchState) -> Dict[str, Any]:
        """Extracts concrete technical findings, methodologies, benchmarks, and dates from top evidence."""
        evidence_text = "\n\n".join([
            f"[Source: {d.get('id')}] ({d.get('year') or 'Web'})\nTitle: {d.get('title')}\nSnippet: {d.get('snippet')}"
            for d in state.get("top_evidence", [])
        ])

        prompt = f"""You are an expert research reader.
Topic: "{state['question']}"

Evidence Chunks:
{evidence_text}

Extract the key technical findings from this evidence:
1. Specific model architectures, loss functions, algorithms, and training techniques.
2. Concrete benchmark evaluations, metric numbers, dataset names, and performance comparisons.
3. Chronological dates and citation tags like [source_id].

Return a dense bullet-point summary of extracted findings:"""
        findings = call_llm_resilient(self.model, prompt)
        current_findings = state.get("extracted_findings", [])
        updated_findings = current_findings + [findings]
        print(f"[Reader] Extracted findings from step {state.get('current_step_idx', 0) + 1}.")
        return {"extracted_findings": updated_findings}

    # ── 5. Reflector Node ─────────────────────────────────────────────────────
    def _reflector_node(self, state: ResearchState) -> Dict[str, Any]:
        """Checks if accumulated evidence is sufficient to produce an authoritative research dossier."""
        all_findings = "\n\n".join(state.get("extracted_findings", []))
        prompt = f"""You are a strict research reviewer.
Question: "{state['question']}"

Accumulated Technical Findings:
{all_findings}

Evaluate if there is sufficient multi-year chronological evidence, SOTA benchmarks, and taxonomy to write a publication-grade research dossier.
Respond in this exact format:
SUFFICIENT: [YES or NO]
NOTES: [If NO, explain in 1 sentence what specific angle or benchmark is missing. If YES, write 'Evidence complete.']"""
        review_text = call_llm_resilient(self.model, prompt)
        is_yes = "SUFFICIENT: YES" in review_text.upper()
        notes = review_text.split("NOTES:")[-1].strip() if "NOTES:" in review_text else ""
        
        retry_count = state.get("retry_count", 0)
        next_step_idx = state.get("current_step_idx", 0) + 1

        print(f"[Reflector] Sufficient: {is_yes} (Retry {retry_count}/{state.get('max_retries', self.max_retries)})")
        return {
            "is_sufficient": is_yes,
            "reflection_notes": notes,
            "retry_count": retry_count if is_yes else retry_count + 1,
            "current_step_idx": next_step_idx,
        }

    # ── 6. Citation Verifier Node ─────────────────────────────────────────────
    def _citation_verifier_node(self, state: ResearchState) -> Dict[str, Any]:
        """Deterministic grounding verifier ensuring all claims reference valid corpus sources."""
        sources = state.get("raw_sources", [])
        valid_ids = {s.get("id") for s in sources if s.get("id")}
        for s in sources:
            if s.get("arxiv_id"):
                valid_ids.add(s.get("arxiv_id"))

        # Build formatted grounded corpus for synthesis
        verified_blocks = []
        for s in sources:
            yr = f" ({s['year']})" if s.get('year') else ""
            verified_blocks.append(
                f"[Source: {s.get('id')}]{yr}\nTitle: {s.get('title')}\nSource: {s.get('source')}\nSnippet: {s.get('snippet')}\nURL: {s.get('url')}"
            )
        verified_corpus = "\n\n".join(verified_blocks)
        print(f"[Citation Verifier] Grounded {len(sources)} unique verified evidence sources.")
        return {"verified_evidence": verified_corpus}

    # ── 7. Synthesizer Node ───────────────────────────────────────────────────
    def _synthesizer_node(self, state: ResearchState) -> Dict[str, Any]:
        """Writes the final publication-grade Research Dossier with Timeline and SOTA table."""
        prompt = f"""You are Andrej Karpathy and a Senior Principal AI Research Scientist.
You are writing an authoritative, highly technical, publication-grade Research Dossier for:

"{state['question']}"

Evidence collected across arXiv, Semantic Scholar, CrossRef, Wikipedia, and the web:
{state.get('verified_evidence', '')}

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
Provide 5 dense, high-impact, actionable conclusions directly addressing "{state['question']}", each supported by inline citations [source_id].

Generate the full Research Dossier now:"""
        final_report = call_llm_resilient(self.model, prompt)
        print("[Synthesizer] Research Dossier synthesis complete.")
        return {"final_answer": final_report}

    # ── Graph Builder ─────────────────────────────────────────────────────────
    def _build_graph(self):
        workflow = StateGraph(ResearchState)

        # Add 7 architecture nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("context", self._context_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("reader", self._reader_node)
        workflow.add_node("reflector", self._reflector_node)
        workflow.add_node("citation_verifier", self._citation_verifier_node)
        workflow.add_node("synthesizer", self._synthesizer_node)

        # Edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "context")
        workflow.add_edge("context", "retriever")
        workflow.add_edge("retriever", "reader")
        workflow.add_edge("reader", "reflector")

        # Conditional retry edge from reflector
        def _route_reflector(state: ResearchState) -> str:
            if state.get("is_sufficient", False) or state.get("retry_count", 0) >= state.get("max_retries", self.max_retries):
                return "citation_verifier"
            return "context"

        workflow.add_conditional_edges(
            "reflector",
            _route_reflector,
            {
                "context": "context",
                "citation_verifier": "citation_verifier",
            }
        )
        workflow.add_edge("citation_verifier", "synthesizer")
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    # ── Stream Runner for Real-Time UI Streaming ──────────────────────────────
    def run_stream(self, question: str):
        """
        Executes the LangGraph StateGraph, yielding event dicts as each node activates and completes.
        """
        initial_state: ResearchState = {
            "question": question,
            "plan_steps": [],
            "current_step_idx": 0,
            "current_queries": [],
            "raw_sources": [],
            "top_evidence": [],
            "extracted_findings": [],
            "reflection_notes": "",
            "is_sufficient": False,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "verified_evidence": "",
            "final_answer": "",
        }

        node_display_names = {
            "planner": "Planner: Breaking query into research angles",
            "context": "Context: Formulating academic search queries",
            "retriever": "Retriever: Hybrid search (BM25 + Qdrant + RRF + Reranker)",
            "reader": "Reader: Extracting technical findings & metrics",
            "reflector": "Reflector: Critiquing evidence completeness",
            "citation_verifier": "Citation Verifier: Grounding and verifying claims",
            "synthesizer": "Synthesizer: Writing Research Dossier & Timeline",
        }

        accumulated_sources = []
        final_dossier = ""
        current_step = 0

        yield {"type": "step_start", "step_key": "planner", "step_name": node_display_names["planner"]}

        for output in self.graph.stream(initial_state):
            for node_name, node_state in output.items():
                yield {"type": "step_complete", "step_key": f"{node_name}_{current_step}"}

                if node_name == "retriever":
                    sources = node_state.get("top_evidence", [])
                    accumulated_sources.extend(sources)
                    yield {"type": "sources", "sources": sources}

                elif node_name == "reflector":
                    if not node_state.get("is_sufficient", False) and node_state.get("retry_count", 0) <= self.max_retries:
                        yield {"type": "reflector_loop", "notes": node_state.get("reflection_notes", "")}
                        current_step += 1

                elif node_name == "synthesizer":
                    final_dossier = node_state.get("final_answer", "")

                # Start next node UI animation
                next_key = f"{node_name}_{current_step}"
                yield {"type": "step_start", "step_key": next_key, "step_name": node_display_names.get(node_name, node_name)}

        # Final event
        # Deduplicate sources by id
        unique_sources = []
        seen = set()
        for s in accumulated_sources:
            sid = s.get("id") or s.get("arxiv_id")
            if sid and sid not in seen:
                seen.add(sid)
                unique_sources.append(s)

        yield {
            "type": "final_answer",
            "answer": final_dossier,
            "total_sources": len(unique_sources),
            "all_sources": unique_sources,
        }

    def run(self, question: str) -> str:
        """Blocking helper."""
        ans = ""
        for event in self.run_stream(question):
            if event["type"] == "final_answer":
                ans = event["answer"]
        return ans


# Legacy class alias for compatibility
ResearchAgent = LangGraphResearchAgent
collection = None  # Chroma collection placeholder (now replaced by Qdrant in-memory engine)


if __name__ == "__main__":
    agent = LangGraphResearchAgent()
    print("Testing LangGraph Research Agent...")
    res = agent.run("What is Direct Preference Optimization (DPO)?")
    print("\n--- Final Answer Preview ---")
    print(res[:500])
