# Corrective RAG (CRAG) with LangGraph

A Corrective Retrieval-Augmented Generation pipeline built with **LangGraph**. Instead of blindly trusting whatever the vector store returns, the system **grades the retrieved context** with an LLM and, when the context is weak, **falls back to live web search** before generating an answer.

The knowledge base is a PDF textbook (*Introduction to Machine Learning with Python*), indexed locally with FAISS.

---

## How it works

```
                    ┌──────────┐
        START ───►  │ retrieve │  (FAISS, top-4 chunks)
                    └────┬─────┘
                         ▼
                ┌──────────────────┐
                │ evaluate_context │  LLM scores relevance 0–1
                └────────┬─────────┘
                         │
          ┌──────────────┴───────────────┐
   verdict == CORRECT             INCORRECT / AMBIGUOUS
          │                              │
          │                     ┌────────▼────────┐
          │                     │  rewrite_query  │  question → keyword web query
          │                     └────────┬────────┘
          │                     ┌────────▼────────┐
          │                     │    websearch    │  Tavily (5 results)
          │                     └────────┬────────┘
          ▼                              ▼
                    ┌──────────────────┐
                    │      refine      │  keep only relevant info
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │     generate     │  answer only from context
                    └────────┬─────────┘
                             ▼
                            END
```

### Verdict logic

| Score        | Verdict     | Context used for refinement          |
|--------------|-------------|--------------------------------------|
| `> 0.7`      | `CORRECT`   | Retrieved PDF chunks only            |
| `0.3 – 0.7`  | `AMBIGUOUS` | PDF chunks **+** web results         |
| `< 0.3`      | `INCORRECT` | Web results only                     |

### Nodes

| Node               | Purpose |
|--------------------|---------|
| `retrieve`         | Similarity search over the FAISS index (`k=4`). |
| `evaluate_context` | Structured-output LLM call (`ContextAssessment`) returning a score and reason, mapped to a verdict. |
| `rewrite_query`    | Turns the question into a short keyword web query (6–14 words, adds a recency constraint if relevant). |
| `websearch`        | Tavily search, results wrapped as `Document`s with title/URL metadata. |
| `refine`           | Extracts only relevant information from the chosen context, preserving meaning and source IDs. |
| `generate`         | Answers strictly from the refined context; says "I don't know" otherwise. |

---

## Tech stack

- **Orchestration:** LangGraph, LangChain
- **Main LLM (evaluator / rewriter / refiner / generator):** Groq `openai/gpt-oss-20b`
- **Secondary LLM:** Google Gemini `gemini-2.5-flash` (initialized as `refine_llm`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs locally)
- **Vector store:** FAISS (persisted to `faiss_index/`)
- **Web search:** Tavily
- **PDF loading:** PyPDF
- **Chunking:** `RecursiveCharacterTextSplitter` (900 chars, 150 overlap)

---

## Project structure

```
.
├── crag.py / crag.ipynb      # Main pipeline (cell-style script with # %% markers)
├── faiss_index/              # Auto-created on first run
├── .env                      # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create an environment

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
GOOGLE_API_KEY=your_google_gemini_key
```

| Key              | Get it from |
|------------------|-------------|
| `GROQ_API_KEY`   | https://console.groq.com |
| `TAVILY_API_KEY` | https://tavily.com |
| `GOOGLE_API_KEY` | https://aistudio.google.com |

### 4. Add your PDF

Update the path in the loader to point to your own PDF:

```python
loader = PyPDFLoader("path/to/your/document.pdf")
```

> The original code uses a hard-coded Windows path (`A:/langgraph_tutorial/...`). Change this before running on your machine.

---

## Usage

Run the script/notebook cell by cell, or call the compiled graph directly:

```python
res = app.invoke({"query": "What is Batch Normalization"})

print("verdict:", res["verdict"])
print("reason:", res["reason"])
print("web_query:", res["web_query"])
print("answer:", res["answer"])
```

### Example

A question the textbook doesn't cover well (e.g. *"What is Batch Normalization?"*) should score low, get marked `INCORRECT`, trigger a web search, and be answered from web results. A question the book covers directly (e.g. *"What is machine learning?"*) should be `CORRECT` and answered from the PDF alone.

### Inspecting intermediate state

The graph returns the full state, so you can debug each stage:

```python
res["og_context"]        # context before refinement
res["refined_context"]   # context after refinement
res["web_docs"]          # raw Tavily results (when web search ran)
```

---

## Notes and caveats

- **First run builds the FAISS index** (embeds every chunk) and saves it to `faiss_index/`. Later runs load it from disk. Delete the folder if you change the PDF or chunking settings.
- `allow_dangerous_deserialization=True` is required to load a local FAISS index. Only load indexes you created yourself.
- The `web_query` key is only set when the web branch runs. On a `CORRECT` verdict, `res["web_query"]` will raise a `KeyError`; use `res.get("web_query")`.
- `ChatGoogleGenerativeAI` is initialized at import time, so `GOOGLE_API_KEY` must be set even though `refine_llm` is not yet wired into the graph.
- `TavilySearchResults` from `langchain_community` is deprecated in newer LangChain versions; the maintained replacement is the `langchain-tavily` package (`TavilySearch`).
- `requirements.txt` is unpinned. Once everything works, run `pip freeze > requirements.txt` to lock the exact versions that worked for you.

---

## Possible improvements

- Use `refine_llm` (Gemini) for the refinement step to reduce load on the Groq rate limit
- Add a per-chunk relevance grader instead of grading the whole context at once
- Cite sources (PDF page numbers / web URLs) in the final answer
- Add a Streamlit or FastAPI front end
- Evaluate with a golden dataset (e.g. RAGAS or DeepEval) to compare CRAG against plain RAG

---

## Acknowledgements

Based on the Corrective RAG approach: *Corrective Retrieval Augmented Generation* (Yan et al., 2024).