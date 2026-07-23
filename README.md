# NeuralRays RAG Chatbot

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot prototype for the NeuralRays AI website.

The chatbot answers user questions using information extracted from the NeuralRays website. It crawls the site, cleans the content, creates chunks, generates local embeddings, stores them in ChromaDB, retrieves relevant context, reranks results, and returns grounded answers with source URLs.

The project has also been extended with enterprise RAG features including:

- Domain-agnostic evaluation framework
- Semantic/layout-aware chunking
- PII protection pipeline
- Reranker adapter
- ACL-aware source connector
- MCP-style source server
- Production/DataOps/LLMOps documentation

---

## Objective

The original objective was to build a Week 1 RAG chatbot prototype for the NeuralRays website.

The extended objective is to make the prototype closer to an enterprise RAG system by adding:

- Safe PII handling
- Reusable evaluation
- Better retrieval quality
- Source permissions/ACL handling
- Production architecture planning

---

## Example Questions

The chatbot can answer questions such as:

- What does NeuralRays do?
- What AI services does NeuralRays offer?
- Does NeuralRays offer cloud transformation?
- How can I contact NeuralRays?
- Who is the CEO of NeuralRays?
- Does NeuralRays list pricing?
- What success stories does NeuralRays describe?

---

## Current Features

### Core RAG Pipeline

- Python-based website crawler
- Website text extraction and cleaning
- Duplicate page handling
- Structured website content saved to JSON
- Semantic/layout-aware chunking
- Local embedding generation using `sentence-transformers`
- ChromaDB vector database storage
- Local retrieval-based chatbot logic
- Reranker adapter for improved result ordering
- Source-grounded answer generation
- Source URLs returned with chatbot answers
- Handling for services, contact, pricing, success stories and team questions
- FastAPI `/chat` API endpoint
- Browser-based chatbot interface

### Evaluation Framework

- Domain-agnostic evaluation framework
- Tenant-based evaluation
- Auto-generated evaluation questions
- Local metric adapter
- Evaluation result JSON output
- HTML dashboard generation
- Quality metrics:
  - faithfulness
  - correctness
  - context precision
  - answer relevancy
  - hallucination
  - source presence
- Operational metrics:
  - retrieval latency
  - generation latency
  - total latency
  - estimated prompt tokens
  - estimated completion tokens
  - estimated total tokens
  - estimated cost
  - overall RAG score

### PII Protection

- PII detection
- Partial masking
- Full masking
- Tokenisation with placeholders
- SQLite token store for reversible token mapping
- Ingestion-time PII guard
- Prompt sanitisation
- Output validation
- Audit logging without storing raw PII in the audit log

### Source Connector / ACL

- Mock enterprise source connector
- Sample enterprise documents
- ACL metadata using allowed users and allowed groups
- Permission-aware retrieval
- Deny-by-default ACL logic
- MCP-style source server
- Enterprise source sync into ChromaDB
- Separate ChromaDB collection for enterprise documents

### Documentation

- Chunking and retrieval strategy study
- Production deployment, DataOps and LLMOps architecture document

---

## Tech Stack

- Python
- BeautifulSoup
- Requests
- ChromaDB
- Sentence Transformers
- FastAPI
- Uvicorn
- Pydantic
- python-dotenv
- SQLite
- Pytest

---

## RAG Pipeline

The project follows this RAG workflow:

```text
Website content
→ crawl pages
→ clean text
→ create semantic/layout-aware chunks
→ generate local embeddings
→ store in ChromaDB
→ retrieve candidate chunks
→ rerank chunks
→ build grounded answer
→ return answer with sources
```

Enterprise-style retrieval adds:

```text
User query
→ retrieve candidates
→ apply ACL filtering
→ rerank authorised chunks
→ apply token budget
→ sanitise prompt
→ generate answer
→ validate output
→ audit activity
```

---

## Project Structure

```text
NeuralRays-RAG-Chatbot/
│
├── data/
│   ├── chroma_db/                  # Local ChromaDB vector database, ignored by Git
│   └── pages.json                  # Extracted NeuralRays website content
│
├── static/
│   ├── index.html                  # Browser-based chatbot interface
│   └── nr-logo.png                 # NeuralRays logo used in the UI
│
├── eval/
│   ├── golden_questions.jsonl      # Manual evaluation questions
│   ├── run_eval.py                 # Earlier/simple evaluation runner
│   ├── scoring.py                  # Earlier/simple scoring logic
│   └── results/                    # Evaluation outputs
│
├── evaluation/
│   ├── config.py
│   ├── dashboard.py                # HTML dashboard generator
│   ├── dataset_generator.py        # Auto question generation
│   ├── evaluation_service.py       # Main domain-agnostic evaluation service
│   ├── models.py                   # Evaluation data models
│   ├── storage.py
│   ├── results/                    # JSON/HTML evaluation results
│   └── metric_adapters/
│       ├── base.py
│       ├── simple_adapter.py
│       ├── ragas_adapter.py
│       └── deepeval_adapter.py
│
├── pii/
│   ├── audit.py                    # PII audit logging
│   ├── config.py                   # PII pipeline configuration
│   ├── ingestion_guard.py          # Ingestion-time PII masking/tokenisation
│   ├── llm_guard.py                # Prompt sanitisation before LLM calls
│   ├── output_validator.py         # Final output PII validation
│   ├── redactor.py                 # Core PII detection/redaction
│   └── token_store.py              # SQLite token mapping
│
├── connectors/
│   ├── acl.py                      # ACL permission logic
│   ├── mcp_server.py               # MCP-style source server demo
│   ├── mock_source.py              # Mock enterprise source connector
│   ├── models.py                   # Connector data models
│   ├── sample_documents.jsonl      # Mock enterprise documents
│   └── sync.py                     # Sync enterprise docs into ChromaDB
│
├── docs/
│   ├── chunking_retrieval_strategy.md
│   └── production_architecture.md
│
├── tests/
│   ├── test_llm_guard.py
│   ├── test_pii_pipeline.py
│   ├── test_pii_redactor.py
│   └── test_source_connector_acl.py
│
├── app.py                          # FastAPI app and chatbot API endpoint
├── chatbot.py                      # Chatbot retrieval and answer logic
├── chunker.py                      # Semantic/layout-aware chunking
├── crawler.py                      # Website crawler
├── ingest.py                       # Embedding and ChromaDB ingestion pipeline
├── reranker.py                     # Reranker adapter
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AashraySurya/NeuralRays-RAG-Chatbot.git
cd NeuralRays-RAG-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On Mac/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Before running the chatbot, make sure the virtual environment is activated.

---

### Step 1: Crawl the NeuralRays website

```bash
python crawler.py
```

This extracts website content and saves it to:

```text
data/pages.json
```

Expected output:

```text
Saved 6 unique pages to data/pages.json
```

---

### Step 2: Create chunks and build the vector database

```bash
python ingest.py
```

This will:

- Load `data/pages.json`
- Create semantic/layout-aware chunks
- Generate local embeddings
- Store chunks and embeddings in ChromaDB

The vector database is created locally at:

```text
data/chroma_db/
```

This folder is ignored by Git because it is generated locally.

---

### Step 3: Run the chatbot in the terminal

```bash
python chatbot.py
```

Example questions:

```text
What does NeuralRays do?
What AI services does NeuralRays offer?
Does NeuralRays offer cloud transformation?
How can I contact NeuralRays?
Who is the CEO of NeuralRays?
```

To exit:

```text
exit
```

---

### Step 4: Run the chatbot in the browser

```bash
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

API endpoint:

```text
POST /chat
```

Example request:

```json
{
  "question": "What AI services does NeuralRays offer?"
}
```

Example response:

```json
{
  "question": "What AI services does NeuralRays offer?",
  "answer": "NeuralRays offers AI services including data strategy consulting, data science and AI consulting, AI solution development and AI-driven automation.",
  "sources": [
    "https://neuralrays.ai/ai-services"
  ]
}
```

---

## Running the Evaluation Framework

Run the domain-agnostic evaluation service:

```bash
python -m evaluation.evaluation_service --tenant_id neuralrays --adapter simple --max_questions 10
```

Generate the HTML dashboard:

```bash
python -m evaluation.dashboard --tenant_id neuralrays
```

Open the latest dashboard from:

```text
evaluation/results/neuralrays/
```

Latest evaluation metrics include both quality and operational performance, such as:

```text
pass rate
faithfulness
correctness
context precision
answer relevancy
hallucination
source presence
retrieval latency
generation latency
total latency
estimated tokens
estimated cost
overall RAG score
```

---

## Running PII Tests

Run all PII-related tests:

```bash
python -m pytest tests/test_pii_redactor.py tests/test_llm_guard.py tests/test_pii_pipeline.py
```

The PII pipeline supports:

```text
partial_mask
full_mask
tokenise
```

Examples:

```text
Original: rahul@gmail.com
Partial: rah***@gmail.com
Full: ***************
Tokenised: <email_cust_001>
```

The tokenised placeholder can be stored locally in SQLite and remapped later if the user is authorised.

---

## Running the Source Connector / ACL Demo

Run the MCP-style source server demo:

```bash
python -m connectors.mcp_server
```

This demonstrates:

- available MCP-style tools
- authorised engineering document retrieval
- authorised HR document retrieval
- blocked access when an engineering user searches for a restricted HR document

Run the enterprise source sync:

```bash
python -m connectors.sync
```

This syncs mock enterprise documents into a separate ChromaDB collection:

```text
enterprise_source_documents
```

---

## Running Tests

Run the full test suite:

```bash
python -m pytest
```

Run specific test groups:

```bash
python -m pytest tests/test_pii_redactor.py
python -m pytest tests/test_llm_guard.py
python -m pytest tests/test_pii_pipeline.py
python -m pytest tests/test_source_connector_acl.py
```

---

## Documentation

Additional documentation is stored in:

```text
docs/chunking_retrieval_strategy.md
docs/production_architecture.md
```

### Chunking and Retrieval Strategy

This document covers:

- chunking strategies by document type
- tender document handling
- hybrid retrieval
- reranking
- token budgeting
- context compression
- LLM context limit handling

### Production Architecture

This document covers:

- local development workflow
- CI/CD
- prompt versioning
- vector database management
- DataOps
- LLMOps
- monitoring
- observability
- scalability
- rollback strategy

---

## Notes About API Keys

This prototype currently uses local embeddings through `sentence-transformers`, so it does not require an OpenAI API key for the retrieval pipeline.

The current answer generation is local/rule-based and source-grounded.

A future version could integrate an LLM such as:

- OpenAI
- Azure OpenAI
- Gemini
- Claude
- Ollama/local LLM

Any API key should be stored locally in a `.env` file and should not be committed to GitHub.

---

## Git Workflow

Development is carried out using feature branches.

Example branch workflow:

```bash
git checkout main
git pull origin main
git checkout -b feature/example-feature
```

After completing and testing the feature:

```bash
git add .
git commit -m "Meaningful commit message"
git push -u origin feature/example-feature
```

Then merge back into `main`:

```bash
git checkout main
git pull origin main
git merge feature/example-feature
git push origin main
```

---

## Completed Work

### Core RAG

- Set up GitHub repository
- Set up local Python environment
- Added project dependencies
- Built website crawler
- Extracted NeuralRays website content
- Cleaned duplicate pages
- Saved website data to `data/pages.json`
- Built ingestion pipeline
- Added semantic/layout-aware chunking
- Generated local embeddings
- Stored embeddings in ChromaDB
- Built local chatbot retrieval logic
- Improved chatbot answer relevance and formatting
- Added source URLs
- Added handling for team and role-based questions
- Added FastAPI API endpoint
- Added browser-based chatbot interface

### Evaluation

- Built domain-agnostic evaluation framework
- Added auto question generation
- Added local metric adapter
- Added JSON result output
- Added HTML dashboard
- Added quality metrics
- Added latency, token usage, estimated cost and overall RAG score

### PII Protection

- Added PII redaction module
- Added LLM prompt guard
- Added ingestion-time PII guard
- Added configurable masking modes
- Added SQLite token store
- Added output validator
- Added PII audit logging
- Added PII tests

### Source Connector / ACL

- Added mock enterprise source connector
- Added ACL permission model
- Added MCP-style source server
- Added enterprise document sync
- Added ACL tests

### Documentation

- Added chunking and retrieval strategy study
- Added production deployment, DataOps and LLMOps architecture document

---

## Current Limitations

- The chatbot currently uses local embeddings and structured/rule-based answer generation rather than a full external LLM.
- The vector database is local and must be regenerated after cloning the repository.
- The crawler is designed for the NeuralRays website and may need updates if the website structure changes.
- The source connector currently uses mock enterprise documents rather than real Confluence, SharePoint or S3 credentials.
- RAGAS and DeepEval adapters are placeholders for future integration.
- Cost tracking is estimated and currently defaults to zero because no paid external LLM is used.
- The browser interface is intentionally simple for prototype demonstration purposes.

---

## Planned Improvements

- Integrate a selected LLM provider once an API key or preferred model is confirmed.
- Add real Confluence, SharePoint and S3 connectors.
- Add official MCP SDK support.
- Add hybrid keyword + vector search.
- Add parent-child retrieval.
- Add table-aware PDF/tender document parsing.
- Add token budget manager in code.
- Add context compression.
- Add prompt versioning files.
- Add CI/CD pipeline.
- Add production monitoring and logging.
- Add stronger RAGAS/DeepEval integration.

---

## Demo Flow

A suggested demo order:

1. Show the project structure.
2. Show `crawler.py`.
3. Run:

```bash
python crawler.py
```

4. Show `data/pages.json`.
5. Show `chunker.py` and `ingest.py`.
6. Run:

```bash
python ingest.py
```

7. Show ChromaDB generated locally in:

```text
data/chroma_db/
```

8. Run terminal chatbot:

```bash
python chatbot.py
```

9. Ask sample questions.
10. Run browser chatbot:

```bash
uvicorn app:app --reload
```

11. Open:

```text
http://127.0.0.1:8000
```

12. Run evaluation:

```bash
python -m evaluation.evaluation_service --tenant_id neuralrays --adapter simple --max_questions 10
python -m evaluation.dashboard --tenant_id neuralrays
```

13. Show the evaluation dashboard.
14. Run PII tests:

```bash
python -m pytest tests/test_pii_redactor.py tests/test_llm_guard.py tests/test_pii_pipeline.py
```

15. Run source connector demo:

```bash
python -m connectors.mcp_server
```

16. Explain that the engineering user cannot retrieve the restricted HR document.
17. Show the documentation files in `docs/`.

---

## Feedback Areas Completed

The project now addresses the four main feedback areas:

| Feedback Area | Status |
|---|---|
| PII protection across the RAG lifecycle | Completed |
| Generic evaluation framework | Completed |
| Chunking and retrieval strategy | Documented |
| Production deployment, DataOps and LLMOps architecture | Documented |

---

## Summary

This project started as a Week 1 RAG chatbot prototype for the NeuralRays website.

It has now been extended into a broader enterprise RAG prototype with:

- website crawling
- semantic chunking
- local embeddings
- ChromaDB retrieval
- reranking
- source-grounded answers
- evaluation dashboard
- PII protection
- ACL-aware enterprise source connector
- production architecture documentation

The system remains local and prototype-based, but it now demonstrates the key building blocks needed for a production-ready enterprise RAG system.