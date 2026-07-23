# Production Deployment, DataOps and LLMOps Architecture

## 1. Purpose

This document summarises the target production architecture for the RAG chatbot project.

It covers:

- Local development workflow
- CI/CD
- Prompt versioning
- Vector database management
- DataOps
- LLMOps
- Monitoring and observability
- Scalability
- Rollback strategy

The current project is a local prototype. This document explains how it can move towards a production-ready enterprise RAG system.

---

## 2. Current Prototype

The current prototype includes:

```text
Crawler
→ chunker
→ embeddings
→ ChromaDB
→ retrieval
→ reranker
→ answer generation
→ evaluation framework
→ PII protection
→ ACL-aware source connector
```

Implemented areas:

- Website crawling
- Semantic/layout-aware chunking
- Local embeddings
- ChromaDB vector storage
- Chatbot retrieval
- Reranking
- Source-grounded answers
- Evaluation dashboard
- PII detection, masking and tokenisation
- Prompt sanitisation
- Output validation
- ACL-aware source connector
- MCP-style source server
- Local tests

---

## 3. Target Production Architecture

Recommended production flow:

```text
User
→ Web/API interface
→ Authentication
→ Query router
→ PII prompt guard
→ Retriever
→ ACL filter
→ Reranker
→ Token budget manager
→ Prompt builder
→ LLM provider
→ Output validator
→ Response with citations
→ Audit logging and monitoring
```

Document ingestion flow:

```text
Enterprise sources
→ source connectors
→ document parser
→ PII ingestion guard
→ chunking strategy
→ embeddings
→ vector database
→ metadata and ACL storage
→ evaluation dataset update
```

---

## 4. Local Development Workflow

Typical setup:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Common commands:

```powershell
python crawler.py
python chunker.py
python ingest.py
python chatbot.py
python app.py
```

Evaluation:

```powershell
python -m evaluation.evaluation_service --tenant_id neuralrays --adapter simple --max_questions 10
python -m evaluation.dashboard --tenant_id neuralrays
```

Tests:

```powershell
python -m pytest
```

Source connector:

```powershell
python -m connectors.mcp_server
python -m connectors.sync
```

---

## 5. Git Workflow

Use feature branches and keep `main` stable.

Recommended flow:

```text
main
→ feature branch
→ code changes
→ tests
→ commit
→ push
→ merge to main
```

Example:

```powershell
git checkout main
git pull origin main
git checkout -b feature/new-task

python -m pytest

git add .
git commit -m "Describe the change"
git push -u origin feature/new-task

git checkout main
git pull origin main
git merge feature/new-task
git push origin main
```

---

## 6. CI/CD Pipeline

Recommended CI/CD flow:

```text
Push code
→ install dependencies
→ run tests
→ run PII tests
→ run ACL tests
→ run evaluation
→ build package
→ deploy to staging
→ smoke test
→ deploy to production
```

Deployment should be blocked if:

- Tests fail
- PII checks fail
- ACL checks fail
- Evaluation score drops
- Source presence drops
- Hallucination increases
- Required secrets are missing

---

## 7. Secrets and Environment Variables

Secrets must not be committed to Git.

Examples:

```text
OPENAI_API_KEY
AZURE_OPENAI_API_KEY
DATABASE_URL
VECTOR_DB_API_KEY
CONFLUENCE_TOKEN
SHAREPOINT_CLIENT_SECRET
S3_ACCESS_KEY
S3_SECRET_KEY
```

Use `.env` locally and a secret manager in production.

Recommended `.gitignore` entries:

```text
.env
pii/audit_log.jsonl
pii/pii_tokens.sqlite3
data/chroma_db/
__pycache__/
*.pyc
```

Example environment variables:

```text
APP_ENV=development
TENANT_ID=neuralrays
VECTOR_DB_PATH=data/chroma_db
DEFAULT_LLM_PROVIDER=local
ALLOW_RAW_PII_TO_EXTERNAL_LLM=false
PII_MASKING_MODE=partial_mask
```

---

## 8. Prompt Versioning

Prompts should be versioned and tested before deployment.

Recommended folder:

```text
prompts/
├── answer_prompt_v1.md
├── answer_prompt_v2.md
├── pii_safety_prompt_v1.md
└── summarisation_prompt_v1.md
```

Each prompt should record:

```text
prompt_name
prompt_version
owner
purpose
change_summary
created_date
```

Prompt release flow:

```text
Create new prompt
→ run evaluation
→ compare with old prompt
→ deploy to staging
→ review outputs
→ deploy to production
```

---

## 9. Vector Database Management

The prototype currently uses local ChromaDB.

Current collections:

```text
neuralrays_website
enterprise_source_documents
```

Production vector storage should support:

- Tenant separation
- Metadata filtering
- ACL metadata
- Backups
- Reindexing
- Versioning
- Rollback
- Monitoring

Recommended metadata:

```text
tenant_id
document_id
chunk_id
source_url
source_type
title
section_heading
page_number
chunk_type
embedding_model
allowed_users
allowed_groups
classification
```

---

## 10. Vector Index Versioning

Vector indexes should be versioned.

Example:

```text
neuralrays_website_v1
neuralrays_website_v2
enterprise_source_documents_v1
enterprise_source_documents_v2
```

Recommended deployment pattern:

```text
Build new index
→ run evaluation
→ compare with current index
→ switch alias to new index
→ monitor
→ rollback alias if needed
```

This allows safe rollback without rebuilding the old index.

---

## 11. DataOps Workflow

DataOps manages how source documents move into the RAG system.

Recommended flow:

```text
Source connector
→ document fetch
→ validation
→ parsing
→ PII detection
→ redaction/tokenisation
→ chunking
→ embedding
→ vector storage
→ audit log
```

Each ingestion run should track:

```text
run_id
tenant_id
source_name
documents_fetched
documents_added
documents_updated
chunks_created
pii_items_detected
embedding_model
index_version
status
```

---

## 12. LLMOps Workflow

LLMOps manages model behaviour, prompts, cost, latency and quality.

Track:

```text
model_provider
model_name
model_version
temperature
prompt_version
retrieval_strategy_version
embedding_model_version
reranker_version
```

For each response, log:

```text
request_id
tenant_id
user_id
retrieved_chunk_ids
source_urls
prompt_version
latency_ms
estimated_tokens
estimated_cost
output_validation_status
```

Sensitive logs should be redacted.

---

## 13. PII Protection

Production PII protection should cover:

```text
Document ingestion
→ PII detection
→ redaction/tokenisation
→ safe vector storage
→ prompt sanitisation
→ output validation
→ audit logging
```

Supported modes:

```text
partial_mask
full_mask
tokenise
```

Example:

```text
Original: rahul@gmail.com
Partial: rah***@gmail.com
Full: ***************
Tokenised: <email_cust_001>
```

Raw PII should not be sent to external LLMs unless explicitly permitted.

---

## 14. Retrieval-Time Access Control

Access control must happen before prompt construction.

Recommended flow:

```text
User asks question
→ identify tenant/user/groups
→ retrieve candidate chunks
→ filter by tenant
→ filter by allowed users/groups
→ rerank authorised chunks
→ build prompt only from authorised content
```

Default rule:

```text
Deny access unless user, group or public access is explicitly allowed.
```

The prototype demonstrates this through the ACL-aware source connector.

---

## 15. Monitoring and Observability

Production monitoring should track:

- Request count
- Error count
- Retrieval latency
- Generation latency
- Total latency
- Token usage
- Estimated cost
- Evaluation scores
- Hallucination rate
- Source presence
- PII detections
- Output validation failures
- ACL denials
- Connector failures

Useful dashboards:

- System health
- RAG quality
- Cost
- Latency
- PII/security
- Data ingestion

---

## 16. Evaluation Gates

Before deployment, run evaluation checks for:

```text
faithfulness
correctness
context precision
answer relevancy
hallucination
source presence
latency
token usage
estimated cost
overall RAG score
```

Deployment should be blocked if:

- PII tests fail
- ACL tests fail
- Source presence drops
- Hallucination increases too much
- Overall RAG score drops significantly
- Latency exceeds threshold

---

## 17. Scalability

To scale the system:

- Separate API service from ingestion workers.
- Use background jobs for ingestion.
- Use a managed vector database in production.
- Use caching for frequent queries.
- Use async processing where useful.
- Use queue-based document ingestion.
- Scale API servers horizontally.
- Maintain strong tenant isolation.

Possible production services:

```text
API service
Ingestion worker
Evaluation worker
Scheduler
Vector database
Metadata database
Object storage
Monitoring service
Secret manager
```

---

## 18. Rollback Strategy

Rollback should cover:

- Code
- Prompts
- Vector indexes
- Model configuration
- Connectors

Examples:

```text
Code rollback: deploy previous application version
Prompt rollback: switch answer_prompt_v2 back to answer_prompt_v1
Index rollback: switch alias from index_v2 back to index_v1
Connector rollback: pause sync and restore previous index
```

---

## 19. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Raw PII stored in vector DB | Use ingestion-time masking/tokenisation |
| Raw PII sent to LLM | Use prompt sanitisation |
| Unauthorised retrieval | Use ACL filtering |
| Hallucinated answers | Use citations and evaluation gates |
| High latency | Use reranking, caching and token budgeting |
| High cost | Track tokens and compress context |
| Bad ingestion | Use versioned indexes and rollback |
| Prompt regression | Use prompt versioning and evaluation |
| Cross-tenant leakage | Use tenant isolation and metadata filtering |

---

## 20. Current Prototype vs Production Target

| Area | Current Prototype | Production Target |
|---|---|---|
| Deployment | Local | Cloud or managed environment |
| Vector DB | Local ChromaDB | Managed/vector DB with backups |
| Sources | Website + mock enterprise source | Real Confluence, SharePoint, S3, DBs |
| PII | Detection, masking, tokenisation | Full lifecycle policy-driven handling |
| ACL | Mock ACL-aware retrieval | Real enterprise permissions |
| Evaluation | Local dashboard | CI/CD evaluation gates |
| Monitoring | Local output | Central logs and dashboards |
| Prompt management | Code-based | Versioned prompts |
| Rollback | Git-based | Code, prompt, model and index rollback |

---

## 21. Conclusion

The current project is a strong local RAG prototype.

To move towards production, it should add:

```text
CI/CD
prompt versioning
vector index versioning
secure secret management
monitoring
DataOps workflows
LLMOps workflows
scalability controls
rollback strategy
```

The recommended production design is a secure, multi-tenant RAG architecture where PII is handled before vector storage, access control is enforced during retrieval, prompts are versioned, outputs are validated and deployments are checked through automated evaluation gates.