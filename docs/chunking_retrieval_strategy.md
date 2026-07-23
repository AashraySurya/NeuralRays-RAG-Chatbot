# Chunking and Retrieval Strategy Study

## 1. Purpose

This document summarises the chunking and retrieval strategy for the RAG chatbot project.

It addresses three key points:

- Which chunking strategies are suitable for different document types.
- How to handle large, context-rich documents such as power or railway tender documents.
- How to manage cases where retrieved content exceeds the LLM context limit.

---

## 2. Current Prototype Approach

The current prototype uses semantic/layout-aware chunking.

Current flow:

```text
Source content
→ clean text
→ detect headings/sections
→ create metadata-rich chunks
→ embed chunks
→ store in ChromaDB
→ retrieve candidates
→ rerank candidates
→ pass final context to answer generation
```

Each chunk stores metadata such as:

```text
tenant_id
source_url
title
section_heading
chunk_index
chunk_type
word_count
```

This is more useful than basic fixed-size chunking because it keeps the chunk closer to the original document structure.

---

## 3. Chunking Strategies Compared

| Strategy | Best For | Limitation |
|---|---|---|
| Fixed-size chunking | Simple articles, basic web pages | Can split sections or clauses incorrectly |
| Sliding window chunking | Long text where nearby context matters | Creates duplicate content |
| Heading-based chunking | Policies, tenders, reports, manuals | Depends on clean headings |
| Semantic chunking | Website content, mixed-format documents | More complex to tune |
| Parent-child chunking | Long contracts, technical documents, tenders | Needs more retrieval logic |
| Table-aware chunking | Tender tables, pricing sheets, compliance matrices | Requires table extraction |
| Summary-first retrieval | Very large documents and reports | May lose detail |
| Hybrid chunking | Enterprise RAG and tender documents | More complex but most flexible |

---

## 4. Recommended Strategy by Document Type

| Document Type | Recommended Strategy |
|---|---|
| Website pages | Semantic/layout-aware chunking |
| FAQs | Question-answer pair chunking |
| Policies | Heading-based chunking with overlap |
| Contracts | Heading-based + parent-child chunking |
| Tender documents | Hybrid chunking |
| Technical manuals | Heading-based + semantic chunking |
| Compliance matrices | Table-aware chunking |
| Long reports | Summary-first + parent-child retrieval |

---

## 5. Recommended Strategy for Power/Railway Tender Documents

Large tender documents usually include:

- Scope of work
- Eligibility criteria
- Technical requirements
- Commercial requirements
- Timelines
- Evaluation criteria
- Pricing schedules
- Compliance matrices
- Appendices
- Legal clauses

For these documents, one chunking method is not enough.

Recommended approach:

```text
Document parsing
→ heading detection
→ table extraction
→ parent section chunks
→ smaller child chunks
→ metadata tagging
→ embedding
→ vector storage
```

Useful metadata:

```text
document_id
document_title
section_heading
page_number
chunk_type
requirement_id
table_name
tenant_id
access_control_metadata
```

---

## 6. Retrieval Strategy

The retrieval process should not simply return the first few vector results.

Recommended flow:

```text
User query
→ retrieve broad candidate pool
→ apply ACL filtering
→ rerank candidates
→ deduplicate similar chunks
→ group by document/section
→ apply token budget
→ build final context
→ generate answer with citations
```

For production, hybrid search is recommended:

```text
Vector search + keyword search + metadata filtering + reranking
```

This helps with both natural language questions and exact references such as clause numbers, dates, requirement IDs or tender codes.

---

## 7. Handling LLM Context Limits

Large documents can return more relevant content than the LLM can accept.

The system should use a token budget.

Recommended flow:

```text
Retrieve top 50 candidate chunks
→ apply ACL filtering
→ rerank
→ remove duplicates
→ keep highest-value chunks
→ compress or summarise lower-priority chunks
→ build final prompt within token limit
```

The prompt should reserve space for:

- System instructions
- User question
- Retrieved context
- Answer generation
- Citations/sources

If too much content is retrieved, the system should:

1. Keep exact matches first.
2. Keep highest reranked chunks.
3. Remove duplicates.
4. Prioritise the most relevant section.
5. Compress lower-priority context.
6. Ask a clarification question if the query is too broad.

---

## 8. Context Compression

Context compression can reduce the amount of text sent to the LLM.

Recommended methods:

| Method | Use Case |
|---|---|
| Extractive compression | Exact clauses, tender requirements, policy wording |
| Abstractive compression | High-level summaries |
| Section-level compression | Long reports or tender sections |

For tenders and contracts, extractive compression is safer because exact wording may matter.

---

## 9. Access Control During Retrieval

Access control must happen before content is passed to the LLM.

Recommended flow:

```text
Retrieve candidate chunks
→ check tenant ID
→ check allowed users/groups
→ remove unauthorised chunks
→ rerank authorised chunks
→ pass only authorised context to the LLM
```

The prototype already demonstrates this through the ACL-aware source connector and MCP-style server.

---

## 10. Evaluation Plan

Each chunking strategy should be tested using the evaluation framework.

Compare strategies using:

- Context precision
- Faithfulness
- Answer relevancy
- Hallucination rate
- Latency
- Token usage
- Estimated cost
- Overall RAG score

This will help identify which strategy works best for each document type.

---

## 11. Recommendation

For the current NeuralRays website chatbot:

```text
Semantic/layout-aware chunking
+ metadata
+ reranking
```

For large tender documents:

```text
Heading-based parent-child chunking
+ table-aware extraction
+ hybrid search
+ reranking
+ token budgeting
+ context compression
```

For production enterprise RAG, the system should support configurable chunking by document type rather than using one fixed strategy for all content.

---

## 12. Conclusion

The current semantic/layout-aware chunking is suitable for website content and basic enterprise knowledge retrieval.

For large and complex documents such as power or railway tenders, the recommended approach is a hybrid strategy combining heading-based chunking, table-aware extraction, parent-child retrieval, reranking, ACL filtering, token budgeting and context compression.