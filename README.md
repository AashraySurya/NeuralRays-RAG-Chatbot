# NeuralRays RAG Chatbot

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot prototype for the NeuralRays AI website.

The chatbot is designed to help users navigate the NeuralRays website by answering questions using information extracted from the site. It crawls the website, processes the content, creates embeddings, stores them in a local vector database, and retrieves relevant context when a user asks a question.

## Objective

The objective of this Week 1 prototype is to build a basic RAG chatbot that can answer user queries using information from the NeuralRays website.

The chatbot should be able to answer questions such as:

- What does NeuralRays do?
- What AI services does NeuralRays offer?
- Does NeuralRays offer cloud transformation?
- How can I contact NeuralRays?
- Does NeuralRays list pricing?

## Current Features

- Python-based website crawler
- Website text extraction and cleaning
- Duplicate page handling
- Structured website content saved to JSON
- Text chunking for RAG processing
- Local embedding generation using `sentence-transformers`
- ChromaDB vector database storage
- Local retrieval-based chatbot logic
- Basic answer generation using retrieved NeuralRays website content
- Source URLs returned with chatbot answers
- GitHub feature branch workflow

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

## RAG Pipeline

The project follows this RAG workflow:

1. Crawl the NeuralRays website
2. Extract and clean website text
3. Save extracted content to `data/pages.json`
4. Split website content into smaller chunks
5. Generate embeddings for each chunk
6. Store embeddings and metadata in ChromaDB
7. Retrieve relevant chunks for a user query
8. Generate a grounded chatbot response using retrieved website content

## Project Structure

```text
NeuralRays-RAG-Chatbot/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── crawler.py
├── ingest.py
├── chatbot.py
│
├── data/
│   ├── pages.json
│   └── chroma_db/          # Generated locally and ignored by Git
│
└── tests/
    └── sample_questions.md
```

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

On Windows:

```bash
venv\Scripts\Activate
```

On Mac/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Step 1: Crawl the NeuralRays website

Run:

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

### Step 2: Create embeddings and build the vector database

Run:

```bash
python ingest.py
```

This will:

- Load `data/pages.json`
- Split the website content into chunks
- Generate local embeddings
- Store the chunks and embeddings in ChromaDB

The vector database is created locally at:

```text
data/chroma_db/
```

This folder is ignored by Git because it is generated locally.

### Step 3: Run the chatbot in the terminal

Run:

```bash
python chatbot.py
```

You can then ask questions such as:

```text
What does NeuralRays do?
```

```text
What AI services does NeuralRays offer?
```

```text
Does NeuralRays offer cloud transformation?
```

```text
How can I contact NeuralRays?
```

To exit the chatbot, type:

```text
exit
```

## Example Questions

Example test questions are stored in:

```text
tests/sample_questions.md
```

Sample questions include:

1. What does NeuralRays do?
2. What AI services does NeuralRays offer?
3. Does NeuralRays provide data science consulting?
4. What is AI-driven automation?
5. Does NeuralRays help with cloud transformation?
6. What digital services are listed on the website?
7. Are there any success stories or case studies?
8. How can I contact NeuralRays?
9. Where is NeuralRays located?
10. Does NeuralRays list pricing on the website?

## Notes About API Keys

This prototype currently uses local embeddings through `sentence-transformers`, so it does not require an OpenAI API key for the retrieval pipeline.

A future version could integrate an LLM such as OpenAI, Azure OpenAI, Gemini, Claude, or a local Ollama model to improve answer generation.

Any API key should be stored locally in a `.env` file and should not be committed to GitHub.

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

## Completed Work So Far

- Set up GitHub repository
- Set up local Python environment
- Added project dependencies
- Built website crawler
- Extracted NeuralRays website content
- Cleaned duplicate pages
- Saved website data to `data/pages.json`
- Built ingestion pipeline
- Generated local embeddings
- Stored embeddings in ChromaDB
- Built local chatbot retrieval logic
- Improved chatbot answer relevance and formatting

## Current Limitations

- The chatbot currently uses rule-based/simple answer formatting rather than a full LLM generation layer.
- The frontend/API layer is not yet complete.
- The vector database is local and must be regenerated after cloning the repository.
- The crawler is designed specifically for the NeuralRays website and may need updates if the website structure changes.

## Planned Improvements

- Add a FastAPI `/chat` endpoint
- Add a simple chatbot interface
- Improve answer generation with an LLM
- Add stronger fallback handling when information is not found
- Add automated testing using the sample questions
- Improve README with demo instructions
- Prepare final demonstration flow

## Demo Flow

For demonstration, the project can be shown in this order:

1. Show the website crawler
2. Show `data/pages.json`
3. Run the ingestion pipeline
4. Show ChromaDB vector database generation
5. Run the chatbot in the terminal
6. Ask sample questions
7. Show retrieved answers and source URLs