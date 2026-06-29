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
- Who is the CEO of NeuralRays?
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
- Handling for common website questions such as services, contact details, pricing, success stories, and team roles
- FastAPI `/chat` API endpoint
- Simple browser-based chatbot interface
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
├── data/
│   ├── chroma_db/          # Local ChromaDB vector database, ignored by Git
│   └── pages.json          # Extracted NeuralRays website content
│
├── static/
│   └── index.html          # Simple browser-based chatbot interface
│
├── tests/
│   └── sample_questions.md # Sample questions for testing the chatbot
│
├── venv/                   # Local virtual environment, ignored by Git
│
├── .env.example            # Example environment variable file
├── .gitignore              # Files and folders excluded from Git
├── app.py                  # FastAPI app and chatbot API endpoint
├── chatbot.py              # Chatbot retrieval and answer logic
├── crawler.py              # Website crawler for NeuralRays content
├── ingest.py               # Chunking, embedding, and ChromaDB ingestion pipeline
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
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

Before running the chatbot, make sure the virtual environment is activated.

On Windows:

```bash
venv\Scripts\Activate
```

On Mac/Linux:

```bash
source venv/bin/activate
```

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

```text
Who is the CEO of NeuralRays?
```

To exit the terminal chatbot, type:

```text
exit
```

### Step 4: Run the chatbot in the browser

The project also includes a simple browser-based chatbot interface using FastAPI.

Run:

```bash
uvicorn app:app --reload
```

Then open this URL in your browser:

```text
http://127.0.0.1:8000
```

This will open the chatbot web interface.

You can also access the FastAPI documentation here:

```text
http://127.0.0.1:8000/docs
```

The API endpoint is:

```text
POST /chat
```

Example request body:

```json
{
  "question": "What AI services does NeuralRays offer?"
}
```

Example response:

```json
{
  "question": "What AI services does NeuralRays offer?",
  "answer": "Neural Rays offers several AI services, including Data Strategy Consulting, Data Science and AI Consulting, AI Solution Development, and AI-Driven Automation.",
  "sources": [
    "https://neuralrays.ai/ai-services"
  ]
}
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
10. Who is the CEO of NeuralRays?
11. Does NeuralRays list pricing on the website?

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
- Added handling for team and role-based questions
- Added FastAPI API endpoint
- Added simple browser-based chatbot interface

## Current Limitations

- The chatbot currently uses local embeddings and structured answer formatting rather than a full LLM generation layer.
- The vector database is local and must be regenerated after cloning the repository.
- The crawler is designed specifically for the NeuralRays website and may need updates if the website structure changes.
- The browser interface is intentionally simple for prototype demonstration purposes.

## Planned Improvements

- Integrate a selected LLM provider once an API key or preferred model is confirmed
- Improve fallback handling when information is not found on the website
- Add automated testing using the sample questions
- Improve the browser interface styling and user experience
- Add more detailed evaluation of retrieval accuracy
- Prepare final demonstration flow

## Demo Flow

For demonstration, the project can be shown in this order:

1. Show the website crawler in `crawler.py`
2. Run `python crawler.py`
3. Show the extracted content in `data/pages.json`
4. Run `python ingest.py`
5. Show that ChromaDB is generated locally in `data/chroma_db/`
6. Run the terminal chatbot using `python chatbot.py`
7. Run the browser chatbot using `uvicorn app:app --reload`
8. Open `http://127.0.0.1:8000`
9. Ask sample questions through the browser interface
10. Show that answers include relevant NeuralRays website source URLs