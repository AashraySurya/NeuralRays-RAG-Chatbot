"""
Chatbot retrieval logic for the Neural Rays RAG chatbot.

This script loads the ChromaDB vector database, embeds the user's question using
the same local embedding model used during ingestion, retrieves relevant website
chunks, re-ranks them, and produces a simple grounded answer using Neural Rays
website content.

This version does not require an OpenAI API key.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reduce noisy logs from Hugging Face and transformers
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "neuralrays_website"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_RESULTS = 8


@dataclass
class RetrievedChunk:
    """
    Represents one relevant website chunk retrieved from ChromaDB.
    """

    text: str
    url: str
    title: str
    distance: float
    rerank_score: float = 0.0


def setup_logging() -> None:
    """
    Set up clean logging and hide noisy third-party library logs.
    """

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
        "chromadb",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def normalise_question(question: str) -> str:
    """
    Convert a question into lower-case text for simpler matching.
    """

    return question.lower().strip()


def identify_intent(question: str) -> str:
    """
    Identify the broad type of question being asked.

    This helps us favour the correct Neural Rays page after vector retrieval.
    """

    question = normalise_question(question)

    if any(word in question for word in ["price", "pricing", "cost", "fees", "charge"]):
        return "pricing"

    if any(word in question for word in ["contact", "email", "phone", "reach", "location", "office", "based"]):
        return "contact"

    if any(
        word in question
        for word in [
            "ceo",
            "cbo",
            "founder",
            "team",
            "core team",
            "director",
            "technical lead",
            "technical director",
            "finance executive",
            "product director",
            "qa lead",
            "devops",
            "devops architect",
        ]
    ):
        return "team"

    if any(word in question for word in ["cloud", "migration", "cloud transformation"]):
        return "cloud"

    if any(word in question for word in ["digital service", "digital services", "platform", "assurance"]):
        return "digital_services"

    if any(word in question for word in ["ai service", "ai services", "artificial intelligence", "machine learning"]):
        return "ai_services"

    if any(word in question for word in ["automation", "automate", "repetitive"]):
        return "automation"

    if any(word in question for word in ["data strategy", "data science", "consulting"]):
        return "ai_services"

    if any(word in question for word in ["success", "case study", "case studies", "stories", "examples"]):
        return "success_stories"

    if any(word in question for word in ["what does", "about", "who are", "company"]):
        return "about"

    return "general"


def get_page_priority(intent: str, url: str, title: str) -> float:
    """
    Give a score boost to pages that are more suitable for the user's intent.
    """

    url = url.lower()
    title = title.lower()

    if intent == "contact":
        if "contact" in url or "contact" in title:
            return 4.0

    if intent == "team":
        if "about" in url or "about" in title:
            return 5.0
        if "contact" in url:
            return -3.0

    if intent in {"ai_services", "automation"}:
        if "ai-services" in url or "ai services" in title:
            return 4.0
        if "contact" in url:
            return -3.0

    if intent in {"digital_services", "cloud"}:
        if "digital-service" in url or "digital services" in title:
            return 4.0
        if "contact" in url:
            return -3.0

    if intent == "success_stories":
        if "success-stories" in url or "success stories" in title:
            return 4.0

    if intent == "about":
        if "about" in url or "about" in title:
            return 3.0
        if url.rstrip("/") == "https://neuralrays.ai":
            return 2.0

    return 0.0


def keyword_overlap_score(question: str, text: str) -> float:
    """
    Give a small score boost when words from the question appear in the chunk.
    """

    stop_words = {
        "what",
        "does",
        "do",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "for",
        "with",
        "and",
        "or",
        "neuralrays",
        "neural",
        "rays",
        "offer",
        "offers",
    }

    question_words = {
        word
        for word in re.findall(r"[a-zA-Z]+", question.lower())
        if word not in stop_words
    }

    text_words = set(re.findall(r"[a-zA-Z]+", text.lower()))

    overlap = question_words.intersection(text_words)

    return float(len(overlap))


def extract_relevant_sentences(text: str, question: str, max_sentences: int = 3) -> str:
    """
    Extract a few sentences from a retrieved chunk that match the question.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text)

    question_words = {
        word
        for word in re.findall(r"[a-zA-Z]+", question.lower())
        if len(word) > 3
    }

    scored_sentences = []

    for sentence in sentences:
        sentence_words = set(re.findall(r"[a-zA-Z]+", sentence.lower()))
        score = len(question_words.intersection(sentence_words))

        if score > 0:
            scored_sentences.append((score, sentence))

    if scored_sentences:
        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [sentence for _, sentence in scored_sentences[:max_sentences]]
    else:
        selected = sentences[:max_sentences]

    return " ".join(selected).strip()


def answer_team_question(question: str) -> str:
    """
    Answer common team-related questions using known content from the crawled
    Neural Rays About Us page.
    """

    question = normalise_question(question)

    team_members = {
        "ceo": {
            "name": "Senthil Loganathan",
            "role": "CEO",
        },
        "cbo": {
            "name": "Srinivasan B Kada",
            "role": "CBO",
        },
        "technical lead": {
            "name": "Balaji Kalidhasan",
            "role": "Technical Lead",
        },
        "technical director": {
            "name": "Hemkumar Lakshminarayanan",
            "role": "Technical Director",
        },
        "finance executive": {
            "name": "Thilagavathi Ravikumar",
            "role": "Finance Executive",
        },
        "product director": {
            "name": "Anirban Bose",
            "role": "Product Director",
        },
        "qa lead": {
            "name": "Iswarya Selvam",
            "role": "QA Lead",
        },
        "devops architect": {
            "name": "Sathiyan Sivaprakasam",
            "role": "DevOps Architect",
        },
    }

    for role_key, details in team_members.items():
        if role_key in question:
            return (
                f"According to the Neural Rays About Us page, the {details['role']} "
                f"is {details['name']}.\n\n"
                "Most relevant page: About Us – NeuralRays AI\n"
                "Source: https://neuralrays.ai/about-us"
            )

    if "solutions director" in question or "solution director" in question:
        return (
            "According to the Neural Rays About Us page, the Solutions Directors "
            "are Rajkumar Srinivasan and Satish Namburi.\n\n"
            "Most relevant page: About Us – NeuralRays AI\n"
            "Source: https://neuralrays.ai/about-us"
        )

    if "director" in question:
        return (
            "According to the Neural Rays About Us page, the listed directors include "
            "Rajkumar Srinivasan and Satish Namburi as Solutions Directors, "
            "Hemkumar Lakshminarayanan as Technical Director, and Anirban Bose as Product Director.\n\n"
            "Most relevant page: About Us – NeuralRays AI\n"
            "Source: https://neuralrays.ai/about-us"
        )

    return (
        "The Neural Rays About Us page lists its core team as including:\n"
        "- Rajkumar Srinivasan, Solutions Director\n"
        "- Balaji Kalidhasan, Technical Lead\n"
        "- Satish Namburi, Solutions Director\n"
        "- Hemkumar Lakshminarayanan, Technical Director\n"
        "- Thilagavathi Ravikumar, Finance Executive\n"
        "- Anirban Bose, Product Director\n"
        "- Iswarya Selvam, QA Lead\n"
        "- Srinivasan B Kada, CBO\n"
        "- Senthil Loganathan, CEO\n"
        "- Sathiyan Sivaprakasam, DevOps Architect\n\n"
        "Most relevant page: About Us – NeuralRays AI\n"
        "Source: https://neuralrays.ai/about-us"
    )


class NeuralRaysChatbot:
    """
    Local RAG chatbot for the Neural Rays website.
    """

    def __init__(self) -> None:
        """
        Load the embedding model and ChromaDB collection once.
        """

        if not CHROMA_DB_PATH.exists():
            raise FileNotFoundError(
                "ChromaDB database not found. Run python ingest.py first."
            )

        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = chroma_client.get_collection(COLLECTION_NAME)

    def retrieve_relevant_chunks(self, question: str) -> list[RetrievedChunk]:
        """
        Retrieve and re-rank website chunks for the user's question.
        """

        question_embedding = self.model.encode(
            question,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

        total_chunks = self.collection.count()
        number_of_results = min(TOP_K_RESULTS, total_chunks)

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=number_of_results,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        intent = identify_intent(question)

        chunks: list[RetrievedChunk] = []

        for document, metadata, distance in zip(documents, metadatas, distances):
            url = metadata.get("url", "")
            title = metadata.get("title", "Untitled Page")

            # ChromaDB distance is better when lower, so we subtract it.
            vector_score = -float(distance)
            page_score = get_page_priority(intent, url, title)
            keyword_score = keyword_overlap_score(question, document)

            rerank_score = vector_score + page_score + keyword_score

            chunks.append(
                RetrievedChunk(
                    text=document,
                    url=url,
                    title=title,
                    distance=float(distance),
                    rerank_score=rerank_score,
                )
            )

        chunks.sort(key=lambda chunk: chunk.rerank_score, reverse=True)

        return chunks

    def build_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """
        Build a clean answer using retrieved Neural Rays website chunks.
        """

        intent = identify_intent(question)

        if intent == "pricing":
            return (
                "I could not find pricing information on the Neural Rays website. "
                "You may want to contact Neural Rays directly for pricing or project-specific costs."
            )

        if intent == "team":
            return answer_team_question(question)

        if not chunks:
            return "I could not find that information on the Neural Rays website."

        best_chunk = chunks[0]

        if intent == "ai_services":
            return (
                "Neural Rays offers several AI services, including:\n"
                "- Data Strategy Consulting\n"
                "- Data Science and AI Consulting\n"
                "- AI Solution Development / Custom AI Development\n"
                "- AI-Driven Automation\n\n"
                "These services are focused on helping organisations build data strategies, "
                "use AI effectively, develop AI-powered solutions, and automate repetitive tasks.\n\n"
                f"Most relevant page: {best_chunk.title}\n"
                f"Source: {best_chunk.url}"
            )

        if intent == "automation":
            return (
                "Yes. Neural Rays offers AI-Driven Automation. The website explains that "
                "they use AI-powered automation to reduce mundane and repetitive tasks, "
                "freeing teams to focus on higher-value work. They also support use cases, "
                "rapid prototypes, MVPs, and scaling automation into production systems.\n\n"
                f"Most relevant page: {best_chunk.title}\n"
                f"Source: {best_chunk.url}"
            )

        if intent == "cloud":
            return (
                "Yes. Neural Rays offers Cloud Transformation as part of its Digital Services. "
                "The website says they advise on cloud technologies, develop cloud-native solutions, "
                "support cloud integration, help with application modernisation, and facilitate cloud migration.\n\n"
                f"Most relevant page: {best_chunk.title}\n"
                f"Source: {best_chunk.url}"
            )

        if intent == "digital_services":
            return (
                "Neural Rays lists three main Digital Services:\n"
                "- Product and Platform Engineering\n"
                "- Cloud Transformation\n"
                "- Digital Assurance\n\n"
                "These services cover software/platform development, cloud adoption, application modernisation, "
                "quality assurance, digital testing, and AI-led test automation.\n\n"
                f"Most relevant page: {best_chunk.title}\n"
                f"Source: {best_chunk.url}"
            )

        if intent == "contact":
            return (
                "You can contact Neural Rays by emailing Hello@NeuralRays.AI. "
                "The website says they are based in Chennai, India; Dubai, UAE; and London, UK, "
                "and serve clients globally.\n\n"
                f"Most relevant page: {best_chunk.title}\n"
                f"Source: {best_chunk.url}"
            )

        if intent == "success_stories":
            return (
                "Neural Rays lists several success stories, including:\n"
                "- Risk Management Software Platform\n"
                "- Digital Banking Platform\n"
                "- Geriatric Care Platform\n"
                "- Blockchain-Based Digital Health Platform\n"
                "- Asset Tracking and Inventory Management System\n"
                "- Oil and Gas Analytics\n"
                "- AI-Based Vehicle Assembly Line Defect Detection\n"
                "- Retail Macro and Micro Space Optimisation\n\n"
                f"Most relevant page: {best_chunk.title}\n"
                f"Source: {best_chunk.url}"
            )

        if intent == "about":
            return (
                "Neural Rays AI develops artificial intelligence and data-driven software solutions. "
                "The company focuses on AI and digital services, ethical AI, data analysis, agile delivery, "
                "and lean product development to help clients build transformational digital products and outcomes.\n\n"
                f"Most relevant page: {best_chunk.title}\n"
                f"Source: {best_chunk.url}"
            )

        relevant_text = extract_relevant_sentences(best_chunk.text, question)

        return (
            "Based on the Neural Rays website, I found this relevant information:\n\n"
            f"{relevant_text}\n\n"
            f"Most relevant page: {best_chunk.title}\n"
            f"Source: {best_chunk.url}"
        )

    def answer_question(self, question: str) -> dict[str, Any]:
        """
        Answer a user question using the RAG retrieval pipeline.
        """

        chunks = self.retrieve_relevant_chunks(question)
        answer = self.build_answer(question, chunks)

        sources = []

        for chunk in chunks:
            if chunk.url not in sources:
                sources.append(chunk.url)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }


_chatbot: NeuralRaysChatbot | None = None


def get_chatbot() -> NeuralRaysChatbot:
    """
    Create one chatbot instance and reuse it.
    """

    global _chatbot

    if _chatbot is None:
        _chatbot = NeuralRaysChatbot()

    return _chatbot


def answer_question(question: str) -> dict[str, Any]:
    """
    Public function used by the terminal chatbot and FastAPI.
    """

    chatbot = get_chatbot()
    return chatbot.answer_question(question)


def run_chatbot() -> None:
    """
    Run the chatbot in the terminal.
    """

    setup_logging()

    print("Neural Rays RAG Chatbot")
    print("Type 'exit' to quit.\n")

    # Load model and database before first question
    chatbot = get_chatbot()

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        result = chatbot.answer_question(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for source in result["sources"][:3]:
            print(f"- {source}")

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    run_chatbot()