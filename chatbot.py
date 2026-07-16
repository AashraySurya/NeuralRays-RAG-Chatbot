"""
Chatbot retrieval logic for the Neural Rays RAG chatbot.

This script loads the ChromaDB vector database, embeds the user's question using
the same local embedding model used during ingestion, retrieves relevant website
chunks, sends them through reranker.py, and produces a simple grounded answer
using Neural Rays website content.

This version does not require an OpenAI API key.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reduce noisy logs from Hugging Face and transformers
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import chromadb
from sentence_transformers import SentenceTransformer

from reranker import RerankableChunk, rerank_chunks


CHROMA_DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "neuralrays_website"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieve more chunks first, then let reranker.py reduce them to the best few.
INITIAL_RETRIEVAL_K = 20
FINAL_TOP_K = 3


@dataclass
class RetrievedChunk:
    """
    Represents one relevant website chunk retrieved from ChromaDB.
    """

    text: str
    url: str
    title: str
    distance: float
    section_heading: str = ""
    chunk_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
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

    This helps the reranker favour the correct Neural Rays page after vector retrieval.
    """

    question = normalise_question(question)

    if any(word in question for word in ["price", "pricing", "cost", "fees", "charge"]):
        return "pricing"

    if any(word in question for word in ["contact", "email", "phone", "reach", "location", "office", "based"]):
        return "contact"

    if any(
        phrase in question
        for phrase in [
            "ceo",
            "cbo",
            "founder",
            "team",
            "core team",
            "director",
            "solutions director",
            "solution director",
            "technical lead",
            "technical director",
            "finance executive",
            "product director",
            "qa lead",
            "devops",
            "devops architect",
            "who is",
            "whoo is",
            "who are",

            # Team member names
            "rajkumar",
            "rajkumar srinivasan",
            "balaji",
            "balaji kalidhasan",
            "satish",
            "satish namburi",
            "hemkumar",
            "hemkumar lakshminarayanan",
            "thilagavathi",
            "thilagavathi ravikumar",
            "anirban",
            "anirban bose",
            "iswarya",
            "iswarya selvam",
            "srinivasan b kada",
            "senthil",
            "senthil loganathan",
            "sathiyan",
            "sathiyan sivaprakasam",
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

    if any(word in question for word in ["what does", "about", "who are", "company", "organisation", "organization"]):
        return "about"

    return "general"


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
        selected = [sentence for _score, sentence in scored_sentences[:max_sentences]]
    else:
        selected = sentences[:max_sentences]

    return " ".join(selected).strip()


def answer_team_question(question: str) -> str:
    """
    Answer common team-related questions using known content from the crawled
    Neural Rays About Us page.
    """

    question = normalise_question(question)

    people = [
        {
            "name": "Rajkumar Srinivasan",
            "role": "Solutions Director",
            "matches": ["rajkumar", "rajkumar srinivasan"],
        },
        {
            "name": "Balaji Kalidhasan",
            "role": "Technical Lead",
            "matches": ["balaji", "balaji kalidhasan"],
        },
        {
            "name": "Satish Namburi",
            "role": "Solutions Director",
            "matches": ["satish", "satish namburi"],
        },
        {
            "name": "Hemkumar Lakshminarayanan",
            "role": "Technical Director",
            "matches": ["hemkumar", "hemkumar lakshminarayanan"],
        },
        {
            "name": "Thilagavathi Ravikumar",
            "role": "Finance Executive",
            "matches": ["thilagavathi", "thilagavathi ravikumar", "ravikumar"],
        },
        {
            "name": "Anirban Bose",
            "role": "Product Director",
            "matches": ["anirban", "anirban bose"],
        },
        {
            "name": "Iswarya Selvam",
            "role": "QA Lead",
            "matches": ["iswarya", "iswarya selvam"],
        },
        {
            "name": "Srinivasan B Kada",
            "role": "CBO",
            "matches": ["srinivasan b kada", "kada"],
        },
        {
            "name": "Senthil Loganathan",
            "role": "CEO",
            "matches": ["senthil", "senthil loganathan"],
        },
        {
            "name": "Sathiyan Sivaprakasam",
            "role": "DevOps Architect",
            "matches": ["sathiyan", "sathiyan sivaprakasam"],
        },
    ]

    role_answers = {
        "ceo": ("CEO", "Senthil Loganathan"),
        "cbo": ("CBO", "Srinivasan B Kada"),
        "technical lead": ("Technical Lead", "Balaji Kalidhasan"),
        "technical director": ("Technical Director", "Hemkumar Lakshminarayanan"),
        "finance executive": ("Finance Executive", "Thilagavathi Ravikumar"),
        "product director": ("Product Director", "Anirban Bose"),
        "qa lead": ("QA Lead", "Iswarya Selvam"),
        "devops architect": ("DevOps Architect", "Sathiyan Sivaprakasam"),
    }

    # First, handle questions about a specific person.
    for person in people:
        if any(match in question for match in person["matches"]):
            return (
                f"According to the Neural Rays About Us page, {person['name']} "
                f"is the {person['role']}.\n\n"
                "Most relevant page: About Us – NeuralRays AI\n"
                "Source: https://neuralrays.ai/about-us"
            )

    # Then, handle questions about a specific role.
    for role_key, (role_title, name) in role_answers.items():
        if role_key in question:
            return (
                f"According to the Neural Rays About Us page, the {role_title} "
                f"is {name}.\n\n"
                "Most relevant page: About Us – NeuralRays AI\n"
                "Source: https://neuralrays.ai/about-us"
            )

    # Handle Solutions Director separately because there are two.
    if "solutions director" in question or "solution director" in question:
        return (
            "According to the Neural Rays About Us page, the Solutions Directors "
            "are Rajkumar Srinivasan and Satish Namburi.\n\n"
            "Most relevant page: About Us – NeuralRays AI\n"
            "Source: https://neuralrays.ai/about-us"
        )

    # Only list everyone if the user asks a broad team question.
    if any(phrase in question for phrase in ["team", "core team", "who works", "people", "members"]):
        return (
            "The Neural Rays About Us page lists its core team as:\n"
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

    return (
        "I could not identify that specific person or role from the Neural Rays About Us page.\n\n"
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
        Retrieve website chunks using vector search, then rerank them with reranker.py.
        """

        question_embedding = self.model.encode(
            question,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

        total_chunks = self.collection.count()

        if total_chunks == 0:
            return []

        number_of_results = min(INITIAL_RETRIEVAL_K, total_chunks)

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=number_of_results,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        raw_chunks: list[RerankableChunk] = []

        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}

            url = (
                metadata.get("source_url")
                or metadata.get("url")
                or ""
            )

            title = metadata.get("title", "Untitled Page")
            section_heading = metadata.get("section_heading", "")
            chunk_type = metadata.get("chunk_type", "")

            raw_chunks.append(
                RerankableChunk(
                    text=document,
                    url=url,
                    title=title,
                    section_heading=section_heading,
                    chunk_type=chunk_type,
                    metadata=metadata,
                    distance=float(distance),
                )
            )

        intent = identify_intent(question)

        reranked_chunks = rerank_chunks(
            question=question,
            intent=intent,
            chunks=raw_chunks,
            top_n=FINAL_TOP_K,
        )

        return [
            RetrievedChunk(
                text=chunk.text,
                url=chunk.url,
                title=chunk.title,
                section_heading=chunk.section_heading,
                chunk_type=chunk.chunk_type,
                metadata=chunk.metadata or {},
                distance=chunk.distance,
                rerank_score=chunk.rerank_score,
            )
            for chunk in reranked_chunks
        ]

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
            if chunk.url and chunk.url not in sources:
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