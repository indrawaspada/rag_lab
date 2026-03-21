"""Enhanced Web Scraping RAG Pipeline

A Retrieval-Augmented Generation chatbot that scrapes website content
and answers questions based on the extracted information.
"""

import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import numpy as np

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain   

# ============================================================================
# Configuration
# ============================================================================
load_dotenv()

CHUNK_SIZE = 500  # Good balance for RAG - captures ~1-2 sentences
CHUNK_OVERLAP = 100  # 20% overlap for context continuity
MAX_TOKENS = 15000  # Max tokens for LLM response
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.4  # Lower for more deterministic responses

# Validate OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in .env file.")
    exit(1)

# ============================================================================
# Helper Functions
# ============================================================================


def fetch_html(url: str) -> str | None:
    """Fetch HTML content from a URL with proper encoding handling."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None


def process_website(url: str) -> list[Document]:
    """Process website content: fetch, clean, and split into chunks."""
    html_content = fetch_html(url)
    if not html_content:
        raise ValueError("No content could be fetched from the website.")

    # Parse and clean HTML
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()

    # Extract main content
    main_content = soup.find("main") or soup.find("article")
    text_content = (
        main_content.get_text(separator="\n", strip=True)
        if main_content
        else soup.get_text(separator="\n", strip=True)
    )

    # Clean whitespace
    text_content = "\n".join(
        line.strip() for line in text_content.split("\n") if line.strip()
    )

    print(f"Total content size: {len(text_content)} characters")

    # Create documents
    documents = [Document(page_content=text_content, metadata={"source": url})]
    print(f"\nNumber of documents loaded: {len(documents)}")
    print(f"Sample: {text_content[:200]}...")

    # Split into chunks with fallback strategies
    separators = ["\n\n", "\n", " "]
    texts = []

    for separator in separators:
        text_splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separator=separator,
        )
        texts = text_splitter.split_documents(documents)
        if len(texts) > 1:
            break

    print(f"\nNumber of chunks: {len(texts)}")
    for i, chunk in enumerate(texts, 1):
        print(f"  Chunk {i}: {len(chunk.page_content)} characters")

    return texts


def print_sample_embeddings(
    texts: list[Document], embeddings: OpenAIEmbeddings
) -> None:
    """Display sample embedding information."""
    if not texts:
        print("No texts available for embedding sample.")
        return

    sample_text = texts[0].page_content
    sample_embedding = embeddings.embed_query(sample_text)
    print("\nSample Embedding (first 10 dimensions):")
    print(np.array(sample_embedding[:10]))
    print(f"Embedding shape: {np.array(sample_embedding).shape}")


# ============================================================================
# LLM and Prompt Setup
# ============================================================================
llm = ChatOpenAI(
    model_name=MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
)

QA_TEMPLATE = """Context: {context}

Question: {question}

Answer the question concisely based on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

For generic questions, provide a general answer.
"""

PROMPT = PromptTemplate(
    template=QA_TEMPLATE, input_variables=["context", "question"]
)


def rag_pipeline(query: str, llm: ChatOpenAI, vectorstore: FAISS) -> str:
    """Run RAG pipeline: search relevant docs and generate answer."""
    relevant_docs = vectorstore.similarity_search_with_score(query, k=3)

    print("\nTop 3 most relevant chunks:")
    context = ""
    for i, (doc, score) in enumerate(relevant_docs, 1):
        print(f"{i}. Relevance Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:200]}...\n")
        context += doc.page_content + "\n\n"

    # Format prompt and generate answer
    full_prompt = PROMPT.format(context=context, question=query)
    print("Full Prompt sent to the model:")
    print(full_prompt)
    print("\n" + "=" * 50 + "\n")

    response = llm.invoke(full_prompt)
    return response.content


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Welcome to the Enhanced Web Scraping RAG Pipeline.\n")

    while True:
        url = input("Enter website URL (or 'quit' to exit): ").strip()
        if url.lower() == "quit":
            print("Goodbye!")
            break

        try:
            print("\nProcessing website content...")
            texts = process_website(url)

            if not texts:
                print("No content found. Please try a different URL.\n")
                continue

            print("Creating embeddings and vector store...")
            embeddings = OpenAIEmbeddings()
            print_sample_embeddings(texts, embeddings)

            vectorstore = FAISS.from_documents(texts, embeddings)
            print("\nRAG Pipeline ready! Enter 'new' for another site or 'quit' to exit.")

            while True:
                query = input("\nYour query: ").strip()
                if query.lower() == "quit":
                    print("Goodbye!")
                    exit()
                if query.lower() == "new":
                    break
                if not query:
                    continue

                result = rag_pipeline(query, llm, vectorstore)
                print(f"\nRAG Response: {result}")

        except Exception as e:
            print(f"Error: {e}")
            print("Please try a different URL.\n")
