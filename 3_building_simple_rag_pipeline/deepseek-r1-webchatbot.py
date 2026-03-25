import streamlit as st
import os
import requests
from requests.exceptions import RequestException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import tempfile
from langchain_community.document_loaders import BSHTMLLoader

# Configuration variables
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MAX_EMBED_CHARS = 1000
MODEL_NAME = "llama3.2:1b"
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm")
TEMPERATURE = 0.4
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize session state variables
if 'qa' not in st.session_state:
    st.session_state.qa = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def fetch_and_process_website(url):
    """Fetches and processes website content"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        with st.spinner('Fetching website content...'):
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Use a temporary file to store the HTML content
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                delete=False,
                suffix='.html',
            ) as temp_file:
                temp_file.write(response.text)
                temp_file_path = temp_file.name

            try:
                loader = BSHTMLLoader(temp_file_path)
                documents = loader.load()
            except ImportError:
                st.warning("'lxml' is not installed. Falling back to built-in 'html.parser'.")
                loader = BSHTMLLoader(temp_file_path, bs_kwargs={'features': 'html.parser'})
                documents = loader.load()

            # Clean up the temporary file
            os.unlink(temp_file_path)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            texts = text_splitter.split_documents(documents)

            # Guard against oversized chunks that can exceed local embedding context.
            safe_texts = []
            for doc in texts:
                content = doc.page_content.strip()
                if not content:
                    continue
                if len(content) > MAX_EMBED_CHARS:
                    content = content[:MAX_EMBED_CHARS]
                safe_texts.append(Document(page_content=content, metadata=doc.metadata))
            
            return safe_texts

    except Exception as e:
        st.error(f"Error processing website: {str(e)}")
        return None

def check_ollama_server():
    """Checks whether the local Ollama server is reachable before building the pipeline."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except RequestException:
        st.error(
            "Ollama server is not reachable at "
            f"{OLLAMA_BASE_URL}. Start Ollama first with `ollama serve`."
        )
        return False

def initialize_rag_pipeline(texts):
    """Initializes the RAG pipeline with given texts"""
    if not check_ollama_server():
        return None, None

    with st.spinner('Initializing RAG pipeline...'):
        # Set up Ollama language model
        llm = ChatOllama(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
        )
        
        # Create embeddings
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        
        # Create vector store
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
        except ValueError as e:
            if "not found, try pulling it first" in str(e):
                st.error(
                    f'Embedding model "{EMBEDDING_MODEL}" is not available in Ollama. '
                    f'Run: `ollama pull {EMBEDDING_MODEL}`'
                )
                return None, None
            raise
        
        # Set up the retrieval-based QA system
        template = """Context: {context}

        Question: {question}

        Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

        But, if the question is generic, then go ahead and answer the question, example what is a electric vehicle?
        """

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT},
        )
        
        return qa, vectorstore

def main():
    st.title("🤖 RAG Website Query System")
    st.write("Enter a website URL to analyze and ask questions about its content.")
    
    # URL input
    url = st.text_input("Enter website URL:")
    
    # Process button
    if st.button("Process Website") and url:
        texts = fetch_and_process_website(url)
        if texts:
            st.success(f"Successfully processed {len(texts)} text chunks from the website.")
            st.session_state.qa, st.session_state.vectorstore = initialize_rag_pipeline(texts)
            st.session_state.chat_history = []  # Reset chat history for new website
    
    # Show query interface only if pipeline is initialized
    if st.session_state.qa and st.session_state.vectorstore:
        st.write("---")
        st.subheader("Ask Questions")
        
        # Query input
        query = st.text_input("Enter your question:")
        
        if st.button("Ask"):
            if query:
                with st.spinner('Searching for answer...'):
                    # Get relevant documents
                    relevant_docs = st.session_state.vectorstore.similarity_search_with_score(query, k=3)
                    
                    # Display relevant chunks in expander
                    with st.expander("View relevant chunks"):
                        for i, (doc, score) in enumerate(relevant_docs, 1):
                            st.write(f"Chunk {i} (Score: {score:.4f})")
                            st.write(doc.page_content)
                            st.write("---")
                    
                    # Get response
                    response = st.session_state.qa.invoke({"query": query})
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"question": query, "answer": response["result"]})
                
                # Display chat history
                st.write("---")
                st.subheader("Chat History")
                for chat in st.session_state.chat_history:
                    st.write("**Q:** " + chat["question"])
                    st.write("**A:** " + chat["answer"])
                    st.write("---")
    
    # Add sidebar with information
    with st.sidebar:
        st.subheader("About")
        st.write("""
        This is a RAG (Retrieval-Augmented Generation) system that allows you to:
        1. Input any website URL
        2. Process its content
        3. Ask questions about the content
        
        The system uses:
        - Ollama (deepseek-r1) for text generation
        - FAISS for vector storage
        - LangChain for the RAG pipeline
        """)
        
        st.subheader("Model Configuration")
        st.write(f"Model: {MODEL_NAME}")
        st.write(f"Embedding Model: {EMBEDDING_MODEL}")
        st.write(f"Temperature: {TEMPERATURE}")
        st.write(f"Chunk Size: {CHUNK_SIZE}")
        st.write(f"Chunk Overlap: {CHUNK_OVERLAP}")
        st.write(f"Max Embed Chars: {MAX_EMBED_CHARS}")

if __name__ == "__main__":
    main()
