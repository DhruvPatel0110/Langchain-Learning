"""
Minimal RAG demo converted from LCRAG_demo1.ipynb.

Dependencies:
    pip install langchain langchain-community sentence-transformers chromadb transformers torch accelerate python-dotenv

This script is designed to run in a local Python environment without Colab-specific commands.
If you want a smaller model, set MODEL_NAME to "google/flan-t5-small" or another compatible Flan model.

API keys are loaded from .env file automatically.
"""

#--------------------------------- IMPORTS & CONFIGURATION --------------------------------
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_groq import ChatGroq
from transformers import AutoTokenizer, pipeline
import torch

# Model configuration
DEFAULT_MODEL_NAME = "google/flan-t5-small"  # Local LLM fallback
GROQ_MODEL = "llama-3.1-8b-instant"  # Remote Groq LLM


#--------------------------------- DOCUMENT CREATION --------------------------------
# Creates sample documents for the knowledge base (in-memory Document objects)
# These are stored in ChromaDB vector database for semantic search
def create_documents():
    """
    Creates a list of sample Document objects for the RAG system.
    Each Document has:
    - page_content: The actual text content
    - metadata: Additional info (source identifier)

    Returns:
        list: List of Document objects
    """
    return [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. It provides tools for document loading, splitting, embeddings, and chains.",
            metadata={"source": "doc1"},
        ),
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation. It combines retrieval of relevant documents with text generation. First, relevant documents are retrieved based on similarity, then an LLM generates an answer using those documents as context.",
            metadata={"source": "doc2"},
        ),
        Document(
            page_content="Vector databases store embeddings and allow for semantic similarity search. They convert text into numerical vectors that capture meaning.",
            metadata={"source": "doc3"},
        ),
        Document(
            page_content="ChromaDB is an open-source embedding database that works well with LangChain. It's lightweight and perfect for development.",
            metadata={"source": "doc4"},
        ),
        Document(
            page_content="HuggingFace provides thousands of pre-trained models that can run locally or in the cloud. Popular models include FLAN-T5, GPT-2, and Llama.",
            metadata={"source": "doc5"},
        ),
        Document(
            page_content="Embeddings are numerical representations of text that capture semantic meaning. Similar texts have similar embeddings.",
            metadata={"source": "doc6"},
        ),
    ]


#--------------------------------- VECTOR DATABASE & EMBEDDINGS --------------------------------
# Splits documents into chunks and stores them in ChromaDB with embeddings
def build_vector_store(documents):
    """
    Builds a vector database from documents using embeddings.

    Process:
    1. Split documents into manageable chunks
    2. Convert chunks to embeddings (numerical representations)
    3. Store embeddings in ChromaDB for semantic search

    Args:
        documents: List of Document objects

    Returns:
        vectorstore: ChromaDB instance for semantic search
    """
    # Step 1: Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} document chunks")

    # Step 2: Load embedding model (converts text to numerical vectors)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded")

    # Step 3: Create ChromaDB vector store from chunked documents
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_demo",
    )
    print("Vector database created")
    return vectorstore


#--------------------------------- LANGUAGE MODEL SETUP --------------------------------
# Creates an LLM instance for text generation (either remote Groq or local HuggingFace)
def create_llm(model_name: str):
    """
    Creates a Language Model instance for text generation.

    Two modes:
    - REMOTE: Uses Groq API if GROQ_API_KEY is set in .env
    - LOCAL: Falls back to local HuggingFace model if no API key

    Args:
        model_name: Name of HuggingFace model to use as fallback

    Returns:
        llm: LLM instance (ChatGroq or HuggingFacePipeline)
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")

    # Use Groq API if key is available (runs on their servers, faster)
    if groq_api_key:
        print("Using remote Groq API for generation")
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=groq_api_key,
            temperature=0.7,  # Creativity level (0=deterministic, 1=creative)
            max_tokens=512,   # Maximum response length
            max_retries=2,    # Retry failed requests
        )

    # Fallback: Use local HuggingFace model (runs on your CPU/GPU)
    print("Using local HuggingFace model for generation")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_new_tokens=256,
        device=device,
        model_kwargs={
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
        },
    )
    print("Language model loaded")
    return HuggingFacePipeline(pipeline=pipe)


#--------------------------------- PROMPT TEMPLATE --------------------------------
# Defines the instruction template for the LLM
def create_prompt_template():
    """
    Creates a prompt template that instructs the LLM how to answer questions
    using retrieved context.

    The template includes:
    - Context from retrieved documents
    - The user's question
    - Instructions on how to respond

    Returns:
        prompt: PromptTemplate instance
    """
    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and relevant.

Context: {context}

Question: {question}

Answer:"""

    return PromptTemplate(template=template, input_variables=["context", "question"])


#--------------------------------- RAG CHAIN CONSTRUCTION --------------------------------
# Combines retriever, LLM, and prompt into a complete RAG pipeline
def build_qa_chain(llm, retriever):
    """
    Builds a complete Retrieval-Augmented Generation (RAG) chain.

    The chain workflow:
    1. Receives a question
    2. Uses retriever to find similar documents from vector store
    3. Combines retrieved docs + question using prompt template
    4. Passes to LLM for answer generation

    Args:
        llm: Language model instance for generation
        retriever: Vector store retriever for semantic search

    Returns:
        qa_chain: Complete RetrievalQA chain
    """
    prompt = create_prompt_template()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = concatenate retrieved docs into prompt
        retriever=retriever,
        return_source_documents=True,  # Return which documents were used
        chain_type_kwargs={"prompt": prompt},
    )
    print("RAG chain ready!")
    return qa_chain


#--------------------------------- QUERY & OUTPUT --------------------------------
# Handles user queries and displays RAG pipeline results
def ask_question(chain, query: str):
    """
    Executes a query through the RAG pipeline and displays results.

    Shows:
    - Retrieved documents used
    - Generated answer from LLM

    Args:
        chain: RetrievalQA chain instance
        query: User question string
    """
    print(f"\nQuestion: {query}")
    print("\nStep 1: Retrieving relevant documents...")

    # Execute the RAG chain
    result = chain({"query": query})

    # Display retrieved documents
    print(f"Retrieved {len(result['source_documents'])} documents:")
    for i, doc in enumerate(result["source_documents"], start=1):
        snippet = doc.page_content[:150].replace("\n", " ")
        print(f"\n  {i}. {snippet}...")
        print(f"     Source: {doc.metadata.get('source', 'unknown')}")

    # Display LLM-generated answer
    print("\nStep 2: Generating answer using LLM...")
    print(f"\nAnswer: {result['result']}")
    print("\n" + "-" * 60)


#--------------------------------- MAIN EXECUTION --------------------------------
# Orchestrates the complete RAG pipeline setup and execution
def main():
    """
    Main execution function that:
    1. Creates sample documents
    2. Builds vector database with embeddings
    3. Initializes LLM (Groq or local)
    4. Constructs RAG chain
    5. Runs demo queries
    """
    print("Starting Complete RAG Pipeline Demo")
    print("=" * 60)

    # Step 1: Create knowledge base documents
    documents = create_documents()

    # Step 2: Build vector store from documents
    vectorstore = build_vector_store(documents)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Retrieve top 2 matches

    # Step 3: Initialize Language Model (Groq or local)
    print("Loading language model...")
    llm = create_llm(DEFAULT_MODEL_NAME)

    # Step 4: Build complete RAG chain
    qa_chain = build_qa_chain(llm, retriever)

    # Step 5: Execute demo queries
    print("\nRunning Demo Queries...\n")
    ask_question(qa_chain, "What is RAG?")
    ask_question(qa_chain, "What are embeddings and why are they useful?")
    ask_question(qa_chain, "Which database should I use for vector storage?")


#--------------------------------- ENTRY POINT --------------------------------
# Run the script only when executed directly (not when imported)
if __name__ == "__main__":
    main()
