# 02_RAG

This folder focuses on **Retrieval-Augmented Generation (RAG)**.

It builds on the basic LangChain concepts and introduces how LLMs can work with external data sources.

This section represents an **intermediate level**, where instead of relying only on model knowledge, we:
- retrieve relevant information  
- inject it into prompts  
- generate grounded answers  

---

## Install These First

Create and activate your Python environment, then install the main packages:

```bash
pip install python-dotenv langchain langchain-community langchain-core langchain-text-splitters chromadb sentence-transformers langchain-groq
```

These libraries may download **4–5 GB of data**, especially embedding models.

---

## Recommended Environment

It is recommended to use **Google Colab** for this section.

- avoids heavy local downloads  
- better runtime stability  
- faster execution for embeddings and vector storage  

---

## API Keys Needed

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Groq is used for fast LLM inference without running models locally.

---

## Files

### `LCRAG_demo.ipynb`

Implements a complete RAG pipeline step-by-step.

---

## Pipeline Breakdown

### Step 1: Download Libraries

Installs required LangChain modules, vector store, embeddings, and Groq integration.  
Prepares the environment for building the RAG system.

---

### Step 2: Import Libraries

Imports all necessary components such as text splitters, embeddings, vector stores, LLM, prompt templates, and LCEL tools.  
Sets up all building blocks used in the pipeline.

---

### Step 3: Create Sample Documents

Creates a small dataset using `Document` objects.  
Each document contains text content and metadata.  
This acts as the knowledge base for retrieval.

---

### Step 4: Split Documents

Uses `RecursiveCharacterTextSplitter` to break documents into smaller chunks.  
Chunking improves retrieval accuracy and context relevance.

---

### Step 5: Create Embeddings

Loads a HuggingFace embedding model (`all-MiniLM-L6-v2`).  
Converts text chunks into vector representations for similarity search.

---

### Step 6: Create Vector Store

Stores embeddings in **ChromaDB**.  
This enables efficient semantic search over the document chunks.

---

### Step 7: Set Up Retriever

Creates a retriever from the vector store.  
Retrieves top relevant chunks (`k=2`) based on similarity to the query.

---

### Step 8: Load LLM (Groq)

Initializes the LLM using Groq (`llama-3.1-8b-instant`).  
Provides fast inference without local model downloads.

---

### Step 9: Create Prompt Template

Defines a structured prompt using `PromptTemplate`.  
Combines retrieved context and user query to guide the LLM response.

---

### Step 10: Create RAG Chain

Builds the full pipeline using LCEL.  
Connects retriever, prompt, LLM, and output parser into a single flow.

---

### Step 11: Test the System

Runs sample queries through the pipeline.  
Retrieves relevant documents and generates final answers.

---

## Sample Project (Coming Soon)

```text
Sample_Project_RAG.ipynb
```

This will demonstrate a real-world use case using a custom dataset and modified prompts.

