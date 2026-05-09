# 02_RAG

This folder covers **Retrieval-Augmented Generation (RAG)** with LangChain.

The main idea is simple: instead of asking the LLM to answer only from its
training data, we retrieve useful external context first and pass that context
into the prompt. This makes answers more grounded, especially for private
documents, websites, and recent information.

## What Is Inside

| File | Purpose |
| --- | --- |
| `2.1_RAG_Pipeline.ipynb` | Basic RAG pipeline using sample documents, ChromaDB, embeddings, a retriever, and Groq. |
| `2.2_RAG_with_History.ipynb` | RAG with conversation memory, query rewriting, multiple web sources, FAISS, and interactive commands. |
| `2.3_Multisource_RAG.ipynb` | Smart multi-source RAG that routes questions to LLM-only, company policy docs, Wikipedia, or SerpAPI. |
| `Policy.txt` | Private policy handbook for the sample company `XYZ Pvt Ltd`. Used by document RAG. |
| `Output screens/` | Screenshots showing the different outputs and routing behavior. |

## Install These First

For the core notebooks:

```bash
pip install python-dotenv langchain langchain-community langchain-core langchain-text-splitters sentence-transformers langchain-groq
```

For vector stores and the multi-source examples:

```bash
pip install chromadb faiss-cpu beautifulsoup4 requests google-search-results
```

Embedding models may take time and disk space to download the first time.

## Recommended Environment

Google Colab is recommended for these notebooks because:

- package installation is easier
- embeddings and vector stores run more smoothly
- Colab secrets can store API keys safely
- local dependency conflicts are easier to avoid

The notebooks can also run locally, but you may need to adjust the API-key and
file-upload cells.

## API Keys Needed

For normal RAG and LLM responses:

```env
GROQ_API_KEY=your_groq_api_key_here
```

For latest-news or real-time search in `2.3_Multisource_RAG.ipynb`:

```env
SERPAPI_API_KEY=your_serpapi_key_here
```

In Colab, the multi-source notebook reads these from Colab secrets using
`userdata.get(...)`.

## 2.1 - Basic RAG Pipeline

`2.1_RAG_Pipeline.ipynb` builds the first complete RAG flow.

It:

- creates small sample `Document` objects
- splits text with `RecursiveCharacterTextSplitter`
- creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- stores vectors in ChromaDB
- creates a retriever with top-k similarity search
- uses Groq with `llama-3.1-8b-instant`
- builds an LCEL chain with retriever, prompt, LLM, and output parser
- tests questions like "What is RAG?" and "What are embeddings?"

This notebook is the foundation for the later examples.

## 2.2 - RAG With Chat History

`2.2_RAG_with_History.ipynb` adds memory and web-based source documents.

It loads multiple reference pages, including pages about:

- Artificial intelligence
- Large language models
- LangChain
- Retrieval-augmented generation
- Vector databases
- OpenAI

Then it:

- chunks the loaded web documents
- creates embeddings
- stores them in FAISS
- retrieves relevant chunks
- rewrites follow-up questions into standalone questions using chat history
- stores conversation messages with `InMemoryChatMessageHistory`
- supports interactive commands like `sources`, `clear`, and `exit`

This is useful when a user asks follow-up questions such as "What about its
use cases?" where the system needs previous conversation context.

## 2.3 - Multisource RAG

`2.3_Multisource_RAG.ipynb` is the main multi-source demo.

It has four answer paths:

| Route | Source | When It Is Used | Example |
| --- | --- | --- | --- |
| LLM only | Model training data | Default route when no special source is needed | General questions |
| Document RAG | `Policy.txt` company policy file | When the prompt contains `xyz` | `notice period for xyz` |
| Wikipedia RAG | Wikipedia pages loaded as web documents | For normal factual/company/model questions | `what is anthropic` |
| SerpAPI RAG | Real-time web search through SerpAPI | For latest, recent, news, or launch queries | `anthropic's latest model` |

### Sources Used

1. `Policy.txt`
   - Contains the employee handbook and user/company policies for the sample
     company `XYZ Pvt Ltd`.
   - The notebook chunks this file, embeds it, stores it in ChromaDB, and uses
     it whenever the question includes `xyz`.

2. Wikipedia
   - Used for generic factual information.
   - The notebook currently loads Wikipedia pages for `Anthropic` and
     `Claude (language model)` using `WebBaseLoader`.

3. SerpAPI
   - Used for current or latest information.
   - The notebook checks for words like `latest`, `recent`, `news`, or
     `launch`, then searches the web and passes the results to the LLM.

### Routing Logic

The notebook uses a simple keyword router:

```python
q = question.lower()

if "xyz" in q:
    source = "doc"
elif "latest" in q or "recent" in q or "news" in q or "launch" in q:
    source = "serp"
elif "anthropic" in q or "claude" in q or "company" in q:
    source = "wiki"
else:
    source = "llm"
```

This makes the behavior easy to understand:

- private company policy questions go to the private document
- general factual questions go to Wikipedia
- latest/current questions go to SerpAPI
- everything else falls back to the LLM

## Output Examples

The `Output screens/` folder shows the main output checks:

| Screenshot | What It Shows |
| --- | --- |
| `Notice period no docs.jpeg` | Regular LLM answer without the policy document. |
| `notice period xyz.jpeg` | Policy-aware answer after routing to `Policy.txt`. |
| `food on desk no docs.jpeg` | Regular LLM answer without company policy context. |
| `food on desk xyz.jpeg` | Policy-aware answer using `Policy.txt`. |
| `anthropic wikipedia.jpeg` | Generic information answered using Wikipedia context. |
| `anthropic latest model serpapi.jpeg` | Latest/current information answered using SerpAPI search results. |

These examples demonstrate why source routing matters. The same LLM can answer
very differently depending on whether it is using only training data, private
documents, static web pages, or real-time search results.

## Overall Flow

1. Install dependencies.
2. Load API keys.
3. Load source data.
4. Split documents into chunks.
5. Create embeddings.
6. Store chunks in a vector database.
7. Retrieve relevant context for a question.
8. Route the question to the correct source.
9. Pass retrieved context to the LLM.
10. Generate the final answer.

## Notes

- `Policy.txt` is the local private source for XYZ company policies.
- Wikipedia is better for stable, general knowledge.
- SerpAPI is better for recent or changing information.
- Regular LLM answers are useful as a fallback, but they should not be trusted
  for private company rules or latest news.
- The routing in `2.3_Multisource_RAG.ipynb` is intentionally simple so the
  source-selection behavior is easy to see and modify.
