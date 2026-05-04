# LangChain Learning

This repo is a beginner-friendly, step-by-step approach toward learning LangChain with small Python files.

Each file focuses on one concept only, and the code is written in a simple layout with section-wise comments.

## Install These First

Create and activate your Python environment, then install the main packages:

```bash
pip install python-dotenv langchain langchain-community langchain-classic langchain-groq
```

Install tool-specific packages used in later lessons:

```bash
pip install wikipedia numexpr tavily-python langchain-tavily langchain-experimental rank-bm25
```

What these are for:

- `python-dotenv`: loads API keys from `.env`
- `langchain`, `langchain-community`, `langchain-classic`: LangChain core + older tutorial-compatible tools
- `langchain-groq`: connects LangChain to Groq models
- `wikipedia`: used by the action agent
- `numexpr`: required by `llm-math`
- `tavily-python`, `langchain-tavily`: used for web search in Plan and Execute
- `langchain-experimental`: used for Plan and Execute agents
- `rank-bm25`: free keyword-based retriever for document Q&A

## API Keys Needed

Create a `.env` file in this folder:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

`GROQ_API_KEY` is needed for all files because the LLM is `ChatGroq`.

`TAVILY_API_KEY` is only needed for:

```text
06_Plan_and_Execute.py
```

No OpenAI API key is needed in this version of the project.

## Files

### `01_basic_llm_call.py`

Makes a basic call to an LLM using `ChatGroq`.
It sends one prompt and prints the model response.
This is the simplest starting point for understanding how an LLM call works.

### `02_Prompt_Template_Chains.py`

Introduces `PromptTemplate`.
Instead of writing the full prompt manually every time, the prompt has variables.
This teaches how to reuse prompt structures with different inputs.

### `03_Sequential_chains.py`

Shows a simple two-step flow.
The first step generates a company name, and the second step uses that name to create a tagline.
This teaches how one LLM output can become the input for another step.

### `04_Action_agent.py`

Builds an action agent with tools.
It uses Wikipedia to find information and `llm-math` to calculate an answer.
This teaches how an agent decides when to use tools instead of only answering from memory.

### `05_human_as_tool.py`

Shows how a human can act as a tool.
When the agent does not know something, it can ask the user in the terminal.
This teaches human-in-the-loop agent behavior.

### `06_Plan_and_Execute.py`

Creates a Plan and Execute agent.
The planner breaks a task into steps, and the executor completes those steps using Tavily search and math tools.
This teaches how larger tasks can be solved through planning before execution.

### `07_Memory_ChatBot.py`

Creates a chatbot with conversation memory.
The bot remembers earlier messages in the same chat session.
This teaches why memory is useful for follow-up questions and natural conversations.

### `08_History_Context_ChatBot.py`

Shows how chat history can be stored and restored.
It converts previous messages into dictionaries, loads them back, and continues the conversation.
This teaches how saved chat context can be reused later.

### `09_Document_Loading_and_Analysis.py`

Loads text from `dataset.txt`, splits it into chunks, retrieves relevant chunks with BM25, and answers questions using Groq.
This is a free alternative to OpenAI embeddings and Chroma.
Before running it, add text into `dataset.txt`.

## Dataset Note

`09_Document_Loading_and_Analysis.py` expects:

```text
dataset.txt
```

You can use any public text file, such as India-related Wikipedia text or open government information.
If `dataset.txt` is missing or empty, the script will stop and ask you to add text first.

## Recommended Learning Order

Run the files in number order:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09
```

That order starts with one simple LLM call and slowly builds toward agents, memory, and document question answering.
