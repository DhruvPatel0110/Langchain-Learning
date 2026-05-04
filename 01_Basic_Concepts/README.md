# 01_Basic_Concepts

This folder is a structured walkthrough of the **foundational concepts of LangChain**.

It focuses on building a strong base by covering the **core components step-by-step**, before moving to advanced topics like RAG and complex pipelines.

Each file demonstrates one concept in isolation, making it easier to understand how different parts of LangChain work individually.

---

## Files Overview

### 1.1_basic_llm_call.py

Demonstrates a basic interaction with an LLM.  
A simple prompt is sent to the model and the response is printed.  
This is the starting point for understanding how LLM calls work.  

**Topic:** Basic LLM usage

---

### 1.2_Prompt_Template_Chains.py

Introduces `PromptTemplate` for dynamic prompt creation.  
Instead of hardcoding prompts, variables are used to make prompts reusable.  
This improves flexibility and prompt structure.  

**Topic:** Prompt engineering

---

### 1.3_Sequential_chains.py

Shows how multiple LLM calls can be chained together.  
The output of one step becomes the input of another step.  
This demonstrates building simple multi-step workflows.  

**Topic:** Sequential pipelines

---

### 1.4_Action_agent.py

Implements an agent that can use external tools.  
The agent decides when to fetch information or perform calculations.  
This introduces decision-making beyond simple LLM responses.  

**Topic:** Agents and tool usage

---

### 1.5_human_as_tool.py

Demonstrates how a human can act as a tool in the system.  
The agent can request input from the user when needed.  
This introduces human-in-the-loop workflows.  

**Topic:** Human-in-the-loop systems

---

### 1.6_Plan_and_Execute.py

Implements a planning-based agent architecture.  
The system first breaks a task into steps and then executes them sequentially.  
This is useful for solving complex multi-step problems.  

**Topic:** Planning and execution agents

---

### 1.7_Memory_ChatBot.py

Builds a chatbot with conversation memory.  
The bot retains previous interactions within a session.  
This allows more natural and context-aware conversations.  

**Topic:** Memory in LLM applications

---

### 1.8_History_Context_ChatBot.py

Shows how chat history can be stored and reused.  
Past interactions are saved and reloaded to continue conversations.  
This enables persistent conversational systems.  

**Topic:** Context persistence

---

### 1.9_Document_Loading_and_Analysis.py

Loads external text data and processes it for question answering.  
The document is split into chunks and relevant parts are retrieved.  
The LLM then generates answers based on the retrieved content.  

**Topic:** Document processing and retrieval