# ---------------- IMPORTS ----------------
import sys
import warnings
from dotenv import load_dotenv
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.tools import Tool
from langchain_classic.chains.llm_math.base import LLMMathChain
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
# ---------------- ENV SETUP ----------------
load_dotenv()
# ---------------- PROMPT ----------------
prompt = (
    "Where will the next summer olympics be hosted? "
    "What is the population of that country raised to the 0.43 power?"
)

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# ---------------- SEARCH TOOL ----------------
search = TavilySearch(max_results=3)
def search_tavily(query: str) -> str:
    response = search.invoke({"query": query})
    return str(response["results"])

# ---------------- MATH TOOL ----------------
llm_math_chain = LLMMathChain.from_llm(llm=llm)

def calculate(expression: str) -> str:
    response = llm_math_chain.invoke({"question": expression})
    return response["answer"]

# ---------------- TOOLS ----------------
tools = [
    Tool(
        name="Search",
        func=search_tavily,
        description="Useful for searching the web for current facts and events.",
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Useful for math questions and calculations.",
    ),
]

# ---------------- PLANNER ----------------
planner = load_chat_planner(llm)

# ---------------- EXECUTOR ----------------
executor = load_agent_executor(
    llm,
    tools,
    verbose=True,
)

# ---------------- AGENT ----------------
agent = PlanAndExecute(
    planner=planner,
    executor=executor,
    verbose=True,
)

# ---------------- RUN ----------------
response = agent.invoke({"input": prompt})
print(response["output"])
