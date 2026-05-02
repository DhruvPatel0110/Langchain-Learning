# ---------------- IMPORTS ----------------
import warnings
from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents import AgentType, initialize_agent
from langchain_core._api import LangChainDeprecationWarning
from langchain_groq import ChatGroq

# ---------------- ENV SETUP ----------------
load_dotenv()
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# ---------------- PROMPT ----------------
prompt = "What's my friend Adi's surname?"

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# ---------------- TOOLS ----------------
tools = load_tools(["human"], llm=llm)

# ---------------- AGENT ----------------
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# ---------------- RUN ----------------
response = agent_chain.invoke(prompt)
print(response["output"])
