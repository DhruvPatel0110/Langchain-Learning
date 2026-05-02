# ---------------- IMPORTS ----------------
from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents import initialize_agent
from langchain_groq import ChatGroq

# ---------------- ENV SETUP ----------------
load_dotenv()

# ---------------- PROMPT ----------------
prompt = (
    "Who was the third president of USA, when was he born, "
    "and what is that birth year raised to the power of 3?"
)

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# ---------------- TOOLS ----------------
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# ---------------- AGENT ----------------
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
)

# ---------------- RUN ----------------
response = agent.invoke(prompt)
print(response["output"])