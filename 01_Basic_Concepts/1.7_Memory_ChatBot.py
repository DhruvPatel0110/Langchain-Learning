# ---------------- IMPORTS ----------------

import warnings

from dotenv import load_dotenv
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_core._api import LangChainDeprecationWarning
from langchain_groq import ChatGroq

# ---------------- ENV SETUP ----------------
load_dotenv()
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", message=".*ConversationChain.*")
warnings.filterwarnings("ignore", message=".*migrating_memory.*")


# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)


# ---------------- CONVERSATION MEMORY ----------------
conversation = ConversationChain(
    llm=llm,
    verbose=False,
)

# ---------------- MEMORY EXAMPLE ----------------
# print(conversation.predict(input="Hi there!"))
# print(conversation.predict(input="Can we talk about weather?"))
# print(conversation.predict(input="It's a beautiful day today."))

# ---------------- CHAT BOT ----------------
chatbot = ConversationChain(llm=llm)
print("Welcome to your AI chatbot! What's on your mind?")
for _ in range(3):
    human_input = input("_____You: ")
    ai_response = chatbot.predict(input=human_input)
    print(f"_____AI: {ai_response}")
