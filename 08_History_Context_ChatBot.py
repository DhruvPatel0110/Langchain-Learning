# ---------------- IMPORTS ----------------
import warnings
from dotenv import load_dotenv
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import messages_from_dict, messages_to_dict
from langchain_groq import ChatGroq


# ---------------- ENV SETUP ----------------

load_dotenv()

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", message=".*ConversationChain.*")
warnings.filterwarnings("ignore", message=".*migrating_memory.*")


# ---------------- CREATE CHAT HISTORY ----------------

history = InMemoryChatMessageHistory()

history.add_user_message("hello! let's talk about giraffes")
history.add_ai_message("hi! i'm down to talk about giraffes")


# ---------------- STORE CHAT HISTORY ----------------
stored_messages = messages_to_dict(history.messages)


# ---------------- RETRIEVE CHAT HISTORY ----------------
retrieved_messages = messages_from_dict(stored_messages)
retrieved_history = InMemoryChatMessageHistory(messages=retrieved_messages)
#this is a user created chat history , can also use original conversation history

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

# ---------------- MEMORY ----------------
memory = ConversationBufferMemory(chat_memory=retrieved_history)

# ---------------- CONVERSATION ----------------
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)

# ---------------- RUN ----------------
response = conversation.predict(input="what are they?")
print(response)
