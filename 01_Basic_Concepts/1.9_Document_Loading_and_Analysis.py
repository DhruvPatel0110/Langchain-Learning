# --------------- IMPORTS ----------------
from pathlib import Path
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- ENV SETUP ----------------
load_dotenv()

# ---------------- DATASET FILE ----------------
dataset_path = Path("dataset.txt")

if not dataset_path.exists() or dataset_path.read_text(encoding="utf-8").strip() == "":
    raise SystemExit("Add text to dataset.txt first, then run this file again.")

# ---------------- LOAD DOCUMENT ----------------
loader = TextLoader(str(dataset_path), encoding="utf-8")
documents = loader.load()

# ---------------- SPLIT DOCUMENT ----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
texts = text_splitter.split_documents(documents)

# ---------------- RETRIEVER ----------------
retriever = BM25Retriever.from_documents(texts)
retriever.k = 3

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

# ---------------- QUESTION ANSWER CHAIN ----------------
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)

# ---------------- RUN ----------------
question = "provide data relevent to indus river "
response = chain.invoke({"query": question})
print(response["result"])
