import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt = "Give me 5 creative company names for a colorful socks brand."

response = llm.invoke(prompt)

print(response.content)