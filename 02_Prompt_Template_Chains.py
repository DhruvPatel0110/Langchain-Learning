import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize LLM (Groq)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.9
)

# Define template
template = """You are a naming consultant for new companies.
give exactly 5 {company} names for a company that makes {product}?
Rules:
- Only output names
- No explanations
- One name per line"""

# Create prompt template
prompt = PromptTemplate.from_template(template)

# Format prompt with input
formatted_prompt = prompt.format(company="XYZ Startup", product="colorful socks")

# Invoke LLM
response = llm.invoke(formatted_prompt)

# Print result
print(response.content)