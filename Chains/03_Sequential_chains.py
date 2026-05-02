import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load API key
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

# ---------------- FIRST STEP ----------------

First_template = """You are a naming consultant.

Generate EXACTLY 1 company name for a company that makes {product}.

Rules:
- Only output the name
- No explanation
- No extra text
"""

First_prompt = PromptTemplate.from_template(First_template)

# This acts like your "first chain"
def First_chain(product):
    formatted_prompt = First_prompt.format(product=product)
    response = llm.invoke(formatted_prompt)
    return response.content.strip()


# ---------------- SECOND STEP ----------------

Second_template = """You are a branding expert.

Create EXACTLY 1 catchy tagline for the company: {company_name}

Rules:
- Only output the tagline
- No explanation
- No extra text
"""

Second_prompt = PromptTemplate.from_template(Second_template)

# Second "chain"
def Second_chain(company_name):
    formatted_prompt = Second_prompt.format(company_name=company_name)
    response = llm.invoke(formatted_prompt)
    return response.content.strip()


# ---------------- SEQUENTIAL FLOW ----------------

company_name = First_chain("colorful socks")
catchphrase = Second_chain(company_name)

print("Company Name:", company_name)
print("Catchphrase:", catchphrase)