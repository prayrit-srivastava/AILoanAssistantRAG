import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import GOOGLE_API_KEY
from prompts import LOAN_ASSISTANT_PROMPT


# -----------------------------
# Gemini
# -----------------------------

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# -----------------------------
# Load Embedding Model
# -----------------------------

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# -----------------------------
# Load FAISS Database
# -----------------------------

vector_db = FAISS.load_local(
    "vector_db",
    embedding_model,
    allow_dangerous_deserialization=True
)


# -----------------------------
# Main Function
# -----------------------------

def ask_loan_assistant(question: str):
    
    retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = LOAN_ASSISTANT_PROMPT.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    return response.content