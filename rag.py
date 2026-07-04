import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import GOOGLE_API_KEY
from prompts import LOAN_ASSISTANT_PROMPT


# -----------------------------
# Load Embedding Model
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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
# Gemini
# -----------------------------

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
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