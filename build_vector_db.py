import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import GOOGLE_API_KEY

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ----------------------------------------
# STEP 1: Locate all PDF files
# ----------------------------------------

DATA_FOLDER = Path("data")
VECTOR_DB_PATH = "vector_db"

pdf_files = list(DATA_FOLDER.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found inside the data folder."
    )

print(f"\nFound {len(pdf_files)} PDF(s).\n")


# ----------------------------------------
# STEP 2: Load all documents
# ----------------------------------------

documents = []

for pdf in pdf_files:
    print(f"Loading: {pdf.name}")

    loader = PyPDFLoader(str(pdf))
    documents.extend(loader.load())

print(f"\nLoaded {len(documents)} pages.\n")


# ----------------------------------------
# STEP 3: Split into chunks
# ----------------------------------------

print("Splitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.\n")


# ----------------------------------------
# STEP 4: Load Embedding Model
# ----------------------------------------

print("Loading embedding model...")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# ----------------------------------------
# STEP 5: Create FAISS Vector Store
# ----------------------------------------

print("Generating embeddings...")

vector_db = FAISS.from_documents(
    chunks,
    embedding_model
)


# ----------------------------------------
# STEP 6: Save Vector Database
# ----------------------------------------

print("Saving vector database...")

vector_db.save_local(VECTOR_DB_PATH)

print("\n===================================")
print("Vector Database Created Successfully!")
print(f"Saved to: {VECTOR_DB_PATH}/")
print("===================================\n")