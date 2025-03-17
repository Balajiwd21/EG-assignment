import logging
import requests
import pickle
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from ollama import chat, embeddings

# Initialize FastAPI
app = FastAPI()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Credentials & Endpoints
API_BASE_URL = ""
HEADERS = {
    "user": "",
    "pwd": "",
    "managerurl": "",
    "accessID": "",
}
API_ENDPOINTS = {
    "metrics": "getLiveMeasure",
    "alerts": "getAlerts"
}

FAISS_INDEX_PATH = "faiss_index.pkl"

# Function to Convert API JSON Response into Meaningful Sentences
def convert_api_response_to_sentences(api_data):
    """Convert JSON API response into meaningful sentences for embedding."""
    sentences = []

    if "summary" in api_data:
        for item in api_data["summary"]:
            priority = "Critical" if "critical" in item else "Major" if "major" in item else "Minor"
            sentences.append(f"There is {item[priority.lower()]} {priority.lower()} issue(s) detected.")

    if "total" in api_data:
        sentences.append(f"Total alerts detected: {api_data['total']}.")

    if "data" in api_data:
        for alert in api_data["data"]:
            sentences.append(f"Alert: {alert['description']}.")
            sentences.append(f"Component: {alert['componentName']} ({alert['componentType']}).")
            sentences.append(f"Measure: {alert['measure']} under test '{alert['test']}'.")
            sentences.append(f"Priority: {alert['priority'].capitalize()} (Layer: {alert['layer']}).")
            sentences.append(f"Start Time: {alert['startTime']}. Alarm ID: {alert['alarmID']}.")

    return sentences

# Fetch API Data & Convert to Meaningful Sentences
def fetch_all_api_data():
    """Fetch data from APIs and convert it into structured text."""
    documents = []

    for api_name, endpoint in API_ENDPOINTS.items():
        try:
            logger.info(f"Fetching data from {endpoint} API...")
            url = f"{API_BASE_URL}/{endpoint}"
            payload = (
                {"componentName": "10.200.2.192:7077", "componentType": "eG Manager"}
                if api_name == "metrics"
                else {"filterBy": "ComponentType", "filterValues": "eG Manager"}
            )

            response = requests.post(url, headers=HEADERS, json=payload)
            response.raise_for_status()
            api_data = response.json()

            if isinstance(api_data, list):
                api_data = api_data[0]

            sentences = convert_api_response_to_sentences(api_data)
            for sentence in sentences:
                documents.append(Document(page_content=sentence))

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {endpoint}: {e}")

    return documents

# Ollama Local Embedding Model
def ollama_embedding_fn(text: str):
    """Generate embeddings using Ollama's `all-minilm` model."""
    response = embeddings(model="all-minilm", prompt=text)

    if "embedding" not in response:
        raise RuntimeError("Ollama embedding failed.")

    return np.array(response["embedding"], dtype=np.float32)

class LocalOllamaEmbeddings(Embeddings):
    """Custom LangChain Embedding Model using Ollama."""
    def embed_documents(self, texts):
        return [ollama_embedding_fn(text) for text in texts]

    def embed_query(self, text):
        return ollama_embedding_fn(text)

# Load or Create FAISS Index
def load_faiss_index():
    """Load FAISS index if it exists, otherwise create a new one."""
    if os.path.exists(FAISS_INDEX_PATH):
        logger.info("Loading existing FAISS index...")
        with open(FAISS_INDEX_PATH, "rb") as f:
            return pickle.load(f)

    logger.info("No FAISS index found. Creating a new one from API data...")
    documents = fetch_all_api_data()

    if not documents:
        logger.error("No documents fetched from API. FAISS index cannot be created.")
        return None  # Prevent empty FAISS index

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunks, embedding=LocalOllamaEmbeddings())

    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(vector_store, f)

    return vector_store

@app.on_event("startup")
def load_api_data():
    """Initialize FAISS retriever at startup."""
    global retriever
    retriever = load_faiss_index()

    if retriever:
        retriever = retriever.as_retriever()
        logger.info("FAISS retriever initialized.")
    else:
        logger.error("FAISS retriever could not be initialized.")

# API Schema for User Queries
class QuestionRequest(BaseModel):
    question: str

# Retrieve Relevant Answers from FAISS
def extract_relevant_info(user_query, retrieved_docs):
    """Extract relevant metric from FAISS documents."""
    for doc in retrieved_docs:
        if user_query.lower() in doc.page_content.lower():
            return doc.page_content

    return None  # Return None instead of a default message

# Generate Chat Response Using FAISS & Ollama
def generate_chat_response(user_query: str):
    """Retrieve relevant data from FAISS and use it to generate an AI response."""
    try:
        if retriever:
            retrieved_docs = retriever.get_relevant_documents(user_query)
            logger.info(f"Retrieved Docs: {retrieved_docs}")

            metric_response = extract_relevant_info(user_query, retrieved_docs)

            if metric_response:
                return metric_response  # Return valid FAISS response

        # If FAISS fails, use Ollama as a fallback
        response = chat(model="llama3.2:3b", messages=[
            {"role": "system", "content": "You are an AI assistant providing real-time server analytics."},
            {"role": "user", "content": user_query}
        ])

        return response.get("message", {}).get("content", "Error: Unable to retrieve AI response.")

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise HTTPException(status_code=500, detail="API request failed.")

    except KeyError as e:
        logger.error(f"Unexpected response format: {e}")
        raise HTTPException(status_code=500, detail="Unexpected response format.")

    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# API Endpoint for Asking Questions
@app.post("/ask_question/")
async def ask_question(request: QuestionRequest):
    """Chatbot API endpoint to handle user queries."""
    try:
        bot_response = generate_chat_response(request.question)
        return {"answer": bot_response}

    except HTTPException as e:
        raise e

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
