import io
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import loadenv
import os
load_dotenv()
# Initialize FastAPI app
app = FastAPI()

GOOGLE_API_KEY=os.getenv("GEMINI_API_KEY")


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store extracted PDF content
pdf_text = ""

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_bytes):
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise RuntimeError(f"Error reading PDF: {str(e)}")

# Initialize LangChain's Gemini Chat model
gemini_chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", 
    temperature=0.7, 
    google_api_key=GOOGLE_API_KEY
)

# API Endpoint to Upload PDF
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_text
    try:
        pdf_bytes = await file.read()
        pdf_text = extract_text_from_pdf(pdf_bytes)
        if not pdf_text:
            raise HTTPException(status_code=400, detail="Extracted text is empty. Ensure the PDF has readable text.")
        
        logger.info("PDF uploaded and processed successfully.")
        return {"message": "PDF uploaded and processed successfully"}
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# Request Model
class QuestionRequest(BaseModel):
    question: str

# API Endpoint to Ask Questions Based on PDF
@app.post("/ask_question/")
def ask_question(request: QuestionRequest):
    if not pdf_text:
        raise HTTPException(status_code=400, detail="No PDF content available. Upload a PDF first.")
    
    try:
        # Prepare the prompt with the extracted PDF text and user's question
        prompt = f"Extracted Document Text: {pdf_text}\n\nUser Question: {request.question}"
        
        # Use LangChain's Gemini model to generate a response
        response = gemini_chat([
            SystemMessage(content="You are an AI assistant answering questions based on the extracted text from a PDF document."),
            HumanMessage(content=prompt),
        ])
        
        # Ensure response is valid
        if not response or not response.content:
            raise HTTPException(status_code=500, detail="Received an empty response from Gemini AI.")

        logger.info(f"User Question: {request.question}")
        logger.info(f"Chatbot Response: {response.content}")

        return {"answer": response.content}
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
