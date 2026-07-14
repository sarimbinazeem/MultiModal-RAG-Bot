# Imports

import os
from pathlib import Path

import fitz #for text and images extractions
import pdfplumber #for table extraction

from PIL import Image
from dotenv import load_dotenv
from google import genai

#Loading API keys
load_dotenv()
API_KEY= os.getenv("GEMINI_API_KEY")

#API Key error handling
if not API_KEY:
     raise ValueError("GEMINI_API_KEY not found in .env file")
 
#Gemini Client
client=genai.Client(api_key=API_KEY)

#All Paths
PDF_PATH= Path("data/Attention_Is_All_You_Need.pdf")
VECTOR_DIR=Path("chroma_db")
EXTRACTED_DIR= Path("extracted")
IMAGE_DIR= EXTRACTED_DIR / "images"
TEXT_DIR= EXTRACTED_DIR / "text"
TABLE_DIR= EXTRACTED_DIR / "tables"

#Created Required Folders 
IMAGE_DIR.mkdir(parents=True,exist_ok=True)
TEXT_DIR.mkdir(parents=True,exist_ok=True)
TABLE_DIR.mkdir(parents=True,exist_ok=True)
IMAGE_DIR.mkdir(parents=True,exist_ok=True)