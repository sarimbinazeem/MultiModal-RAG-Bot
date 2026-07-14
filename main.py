# Imports

import os
from pathlib import Path
import json

import fitz #for text and images extractions
import pdfplumber #for table extraction

from PIL import Image
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

#Loading API keys
load_dotenv()
API_KEY= os.getenv("GEMINI_API_KEY")

#API Key error handling
if not API_KEY:
     raise ValueError("GEMINI_API_KEY not found in .env file")
 
#Gemini Client
client=genai.Client(api_key=API_KEY)
MODEL= "gemini-2.5-flash"
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=API_KEY
)



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
VECTOR_DIR.mkdir(parents=True,exist_ok=True)

#Prompt Variable
PROMPT= """

You are a helpful assistant.

You are analyzing a figure from the research paper:
Attention is All you need

Describe the image in detail.

If the image contains:
- Flowcharts
- Architechure Diagrams
- Mathematical Equations
- Attention mechanism
-Graphs
- Charts
- Tables

Please explain them carefully and should contain everything that is inside the pdf image.
Do NOT simply list objects.
Explain what the figure is trying to teach
"""

#Knowledge base
documents=[]

#Text Splitter
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

#Extract Text fUNCTION
def extract_text():
    print("\n Extracting Text...")
    
    document = fitz.open(PDF_PATH)
    
    for pageno,page in enumerate(document):
        text = page.get_text()
        
        file_path= TEXT_DIR/ f"page_{pageno +1}.txt"
        
        #write and create the file in text directory
        with open(file_path,"w",encoding="utf-8") as f:
            f.write(text)
            
    document.close()
    print("\nText extraction completed.")
    

#Extract Image FUnction
def extract_images():
    print("\nExtracting images...")
    
    document = fitz.open(PDF_PATH)
    
    image_counter =1
    
    for pageno,page in enumerate(document):
        
        #getting images list to loop through it
        image_list= page.get_images(full=True)
        
        for image in image_list:
            xref=image[0] #it returns a page ID of that image
            base_image= document.extract_image(xref)
            
            #getting image dta
            image_bytes= base_image["image"]
            image_extention= base_image["ext"]
            image_name= f"page_{pageno +1}_img_{image_counter}.{image_extention}"
            image_path= IMAGE_DIR/ image_name
            
            with open(image_path,"wb") as file:
                file.write(image_bytes)
            image_counter+=1
            
    document.close()
    print("Image extraction completed.")
    
    
#Extracting Tables
def extract_tables():
    print("\nExtracting tables...")

    with pdfplumber.open(PDF_PATH) as pdf:
        table_counter = 1
        
        for pageno,page in enumerate(pdf):
            tables= page.extract_tables() 
            
            for table in tables:
                file_name= TABLE_DIR/f"page_{pageno+1}_table_{table_counter}.txt"
                
                with open(file_name,"w",encoding="utf-8") as file:
                    for row in table:
                        #row list
                        row=[str(data) if data else "" for data in row]
                        #writing in file
                        file.write("|".join(row) + "\n")
                        
                table_counter +=1
                

    print("Table extraction completed.")
    
def describe_image():

    print("\nDescribing images using Gemini...")
    
    descriptions=[]
    
    for image_file in IMAGE_DIR.iterdir():
        if image_file.suffix.lower() not in [".png",".jpg",".jpeg"]:
            continue
        image = Image.open(image_file)
        
        response=client.models.generate_content(
            model=MODEL,
            contents=[
                PROMPT,
                image
            ]
        ) 
        
        descriptions.append(
            {
                "image": image_file.name,
                "description": response.text
            }
        )
        
    output_path = EXTRACTED_DIR/ "image_descriptions.json"
        
    with open(output_path,"w",encoding="utf-8" ) as file:
         json.dump(descriptions,file,indent=4,ensure_ascii=False) 
            
    print("Image descriptions saved successfully.")
    
def build_knowledge():
    print("\nBuilding knowledge base...")
    
    #Text
    for file in TEXT_DIR.iterdir():
        
        if file.suffix != ".txt":
            continue
        
        with open(file, "r", encoding="utf-8") as f:

            text = f.read()

        chunks = text_splitter.split_text(text)

        for i, chunk in enumerate(chunks):

            documents.append({

                "content": chunk,

                "metadata":{

                    "type":"text",

                    "source":file.name,

                    "chunk":i+1

                }

            })
            
            #the documents have chunks now instead of pages
        
    #Tables
    for file in TABLE_DIR.iterdir():
        if file.suffix != ".txt":
            continue
        
        with open(file,"r",encoding="utf-8") as f:
            documents.append(
                {

                "content":f.read(),

                "metadata":{

                    "type":"table",

                    "source":file.name
                }
                }
            )
            
    #Images
    with open(EXTRACTED_DIR/"image_descriptions.json","r",encoding="utf-8") as file:
        image_data=json.load(file)
        
    for image in image_data:
        documents.append(
            {
                "content":image["description"],
                "metadata":{

                "type":"image",

                "source":image["image"]
                }
            }
        )
        
    print(f"\nKnowledge Base Created!")

    print(f"Documents : {len(documents)}")
    
#Creating documents into Document() format
def convert_documents():
    docs=[]
    
    for item in documents:
        docs.append(
            Document(
                page_content=item["content"],
                metadata=item["metadata"]
            )
        )
        
    return docs
        
#Vector database creation
def create_vector_database():
    print("\nCreating Vector Database...")
    docs= convert_documents()
    
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=str(VECTOR_DIR),
    ) 
    
    print("Vector Database Created Successfully!")

    return db

#retrieval of user query
def retrieval_documents(query):
    return retriever.invoke(query)

def retrieval(query,documents):
    

    print("\nRETRIEVED CONTEXT")
    print("=" * 70)

    for i, doc in enumerate(documents, start=1):

        print(f"\nResult {i}")

        print(f"Type   : {doc.metadata.get('type')}")

        print(f"Source : {doc.metadata.get('source')}")

        if "chunk" in doc.metadata:
            print(f"Chunk  : {doc.metadata.get('chunk')}")

        print("\nContent:")

        print(doc.page_content[:500])
        print()
        
#RAG PROMPT
RAG_PROMPT= """
You are a helpful assistant.
Answer the user's question ONLY using the retrieved context provided below.

If the answer is not present in the context, reply:

"I couldn't find the answer in the retrieved context."

Do not use outside knowledge.

Retrieved Context:

{context}

Question:

{question}
"""

#To build context
def build_context(retrievedDocuments):
    context=""
    for doc in retrievedDocuments:
        context += f"""
            Type: {doc.metadata.get("type")}
            Source: {doc.metadata.get("source")}

            {doc.page_content}

            ------------------------------------------
            """

    return context
        
#Generating AI response
def chat(query,retrievedDocs):
    context= build_context(retrievedDocs)
    
    prompt= RAG_PROMPT.format(
        context= context,
        question=query
    )
    
    response=client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    
    return response.text

def chatbot():
    print("\n======MultiModal RAG Chatbot=======\n")
    print("Type exit to quit\n")
    
    while True:

        query = input("Ask a question: ").strip()

        if query.lower() == "exit":
            print("\n See You Soon!")
            break
        
        documents= retrieval_documents(query)
        retrieval(query,documents)
        
        answer=chat(query,documents)
        
        print("FINAL ANSWER")
        print("=" * 80)
        print(answer)
        
if __name__ == "__main__":
    if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()):
        print("Existing vector database found.")
        db = Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embedding_model
        )
        
    else:
        extract_text()
        extract_images()
        extract_tables()
        describe_image()
        build_knowledge()
        db = create_vector_database()
        
        
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    chatbot()