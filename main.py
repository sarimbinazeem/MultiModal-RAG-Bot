# Imports

import os
from pathlib import Path
import json

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
MODEL= "gemini-2.5-flash"


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
            id=image[0] #it returns a page ID of that image
            base_image= document.extract_image(id)
            
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
            tables= page.extract_table() 
            
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
    
    output_path = EXTRACTED_DIR/ "image_description.json"
    
    with open(output_path,"w",encoding="utf-8" ) as file:
        json.dump(descriptions,file,indent=4,ensure_ascii=False) 
        
    print("Image descriptions saved successfully.")
    
def build_knowledge():
    print("\nBuilding knowledge base...")
    
    #Text
    for file in TEXT_DIR.iterdir():
        
        if file.suffix != ".txt":
            continue
        
        with open(file,"r",encoding="utf-8") as f:
            documents.append(
                "content":f.read(),
                "metadata": {
                    "type":"text",

                    "source":file.name

                }
            )
            
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