# Copyright (c) 2024, WSO2 LLC. (https://www.wso2.com/) All Rights Reserved.

# WSO2 LLC. licenses this file to you under the Apache License,
# Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.

import base64
import io
import os
import uvicorn
import PyPDF2
import chardet
import docx
import openai
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from tenacity import retry, stop_after_attempt, wait_exponential

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_version = os.getenv('API_VERSION')
azure_deployment = os.getenv('AZURE_DEPLOYMENT_NAME')

# Initialize Azure OpenAI
llm = openai.AzureOpenAI(
    api_key=api_key,
    azure_deployment=azure_deployment,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

supported_content_types = [
    "application/pdf",
    "text/plain",
    "image/jpeg",
    "image/png",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
]

@app.post("/generate_mapping_instruction/")
async def generate_mapping_instruction(file: UploadFile = None, text: str = Form(None)):
    return await process_input(file, text, process_type="mapping_instruction")

@app.post("/generate_record/")
async def generate_record(file: UploadFile = None, text: str = Form(None)):
    return await process_input(file, text, process_type="records")

async def process_input(file: UploadFile, text: str, process_type: str):
    if file:
        if file.content_type not in supported_content_types:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a supported text or image file.")
        return await process_file(file, process_type)
    elif text:
        return await process_text(text, process_type)
    else:
        raise HTTPException(status_code=400, detail="No file or text provided. Please upload a file or provide text input.")

async def process_file(file: UploadFile, process_type: str):
    file_location = f"data/input/{file.filename}"
    os.makedirs(os.path.dirname(file_location), exist_ok=True)

    content = await file.read()
    with open(file_location, "wb") as f:
        f.write(content)

    file_content = ""
    try:
        if file.content_type == "application/pdf":
            with open(file_location, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    page_content = extract_text_and_images_from_pdf_page(page, process_type)
                    file_content += page_content

            file_content = process_with_gpt4(text_content=file_content, content_type=file.content_type,
                                             process_type=process_type)

        elif file.content_type == "text/plain":
            detected_encoding = chardet.detect(content)
            file_content = content.decode(detected_encoding['encoding'])
            file_content = process_with_gpt4(text_content=file_content, content_type=file.content_type,
                                             process_type=process_type)

        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file_location)
            file_content = extract_text_and_images_from_doc_page(doc, process_type)
            file_content = process_with_gpt4(text_content=file_content, content_type=file.content_type,
                                             process_type=process_type)

        elif file.content_type in ["image/jpeg", "image/png"]:
            with open(file_location, "rb") as image_file:
                img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                file_content = process_with_gpt4(base64_content=img_base64, content_type=file.content_type,
                                                 process_type=process_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    return {"file_content": file_content}

async def process_text(text: str, process_type: str):
    try:
        file_content = process_with_gpt4(text_content=text, content_type="text/plain", process_type=process_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

    return {"file_content": file_content}

def extract_text_and_images_from_pdf_page(page, process_type):
    content = ""
    text_lines = []
    if text := page.extract_text():
        text_lines = text.splitlines()

    xobjects = page['/Resources'].get('/XObject', {})
    image_keys = list(xobjects.keys())
    image_index = 0

    for line in text_lines:
        content += line + "\n"
        if image_index < len(image_keys):
            obj_key = image_keys[image_index]
            vobj = xobjects[obj_key].get_object()

            if vobj['/Subtype'] == '/Image':
                img_format, img = None, None
                if vobj['/Filter'] == '/FlateDecode':
                    buf = vobj.get_data()
                    size = (vobj['/Width'], vobj['/Height'])
                    img = Image.frombytes('RGB', size, buf)
                    img_format = 'PNG'
                elif vobj['/Filter'] == '/DCTDecode':
                    buf = vobj.get_data()
                    img = Image.open(io.BytesIO(buf))
                    img_format = 'JPEG'

                if img:
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format=img_format)
                    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    img_summary = summarize_content_with_gpt4(base64_content=img_base64,
                                                              content_type=f"image/{img_format.lower()}",
                                                              process_type=f"{process_type}")
                    content += f"\n{img_summary}\n"
                    image_index += 1

    return content

def extract_text_and_images_from_doc_page(doc, process_type):
    file_content = ""
    for para in doc.paragraphs:
        file_content += para.text + "\n"
        if para._element.xpath(".//a:blip"):
            for rel in para._element.xpath(".//a:blip"):
                img_rid = rel.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                img_bytes = doc.part.related_parts[img_rid].blob

                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                img_format = Image.open(io.BytesIO(img_bytes)).format.lower()
                img_summary = summarize_content_with_gpt4(base64_content=img_base64, content_type=f"image/{img_format}",
                                                          process_type=f"{process_type}")

                file_content += f"{img_summary}\n"
    return file_content

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=1, max=5), reraise=True)
def process_with_gpt4(text_content: str = None, base64_content: str = None, content_type: str = None,
                      process_type: str = None) -> str:
    try:
        prompt_file_path = os.path.join("utils", f"{process_type}_extraction.txt")
        with open(prompt_file_path, 'r') as file:
            prompt_text = file.read()

        if content_type in ["image/jpeg", "image/png"]:
            content_input = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{content_type};base64,{base64_content}"
                        }
                    }
                ]
            }
        else:
            content_input = {
                "role": "user",
                "content": prompt_text + "\n\n" + text_content
            }
        response = llm.chat.completions.create(model="gpt-4", messages=[content_input], max_tokens=1000)
        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Error processing with GPT-4: {str(e)}")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=1, max=5), reraise=True)
def summarize_content_with_gpt4(base64_content: str = None, content_type: str = None, process_type: str = None) -> str:
    try:
        prompt_file_path = os.path.join("utils", f"{process_type}_image_summarizer.txt")
        with open(prompt_file_path, 'r') as file:
            summarize_text = file.read()

        content_input = {
            "role": "user",
            "content": [
                {"type": "text", "text": summarize_text},
                {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{base64_content}"}}
            ]
        }

        response = llm.chat.completions.create(model="gpt-4", messages=[content_input], max_tokens=1000)
        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Error summarizing image content with GPT-4: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
