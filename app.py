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
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from tenacity import retry, stop_after_attempt, wait_exponential
import pillow_heif
import anthropic
import re
import subprocess
from docx import Document
from reportlab.pdfgen import canvas

app = FastAPI(
    title="File Upload - Python REST API",
    description="API for processing text, PDFs, images, and Word documents to generate mapping instructions or records",
    version="0.1.0",
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
)

# Changing the default openapi version to 3.0.2 as Choreo does not support the default version
app.openapi_version = "3.0.2"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

supported_content_types = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/heic",
    "image/heif",
    "text/plain"
]

@app.post("/file_upload/generate_mapping_instruction", operation_id="File_upload_generate_mapping_instruction_post", tags=["generate-mapping-instruction"])
async def generate_mapping_instruction(file: UploadFile = None, text: str = Form(None)):
    return await process_input(file, text, process_type="mapping_instruction")

@app.post("/file_upload/generate_record", operation_id="File_upload_generate_record_post", tags=["generate-record"])
async def generate_record(file: UploadFile = None, text: str = Form(None)):
    return await process_input(file, text, process_type="records")

async def process_input(file: UploadFile, text: str, process_type: str):
    if file:
        if file.content_type not in supported_content_types:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a supported text or image file.")
        return await process_file(file, process_type)
    elif text:
        message = await process_text(text, process_type)
        file_content = extract_ballerina_code(message, process_type)
        return {"file_content": file_content}
    else:
        raise HTTPException(status_code=400, detail="No file or text provided. Please upload a file or provide text input.")

# Converts input file to PDF using unoconv
async def convert_to_pdf(input_file, output_file=None):
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + ".pdf"
    try:
        subprocess.run(["unoconv", "-f", "pdf", "-o", output_file, input_file], check=True)
        return output_file
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error during conversion: {str(e)}")

# Extracts Ballerina code or mapping fields from the message.
def extract_ballerina_code(message, process_type):
    if process_type == "records":
        ballerina_code = re.search(r"<ballerina_code>(.*?)</ballerina_code>", str(message), re.DOTALL)
        if ballerina_code:
            raw_content = ballerina_code.group(1)
            return raw_content.encode("utf-8").decode("unicode_escape").strip()
        else:
            print("No Ballerina code found.")
    else:
        mapping_fields = re.search(r"<mapping_fields>(.*?)</mapping_fields>", str(message), re.DOTALL)
        if mapping_fields:
            raw_content = mapping_fields.group(1)
            return raw_content.encode("utf-8").decode("unicode_escape").strip()
        else:
            print("No mapping fields found.")
    return ""

# Convert DOCX to PDF
async def docx_to_pdf(input_path, output_path):
    document = Document(input_path)
    pdf = canvas.Canvas(output_path)
    x, y = 100, 800
    line_height = 20

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            pdf.drawString(x, y, text)
            y -= line_height
            if y < 50:
                pdf.showPage()
                y = 800
    pdf.save()

# Process PDF file
async def process_pdf(file_location, process_type):
    try:
        with open(file_location, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')
        return extraction_using_anthropic(pdf_data=pdf_data, process_type=process_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

# Process DOCX file
async def process_word(file_location, process_type):
    pdf_path = file_location.replace(".docx", ".pdf")
    try:
        await docx_to_pdf(file_location, pdf_path)
        return await process_pdf(pdf_path, process_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX processing error: {str(e)}")

# Converts image file to PDF and processes
async def process_image(file_location, process_type, content_type):
    if content_type in ["image/heic", "image/heif"]:
        pillow_heif.register_heif_opener()
        with open(file_location, "rb") as heic_file:
            image = Image.open(io.BytesIO(heic_file.read()))
            converted_path = file_location.replace(".heic", ".jpg").replace(".heif", ".jpg")
            image.convert('RGB').save(converted_path, format="JPEG")
            file_location = converted_path
            content_type = "image/jpeg"

    with open(file_location, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        file_content = img_extraction_using_anthropic(img_data=img_base64, process_type=process_type, content_type=content_type)

    return file_content

# Saves uploaded file to disk
async def save_file(file: UploadFile, file_location: str):
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    content = await file.read()
    with open(file_location, "wb") as f:
        f.write(content)

async def process_text(text:str, process_type: str):
    try:
        file_content = text_extraction_using_anthropic(text_content=text, process_type=process_type)
        print(file_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

    return {"file_content": file_content}

async def process_file(file: UploadFile, process_type: str):
    file_location = f"data/input/{file.filename}"
    await save_file(file, file_location)

    try:
        if file.content_type == "application/pdf":
            message = await process_pdf(file_location=file_location, process_type=process_type)
        elif file.content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            message = await process_word(file_location=file_location, process_type=process_type)
        elif file.content_type in ["image/jpeg", "image/png", "image/heif", "image/heic"]:
            message = await process_image(file_location=file_location, process_type=process_type, content_type=file.content_type)
        elif file.content_type in ["text/plain"]:
            with open(file_location, "r") as file:
                content = file.read()
            message = await process_text(text=content, process_type=process_type)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        file_content = extract_ballerina_code(message, process_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    return {"file_content": file_content}

@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def extraction_using_anthropic(pdf_data, process_type):
    try:
        prompt_file_path = os.path.join("utils", f"{process_type}_extraction.txt")
        with open(prompt_file_path, 'r') as file:
            prompt_text = file.read()

        client = anthropic.Anthropic()
        message = client.beta.messages.create(
            model="claude-3-5-sonnet-20241022",
            betas=["pdfs-2024-09-25"],
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ],
        )

        return message

    except Exception as e:
        raise Exception(f"Error processing with Claude: {str(e)}")

@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def img_extraction_using_anthropic(img_data, process_type, content_type):
    try:
        prompt_file_path = os.path.join("utils", f"{process_type}_extraction.txt")
        with open(prompt_file_path, 'r') as file:
            prompt_text = file.read()

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content_type,
                                "data": img_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ],
                }
            ],
        )
        return message

    except Exception as e:
        raise Exception(f"Error processing with Claude: {str(e)}")

@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def text_extraction_using_anthropic(text_content, process_type):
    try:
        prompt_file_path = os.path.join("utils", f"{process_type}_extraction.txt")
        with open(prompt_file_path, 'r') as file:
            prompt_text = file.read()

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text + "\n\n" + text_content
                }
            ],
        )
        return message

    except Exception as e:
        raise Exception(f"Error processing with Claude: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)