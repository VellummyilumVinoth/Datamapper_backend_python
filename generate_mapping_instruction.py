import base64
import io
import os

import PyPDF2
import chardet
import docx
import openai
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from tenacity import retry, stop_after_attempt, wait_exponential

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify the frontend domain here
    allow_credentials=True,
    allow_methods=["*"],  # Or specify the methods you want to allow
    allow_headers=["*"],  # Or specify the headers you want to allow
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


@app.post("/generate_mapping_instruction/")
async def generate_mapping_instruction(file: UploadFile):
    supported_content_types = [
        "application/pdf",
        "text/plain",
        "image/jpeg",
        "image/png",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]

    if file.content_type not in supported_content_types:
        raise HTTPException(status_code=400,
                            detail="Unsupported file type. Please upload a supported text or image file.")

    file_location = f"data/input/{file.filename}"
    os.makedirs(os.path.dirname(file_location), exist_ok=True)

    # Read the content of the file once
    content = await file.read()

    with open(file_location, "wb") as f:
        f.write(content)  # Save the file content

    file_content = ""
    try:
        if file.content_type == "application/pdf":
            # Extract text and images from PDF
            with open(file_location, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)

                for page in reader.pages:
                    page_content = extract_text_and_images_from_pdf_page(page)
                    file_content += page_content

            # Summarize the entire content including text and inline image summaries
            file_content = mapping_instruction_extraction_with_gpt4(text_content=file_content,
                                                                    content_type=file.content_type)
            print(file_content)

        elif file.content_type == "text/plain":
            # Detect encoding and read text file content
            detected_encoding = chardet.detect(content)
            file_content = content.decode(detected_encoding['encoding'])
            file_content = mapping_instruction_extraction_with_gpt4(text_content=file_content,
                                                                    content_type=file.content_type)
            print(file_content)

        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Extract text and images from .docx file
            doc = docx.Document(file_location)
            file_content = extract_text_and_images_from_doc_page(doc)

            # Summarize the entire content including text and inline image summaries
            file_content = mapping_instruction_extraction_with_gpt4(text_content=file_content,
                                                                    content_type=file.content_type)
            print(file_content)

        if file.content_type in ["image/jpeg", "image/png"]:
            # Convert image to base64
            with open(file_location, "rb") as image_file:
                img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                img_summary = summarize_mapping_instruction_with_gpt4(base64_content=img_base64,
                                                                      content_type=file.content_type)
                file_content = mapping_instruction_extraction_with_gpt4(text_content=img_summary)
                print(file_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    return {
        "file_content": file_content,
    }


def extract_text_and_images_from_pdf_page(page):
    content = ""

    # Extract text and maintain the sequence
    text_lines = []
    if text := page.extract_text():
        text_lines = text.splitlines()

    xobjects = page['/Resources'].get('/XObject', {})
    image_keys = list(xobjects.keys())
    image_index = 0

    for line in text_lines:
        # Append text to content
        content += line + "\n"

        # Check if an image should be inserted after this line
        if image_index < len(image_keys):
            obj_key = image_keys[image_index]
            vobj = xobjects[obj_key].get_object()

            if vobj['/Subtype'] == '/Image' and '/Filter' in vobj:
                image_format, img = None, None
                if vobj['/Filter'] == '/FlateDecode':
                    buf = vobj.get_data()
                    size = (vobj['/Width'], vobj['/Height'])
                    image_format = 'PNG'
                    img = Image.frombytes('RGB', size=size, data=buf)
                elif vobj['/Filter'] == '/DCTDecode':
                    buf = vobj.get_data()
                    image_format = 'JPEG'
                    img = Image.open(io.BytesIO(buf))

                if img:
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format=image_format)
                    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

                    content_type = "image/jpeg" if image_format == 'JPEG' else "image/png"

                    img_summary = summarize_mapping_instruction_with_gpt4(base64_content=img_base64,
                                                                          content_type=content_type)
                    content += f"\n{img_summary}\n"
                    image_index += 1

    return content


def extract_text_and_images_from_doc_page(doc):
    file_content = ""
    for para in doc.paragraphs:
        # Add text from the paragraph
        file_content += para.text + "\n"

        # Check for inline shapes (images) within the paragraph
        if para._element.xpath(".//a:blip"):
            for rel in para._element.xpath(".//a:blip"):
                img_rid = rel.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                img_bytes = doc.part.related_parts[img_rid].blob

                img_stream = io.BytesIO(img_bytes)
                img = Image.open(img_stream)
                img_format = img.format.lower()

                img_base64 = base64.b64encode(img_bytes).decode('utf-8')

                img_summary = summarize_mapping_instruction_with_gpt4(base64_content=img_base64,
                                                                      content_type=f"image/{img_format}")

                # Insert image summary after the paragraph text
                file_content += f"{img_summary}\n"

    return file_content


# Retry decorator with exponential backoff
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=1, max=5), reraise=True)
def mapping_instruction_extraction_with_gpt4(text_content: str = None, base64_content: str = None,
                                             content_type: str = None) -> str:
    try:
        prompt_file_path = os.path.join("utils", "mapping_instruction_extraction.txt")
        with open(prompt_file_path, 'r') as file:
            prompt_text = file.read()

        # Determine the content type and prepare the input accordingly
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

        response = llm.chat.completions.create(
            model="gpt-4",
            messages=[content_input],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error summarizing content: {str(e)}")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=1, max=5), reraise=True)
def summarize_mapping_instruction_with_gpt4(base64_content: str = None, content_type: str = None) -> str:
    try:
        prompt_file_path = os.path.join("utils", "mapping_instruction_image_summarizer.txt")
        with open(prompt_file_path, 'r') as file:
            summarize_text = file.read()

        content_input = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": summarize_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{content_type};base64,{base64_content}"
                    }
                }
            ]
        }

        response = llm.chat.completions.create(
            model="gpt-4",
            messages=[content_input],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error summarizing content: {str(e)}")
