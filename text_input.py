from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import PyPDF2
import docx
import chardet
import base64
import openai
from dotenv import load_dotenv
from PIL import Image
import io
from tenacity import retry, stop_after_attempt, wait_exponential

app = FastAPI()

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

class TextInput(BaseModel):
    content: str

@app.post("/upload_or_text/")
async def upload_or_text(file: UploadFile = None, text_content: str = Form(None)):
    """
    Handle either file upload or text input, but not both.
    """
    if file and text_content:
        raise HTTPException(status_code=400, detail="Provide either a file or text content, not both.")

    if file is None and text_content is None:
        raise HTTPException(status_code=400, detail="No file or text content provided.")

    file_content = ""

    try:
        if file:
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

            if file.content_type == "application/pdf":
                # Extract text and images from PDF
                with open(file_location, "rb") as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)

                    for page in reader.pages:
                        page_content = extract_text_and_images_from_pdf_page(page)
                        file_content += page_content

            elif file.content_type == "text/plain":
                # Detect encoding and read text file content
                detected_encoding = chardet.detect(content)
                file_content = content.decode(detected_encoding['encoding'])

            elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # Extract text and images from .docx file
                doc = docx.Document(file_location)
                file_content = extract_text_and_images_from_doc_page(doc)

            elif file.content_type in ["image/jpeg", "image/png"]:
                # Convert image to base64
                with open(file_location, "rb") as image_file:
                    img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    img_summary = summarize_image_content_with_gpt4(base64_content=img_base64,
                                                                    content_type=file.content_type)
                    file_content = summarize_content_with_gpt4(text_content=img_summary)

            # Summarize the entire content
            summarized_content = summarize_content_with_gpt4(text_content=file_content, content_type=file.content_type)

        elif text_content:
            # Summarize raw text input
            summarized_content = summarize_content_with_gpt4(text_content=text_content, content_type="text/plain")

        return JSONResponse(content={
            "file_name": file.filename if file else None,
            "content_type": file.content_type if file else "text/plain",
            "file_content": file_content if file else None,
            "original_content": text_content if text_content else file_content,
            "summarized_content": summarized_content
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing content: {str(e)}")


# Extract text and images from PDF page
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

                    img_summary = summarize_image_content_with_gpt4(base64_content=img_base64,
                                                                    content_type=content_type)
                    content += f"\n{img_summary}\n"
                    image_index += 1

    return content

# Extract text and images from DOCX
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

                img_summary = summarize_image_content_with_gpt4(base64_content=img_base64,
                                                                content_type=f"image/{img_format}")

                # Insert image summary after the paragraph text
                file_content += f"{img_summary}\n"

    return file_content

# Retry decorator with exponential backoff for content summarization
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=1, max=5), reraise=True)
def summarize_content_with_gpt4(text_content: str = None, base64_content: str = None, content_type: str = None) -> str:
    try:
        prompt_file_path = os.path.join("utils", "records_extraction.txt")
        with open(prompt_file_path, 'r') as file:
            prompt_text = file.read()

        # Prepare the input for GPT-4 based on the content type
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


# Retry decorator with exponential backoff for image summarization
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=1, max=5), reraise=True)
def summarize_image_content_with_gpt4(base64_content: str = None, content_type: str = None) -> str:
    try:
        prompt_file_path = os.path.join("utils", "records_image_summarizer.txt")
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
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error summarizing image content: {str(e)}")
