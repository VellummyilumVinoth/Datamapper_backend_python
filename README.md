# Context upload for auto datamapping 
Upload the files/contents using the datamapper UI. Use that python backend we retrieve datamappings, record definitons and function signature.
The source code is maintained in this repository.

### Steps to setup development environment

1. Clone the repository and navigate to the backend directory
2. Install Python 3.8 or higher ( 3.10 is recommended )
3. Create a virtual environment and activate it
```bash
python -m venv .venv
source .venv/bin/activate
```

4. Install the dependencies
```bash 
pip install -r requirements.txt
```

```
# Following libraries are used in the project
fastapi
uvicorn
python-dotenv
pypdf2
chardet
python-docx
openai
pillow
tenacity
python-multipart
```

5. Add the environment variables to a .env file in the backend directory
```bash
"AZURE_OPENAI_API_KEY=<your_openai_api_key>"
"AZURE_OPENAI_ENDPOINT=<your_openai_endpoint>"
"API_VERSION=<your_openai_api_version>"
"AZURE_DEPLOYMENT_NAME=<your_openai_deployment_name>"
```

6. Start Copilot server
* With Uvicorn single worker (for development)
```bash
uvicorn main:app --port 8000 --reload 
```

### Production deployment
Copilot server can be easily deployed to WSO2 choreo with Asgardeo as IDP

### Docs
* OpenAPI - http://127.0.0.1:8000/docs
* Redoc - http://127.0.0.1:8000/redoc
* OpenAPI JSON - http://127.0.0.1:8000/openapi.json
* OpenAPI YAML - http://127.0.0.1:8000/openapi.yaml
