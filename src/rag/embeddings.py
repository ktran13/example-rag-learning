import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts, model: str):
    # Returns list of vectors
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]