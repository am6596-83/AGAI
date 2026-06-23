## AGAI_Assignments 1 & 2
## Author: Amy Moffatt

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
# Imports for Assignment 2
from fastapi import FastAPI, UploadFile, File
from app.image_classifier import classify_image

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):      # a new request model that accepts a word parameter
    word: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")      # takes a word, calls get_embedding() from BigramModel, and returns the word along with its embedding vector
def get_embedding(request: EmbeddingRequest):
    embedding = bigram_model.get_embedding(request.word)
    return {"word": request.word, "embedding": embedding}

# New endpoint for Assignment 2
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class = classify_image(image_bytes)
    return {"predicted_class": predicted_class}