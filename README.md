# SPS Applied Generative AI - Assignment 1

## Repository Structure

### Part 1: FastAPI Implementation
- `main.py` - FastAPI application entry point with endpoints for text generation and word embeddings
- `app/bigram_model.py` - Bigram model and word embedding logic using spaCy
- `app/__init__.py` - Marks the app directory as a Python package

### Part 2: Probability Theory
- `Assignment1_Part2.ipynb` - Jupyter notebook with solutions to probability questions

## Running the API
```bash
uv run fastapi dev main.py
```
Access the API at: http://127.0.0.1:8000
Interactive docs at: http://127.0.0.1:8000/docs