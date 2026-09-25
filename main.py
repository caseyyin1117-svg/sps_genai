from fastapi import FastAPI
from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding

app = FastAPI()

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox is very quick.",
    "The lazy dog sleeps all day.",
    "The fox jumps over the dog.",
]

bigram_model = BigramModel(corpus)


@app.get("/")
def read_root():
    return {"message": "Bigram API is running!"}


@app.get("/generate")
def generate_text(start_word: str = "the", length: int = 20):
    generated_text = bigram_model.generate_text(
        start_word=start_word,
        length=length
    )

    return {
        "start_word": start_word,
        "length": length,
        "generated_text": generated_text,
    }
    
    
    
@app.get("/embedding")
def get_embedding(word: str):
    embedding = calculate_embedding(word)

    return {
        "word": word,
        "embedding": embedding
    }