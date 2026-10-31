from fastapi import FastAPI
from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding
import io
import torch
from app.gan_model import Generator
from fastapi.responses import Response
from torchvision.utils import make_grid


from PIL import Image
from fastapi import UploadFile, File
from torchvision import transforms

from app.cnn_model import CNN

app = FastAPI()

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

cnn_model = CNN()

cnn_model.load_state_dict(
    torch.load(
        "models/cnn_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

cnn_model.eval()

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
    
    
    
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]

    return {
        "predicted_class": predicted_class
    }
    
gan_generator = Generator()

gan_generator.load_state_dict(
    torch.load(
        "models/gan_generator.pth",
        map_location="cpu",
        weights_only=True
    )
)

gan_generator.eval()


@app.get("/generate-image")
def generate_image():

    noise = torch.randn(16, 100)

    with torch.no_grad():
        fake_images = gan_generator(noise)

    grid = make_grid(
        fake_images,
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )

    image_array = (
        grid.mul(255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )

    image = Image.fromarray(image_array)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )