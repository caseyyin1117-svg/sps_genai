import torch
from torchvision.utils import save_image

from app.gan_model import Generator


def generate_samples(num_samples=16, z_dim=100):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    generator = Generator(z_dim).to(device)

    generator.load_state_dict(
        torch.load(
            "models/gan_generator.pth",
            map_location=device,
            weights_only=True
        )
    )

    generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, z_dim, device=device)
        fake_images = generator(noise)

    save_image(
        fake_images,
        "models/gan_samples.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )

    print("Generated images saved to models/gan_samples.png")


if __name__ == "__main__":
    generate_samples()