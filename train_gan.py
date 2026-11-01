import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.gan_model import Generator, Discriminator


def train_gan(epochs=20, batch_size=64, z_dim=100):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    generator = Generator(z_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    for epoch in range(epochs):
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            real_labels = torch.ones(
                current_batch_size, 1, device=device
            )
            fake_labels = torch.zeros(
                current_batch_size, 1, device=device
            )

            noise = torch.randn(
                current_batch_size, z_dim, device=device
            )
            fake_images = generator(noise)

            real_output = discriminator(real_images)
            fake_output = discriminator(fake_images.detach())

            loss_d = (
                criterion(real_output, real_labels)
                + criterion(fake_output, fake_labels)
            )

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            noise = torch.randn(
                current_batch_size, z_dim, device=device
            )
            fake_images = generator(noise)

            output = discriminator(fake_images)
            loss_g = criterion(output, real_labels)

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Discriminator Loss: {loss_d.item():.4f}, "
            f"Generator Loss: {loss_g.item():.4f}"
        )

    os.makedirs("models", exist_ok=True)

    torch.save(
        generator.state_dict(),
        "models/gan_generator.pth"
    )

    print("Generator saved to models/gan_generator.pth")


if __name__ == "__main__":
    train_gan()