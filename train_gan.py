import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.gan_model import Generator, Critic


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
    critic = Critic().to(device)

    lr = 5e-5
    n_critic = 5
    clip_value = 0.01

    optimizer_g = optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_c = optim.RMSprop(critic.parameters(), lr=lr)

    for epoch in range(epochs):
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            for _ in range(n_critic):
                noise = torch.randn(
                    current_batch_size, z_dim, 1, 1, device=device
                )
                fake_images = generator(noise).detach()

                critic_real = critic(real_images).mean()
                critic_fake = critic(fake_images).mean()

                loss_c = -(critic_real - critic_fake)

                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()

                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            noise = torch.randn(
                current_batch_size, z_dim, 1, 1, device=device
            )

            fake_images = generator(noise)
            loss_g = -critic(fake_images).mean()

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Critic Loss: {loss_c.item():.4f}, "
            f"Generator Loss: {loss_g.item():.4f}"
        )

    torch.save(generator.state_dict(), "models/gan_generator.pth")
    print("Generator saved to models/gan_generator.pth")


if __name__ == "__main__":
    train_gan()