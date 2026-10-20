# Generative Adversarial Network
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")
z_dim = 64
batch_size = 128

# --- Data ---
loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    ),
    batch_size=batch_size,
    shuffle=True,
)

# --- Generator ---
G = nn.Sequential(
    nn.Linear(z_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 784),
    nn.Tanh()
).to(device)

# --- Discriminator ---
D = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
).to(device)

loss_fn = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=1e-3)
opt_D = optim.Adam(D.parameters(), lr=1e-3)

# --- Training ---
for epoch in range(12):
    for real, _ in loader:
        real = real.view(-1, 784).to(device)
        b = real.size(0)

        # labels
        ones = torch.ones(b, 1, device=device)
        zeros = torch.zeros(b, 1, device=device)

        # ---- Train Discriminator ----
        z = torch.randn(b, z_dim, device=device)
        fake = G(z)

        loss_D = loss_fn(D(real), ones) + loss_fn(D(fake.detach()), zeros)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ---- Train Generator ----
        z = torch.randn(b, z_dim, device=device)
        fake = G(z)
        loss_G = loss_fn(D(fake), ones)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch+1} | D: {loss_D.item():.3f} | G: {loss_G.item():.3f}")

    # --- Save one sample ---
    with torch.no_grad():
        z = torch.randn(16, z_dim, device=device)
        imgs = G(z).view(-1, 1, 28, 28)
        file = f"datasets_and_media/simple_gan_epoch{epoch+1}.png"
        save_image(imgs, file,
                   nrow=4, normalize=True)

    print(f"Saved {file}")
