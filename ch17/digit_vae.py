# Convolutional Variational AutoEncoder (MNIST) with "show internals"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# --- Helpers ---


def tinfo(name, t: torch.Tensor, max_items=6):
    flat = t.detach().flatten()
    sample = flat[:max_items].cpu().tolist()
    print(
        f"""
        {name} 
        shape={tuple(t.shape)} dtype={t.dtype} device={t.device} 
        min={t.min().item():.3f} max={t.max().item():.3f} 
        mean={t.mean().item():.3f} std={t.std().item():.3f} 
        sample={sample}
        """
    )


def vae_loss(x_hat, x, mu, logvar):
    # Reconstruction loss (BCE) + KL(q(z|x) || N(0,I))
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl

# --- Convolutional Variational Autoenoder ---


class ConvVAE(nn.Module):
    """
    Encoder: (B,1,28,28) -> Conv -> Conv -> flatten -> (mu, logvar)
    Decoder: z -> FC -> reshape -> ConvTranspose -> ConvTranspose -> (B,1,28,28)
    """

    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder: 28x28 -> 14x14 -> 7x7
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28->14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14->7
            nn.ReLU(inplace=True),
        )
        self.enc_flat_dim = 64*7*7
        self.fc_mu = nn.Linear(self.enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_flat_dim, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, self.enc_flat_dim)
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4,
                               stride=2, padding=1),  # 7->14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4,
                               stride=2, padding=1),  # 14->28
            nn.Sigmoid(),  # pixles in [0,1]
        )

    def encode(self, x):
        h = self.enc_conv(x)            # (B, 62,7,2)
        h_flat = h.view(x.size(0), -1)  # (B, 3136)
        mu = self.fc_mu(h_flat)         # (B, latent_dim)
        logvar = self.fc_logvar(h_flat)  # (B, latent_dim)
        return mu, logvar, h, h_flat

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, std, eps

    def decode(self, z):
        h_flat = F.relu(self.dec_fc(z))         # (B, 3136)
        h = h_flat.view(z.size(0), 64, 7, 7)    # (B, 64, 7, 7)
        x_hat = self.dec_deconv(h)              # (B, 1, 28, 28)
        return x_hat, h, h_flat

    def forward(self, x):
        mu, logvar, h_enc, h_enc_flat, = self.encode(x)
        z, std, eps = self.reparameterize(mu, logvar)
        x_hat, h_dec, h_dec_flat = self.decode(z)
        return {
            "x_hat": x_hat,
            "mu": mu,
            "logvar": logvar,
            "std": std,
            "eps": eps,
            "z": z,
            "h_enc": h_enc,
            "h_enc_flat": h_enc_flat,
            "h_dec": h_dec,
            "h_dec_flat": h_dec_flat,
        }

# --- Run ---


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    latent_dim = 16
    epochs = 10
    lr = 1e-3

    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    vae = ConvVAE(latent_dim=latent_dim).to(device)
    opt = optim.Adam(vae.parameters(), lr=lr)

    # ---- Show internal on a single batch ----
    x0, _ = next(iter(loader))
    x0 = x0.to(device)
    print("\n--- One forward pass (internals) ---")
    vae.eval()
    with torch.no_grad():
        out = vae(x0)

        tinfo("x", x0)
        tinfo("h_enc", out["h_enc"])
        tinfo("h_enc_flat", out["h_enc_flat"])
        tinfo("mu", out["mu"])
        tinfo("logvar", out["logvar"])
        tinfo("std", out["std"])
        tinfo("eps", out["eps"])
        tinfo("z", out["z"])
        tinfo("h_dec", out["h_dec"])
        tinfo("x_hat", out["x_hat"])

        # Save original vs reconstruction
        orig = x0[:16]
        recon = out["x_hat"][:16]
        save_image(orig, "datasets_and_media/convvae_orig.png", nrow=4)
        save_image(recon, "datasets_and_media/convvae_recon.png", nrow=4)
        print("Saved: convvae_orig.png and convvae_recon.png")

    # --- Train ---
    print("\n--- Training ---")
    vae.train()
    for epoch in range(1, epochs + 1):
        total_loss = total_recon = total_kl = 0.0
        for x, _ in loader:
            x = x.to(device)
            out = vae(x)
            loss, recon, kl = vae_loss(
                out["x_hat"], x, out["mu"], out["logvar"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

        n = len(loader.dataset)
        print(
            f"Epoch {epoch}/{epochs} | loss={total_loss/n:.4f} recon={total_recon/n:.4f} kl={total_kl/n:.4f}"
        )

        # sample new digits from the prior
        vae.eval()
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            x_hat, _, _ = vae.decode(z)
            save_image(
                x_hat, f"datasets_and_media/convvae_samples_epoch_{epoch}.png", nrow=4)
            vae.train()

    print("\nDone. Check:")
    print(" - datasets_and_media/convvae_orig.png")
    print(" - datasets_and_media/convvae_recon.png")
    print(" - datasets_and_media/convvae_samples_epoch_*.png")


if __name__ == "__main__":
    main()
