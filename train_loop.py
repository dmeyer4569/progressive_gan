import torch
from dataloader import get_dataloader
from optim_and_loss import gen, disc, optimizer_g, optimizer_d, criterion, device
from config import Z_DIM, BATCH_SIZE

# Training loop template
def train_gan(epochs=1):
    dataloader = get_dataloader()
    for epoch in range(epochs):
        for real_imgs in dataloader:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            # Train Discriminator
            noise = torch.randn(batch_size, Z_DIM, device=device)
            fake_imgs = gen(noise)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            optimizer_d.zero_grad()
            real_pred = disc(real_imgs)
            fake_pred = disc(fake_imgs.detach())
            loss_real = criterion(real_pred, real_labels)
            loss_fake = criterion(fake_pred, fake_labels)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_pred = disc(fake_imgs)
            loss_g = criterion(fake_pred, real_labels)
            loss_g.backward()
            optimizer_g.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")

if __name__ == "__main__":
    train_gan(epochs=100)
