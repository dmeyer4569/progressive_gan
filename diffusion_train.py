import torch
from torch.utils.data import DataLoader
from datasetloader import CustomImageDataset
from config import *
from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator
import os
from datetime import datetime
from PIL import Image
import torchvision.transforms as T

# Setup run directory
now = datetime.now().strftime("%Y%m%d_%H%M%S")

run_dir = os.path.join("runs", now)
outputs_dir = os.path.join(run_dir, "outputs")
downscaled_dir = os.path.join(run_dir, "downscaled")
epochs_dir = os.path.join(run_dir, "epochs")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(downscaled_dir, exist_ok=True)
os.makedirs(epochs_dir, exist_ok=True)

# Save config
import shutil
shutil.copy("config.py", os.path.join(run_dir, "config.py"))

# Data

# Use smaller image size for lower VRAM usage
IMG_SIZE = IMG_SIZE
transform = T.Compose([
    T.CenterCrop(IMG_SIZE),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])
class CustomImageDatasetSmall(CustomImageDataset):
    def __init__(self, image_dir, max_images=None):
        super().__init__(image_dir)
        if max_images is not None:
            import random
            random.shuffle(self.image_files)
            self.image_files = self.image_files[:max_images]
    def __getitem__(self, idx):
        img = super().__getitem__(idx)
        img = T.Resize((IMG_SIZE, IMG_SIZE))(img)
        return img

# Use only 5000 images, for example:
dataset = CustomImageDatasetSmall(DATASET, max_images=MAX_IMAGES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# Model
model = UNet2DModel(
    sample_size=IMG_SIZE,  # image size
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),  # Smaller model for less VRAM
)
scheduler = DDPMScheduler(num_train_timesteps=1000)

accelerator = Accelerator(mixed_precision='fp16')
model, dataloader = accelerator.prepare(model, dataloader)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Training loop (simplified, 1 epoch)
from tqdm import tqdm
for epoch in range(EPOCHS):
    for step, real_imgs in enumerate(tqdm(dataloader)):
        real_imgs = real_imgs.to(accelerator.device)
        noise = torch.randn_like(real_imgs)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (real_imgs.size(0),), device=real_imgs.device).long()
        noisy_imgs = scheduler.add_noise(real_imgs, noise, timesteps)
        noise_pred = model(noisy_imgs, timesteps).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
    # Save model
    accelerator.save_state(os.path.join(epochs_dir, f"model_epoch_{epoch}.pt"))
    # Sample and save images
    model.eval()
    with torch.no_grad():
        sample = torch.randn(4, 3, IMG_SIZE, IMG_SIZE, device=accelerator.device)
        for t in reversed(range(scheduler.num_train_timesteps)):
            timesteps = torch.full((4,), t, device=accelerator.device, dtype=torch.long)
            with torch.no_grad():
                noise_pred = model(sample, timesteps).sample
            sample = scheduler.step(noise_pred, t, sample).prev_sample
        sample = (sample.clamp(-1, 1) + 1) / 2
        for i, img_tensor in enumerate(sample):
            img = T.ToPILImage()(img_tensor.cpu())
            img.save(os.path.join(run_dir, "outputs", f"sample_{epoch}_{i}.png"))
            img.resize((64, 64), resample=Image.BICUBIC).save(os.path.join(run_dir, "downscaled", f"downscaled_{epoch}_{i}.png"))
    model.train()
