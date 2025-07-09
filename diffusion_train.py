import torch
import shutil
from torch.utils.data import DataLoader
from datasetloader import CustomImageDataset
from config import *
from diffusers import UNet2DModel, DDIMScheduler, EMAModel
from accelerate import Accelerator
import os
from datetime import datetime
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import pandas as pd
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# Setup run directory
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join("runs", now)
outputs_dir = os.path.join(run_dir, "outputs")
downscaled_dir = os.path.join(run_dir, "downscaled")
epochs_dir = os.path.join(run_dir, "epochs")
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(downscaled_dir, exist_ok=True)
os.makedirs(epochs_dir, exist_ok=True)

# Save config
shutil.copy("config.py", os.path.join(run_dir, "config.py"))

# Transform with normalization
transform = T.Compose([
    T.CenterCrop(IMG_SIZE),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),  # Normalize to [-1, 1]
])

class CustomImageDatasetSmall(CustomImageDataset):
    def __init__(self, image_dir, max_images=None):
        if not os.path.isdir(image_dir):
            raise ValueError(f"Image directory {image_dir} not found.")
        super().__init__(image_dir)
        if max_images is not None:
            import random
            random.shuffle(self.image_files)
            self.image_files = self.image_files[:max_images]

dataset = CustomImageDatasetSmall(DATASET, max_images=MAX_IMAGES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# Model
model = UNet2DModel(
    sample_size=IMG_SIZE,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
)

scheduler = DDIMScheduler(num_train_timesteps=250, beta_schedule="squaredcos_cap_v2")

# Accelerator + EMA
accelerator = Accelerator(mixed_precision='bf16')
model, dataloader = accelerator.prepare(model, dataloader)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
ema = EMAModel(parameters=model.parameters(), power=0.75)  # power=0.75 is typical

loss_log = []
metrics_csv = os.path.join(run_dir, "metrics.csv")

# ---- Resume from checkpoint ----
start_epoch = 0
if CHECKPOINT_PT and os.path.exists(CHECKPOINT_PT):
    print(f"Loading checkpoint from {CHECKPOINT_PT}")
    checkpoint = torch.load(CHECKPOINT_PT, map_location=accelerator.device)

    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    start_epoch = checkpoint.get("epoch", 0)
    print(f"Resuming training from epoch {start_epoch}")
# ---------------------------------

# Training Loop
model.train()
for epoch in range(start_epoch, EPOCHS):
    epoch_losses = []
    for step, real_imgs in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        real_imgs = real_imgs.to(accelerator.device)
        noise = torch.randn_like(real_imgs)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (real_imgs.size(0),), device=real_imgs.device).long()

        noisy_imgs = scheduler.add_noise(real_imgs, noise, timesteps)
        noise_pred = model(noisy_imgs, timesteps).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        ema.step(model.parameters())

        epoch_losses.append(loss.item())

        if step % 100 == 0:
            tqdm.write(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

    loss_log.append((epoch, np.mean(epoch_losses)))

    # Save model checkpoint
    checkpoint_path = os.path.join(epochs_dir, f"model_epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
    }, checkpoint_path)

    # Save EMA weights temporarily
    ema.store(model.parameters())
    ema.copy_to(model.parameters())

    # Sampling
    model.eval()
    scheduler.set_timesteps(50)  # inference steps
    with torch.no_grad():
        sample = torch.randn(4, 3, IMG_SIZE, IMG_SIZE, device=accelerator.device)
        for t in scheduler.timesteps:
            timestep_tensor = torch.full((sample.shape[0],), t, device=accelerator.device, dtype=torch.long)
            noise_pred = model(sample, timestep_tensor).sample
            sample = scheduler.step(noise_pred, t, sample).prev_sample

        # Denormalize
        sample = (sample * 0.5 + 0.5).clamp(0, 1)

        # Save individual images
        for i, img_tensor in enumerate(sample):
            img = T.ToPILImage()(img_tensor.cpu())
            img.save(os.path.join(outputs_dir, f"sample_{epoch}_{i}.png"))
            img.resize((64, 64), resample=Image.BICUBIC).save(os.path.join(downscaled_dir, f"downscaled_{epoch}_{i}.png"))

        # Save grid
        save_image(sample, os.path.join(outputs_dir, f"grid_sample_{epoch}.png"), nrow=2, normalize=True)

        # Compute metrics using one sample for quick feedback
        ref_img = real_imgs[0].detach().cpu()
        gen_img = sample[0].detach().cpu()
        ref_img = (ref_img * 0.5 + 0.5).clamp(0, 1)  # denorm

        # Convert to HWC
        ref_np = ref_img.permute(1, 2, 0).numpy()
        gen_np = gen_img.permute(1, 2, 0).numpy()

        # Compute PSNR and SSIM
        psnr = psnr_metric(ref_np, gen_np, data_range=1.0)
        try:
            ssim = ssim_metric(ref_np, gen_np, data_range=1.0, channel_axis=-1)
        except ValueError as e:
            print(f"[Warning] SSIM computation failed at epoch {epoch}: {e}")
            ssim = float('nan')

        # Log metrics
        pd.DataFrame([[epoch, np.mean(epoch_losses), psnr, ssim]], columns=["epoch", "loss", "psnr", "ssim"]).to_csv(
            metrics_csv, mode='a', index=False, header=not os.path.exists(metrics_csv)
        )

    model.train()
    ema.restore(model.parameters())
    print(f"Epoch {epoch} completed. Loss: {np.mean(epoch_losses):.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")    