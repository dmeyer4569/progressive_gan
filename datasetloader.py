
import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import IMG_W, IMG_H

img_w = IMG_W
img_h = IMG_H

class CustomImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transforms.Compose([
            transforms.Lambda(self.pad_and_crop),
            transforms.ToTensor(),
        ])

    def pad_and_crop(self, img):
        # Pad top and bottom to make square (img_hximg_h)
        w, h = img.size
        if w != img_h or h != img_w:
            img = img.resize((img_h, img_w), Image.BICUBIC)
        pad_top = (img_h - img_w) // 2
        pad_bottom = img_h - img_w - pad_top
        img = ImageOps.expand(img, border=(0, pad_top, 0, pad_bottom), fill=0)
        # Center crop to 512x512
        img = transforms.CenterCrop(512)(img)
        return img

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img
