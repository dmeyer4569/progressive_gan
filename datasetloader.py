
import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transforms.Compose([
            transforms.Lambda(self.pad_and_crop),
            transforms.ToTensor(),
        ])

    def pad_and_crop(self, img):
        # Pad top and bottom to make square (1280x1280)
        w, h = img.size
        if w != 1280 or h != 720:
            img = img.resize((1280, 720), Image.BICUBIC)
        pad_top = (1280 - 720) // 2
        pad_bottom = 1280 - 720 - pad_top
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
