import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from config import IMG_SIZE

class CustomImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        self.transform = T.Compose([
            T.CenterCrop(IMG_SIZE),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> output in [-1, 1]
        ])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img
