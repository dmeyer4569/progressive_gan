import torch
from torch.utils.data import DataLoader
from datasetloader import CustomImageDataset
from config import DATASET, BATCH_SIZE

# Create dataset and dataloader

def get_dataloader():
    dataset = CustomImageDataset(DATASET)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    return dataloader
