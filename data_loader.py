### Data Loading and Preprocessing

import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SurgicalToolDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt'))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                boxes.append([class_id, x_center, y_center, w, h])
        
        boxes = np.array(boxes)
        
        if self.transform:
            image = self.transform(image)
        
        return image, boxes

transform = transforms.Compose([
    transforms.ToTensor(),
])

def get_dataloader(image_dir, label_dir, batch_size=8, shuffle=True):
    dataset = SurgicalToolDataset(image_dir, label_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# A function that returns the paths to all images and paths to all labels.
