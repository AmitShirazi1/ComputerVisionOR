import torch
from ultralytics import YOLO
import os
import shutil
import random
from ultralytics import YOLO
# from sklearn.model_selection import train_test_split
from data_loader import get_dataloader
from consts import IMAGE_DATA_PATH


def train(epochs=50):
    # # Load data
    # train_loader = get_dataloader(IMAGE_DATA_PATH+'images/train', IMAGE_DATA_PATH+'labels/train')
    # val_loader = get_dataloader(IMAGE_DATA_PATH+'images/val', IMAGE_DATA_PATH+'labels/val', shuffle=False)
    
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # Using a pretrained YOLOv8n model
    
    # Training
    model.train(data='data.yaml', epochs=epochs, imgsz=640, batch=8, workers=2)
    
    # Save the model
    model.save('model_trained.pt')


''' Shuffling and splitting the data -

def load_data(image_path, label_path):
    images = sorted(os.listdir(image_path))
    labels = sorted(os.listdir(label_path))
    data = list(zip(images, labels))
    random.shuffle(data)
    return data

def split_data(data, test_size=0.2):
    train_data, val_data = train_test_split(data, test_size=test_size)
    return train_data, val_data

def save_split_data(data, image_dir, label_dir, split_dir):
    image_split_dir = os.path.join(split_dir, 'images')
    label_split_dir = os.path.join(split_dir, 'labels')
    os.makedirs(image_split_dir, exist_ok=True)
    os.makedirs(label_split_dir, exist_ok=True)
    
    for image, label in data:
        shutil.copy(os.path.join(image_dir, image), os.path.join(image_split_dir, image))
        shutil.copy(os.path.join(label_dir, label), os.path.join(label_split_dir, label))

image_path = '/datashare/HW1/labeled_image_data/images/train'
label_path = '/datashare/HW1/labeled_image_data/labels/train'
val_image_path = '/datashare/HW1/labeled_image_data/images/val'
val_label_path = '/datashare/HW1/labeled_image_data/labels/val'

# Load and shuffle the data
train_data = load_data(image_path, label_path)
val_data = load_data(val_image_path, val_label_path)
all_data = train_data + val_data

# Iterate through multiple splits
for i in range(5):  # Number of iterations
    train_split, val_split = split_data(all_data)

    # Save split data
    save_split_data(train_split, image_path, label_path, f'split_data/train_split_{i}')
    save_split_data(val_split, image_path, label_path, f'split_data/val_split_{i}')

    # Create a temporary data.yaml for this split
    with open(f'data_split_{i}.yaml', 'w') as f:
        f.write(f"""
                train: split_data/train_split_{i}/images
                val: split_data/val_split_{i}/images
                nc: 3
                names: ['Empty', 'Tweezers', 'Needle_driver']
                """)

    # Train the model on the split data
    model = YOLO('model_initial.pt')
    model.train(data=f'data_split_{i}.yaml', epochs=50, imgsz=640, batch=8, workers=2)
    model.save(f'model_split_{i}.pt')
'''
