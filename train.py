''' This script is used to train the model on the labeled data and then fine-tune it using the pseudo-labeled data. '''

import torch
from ultralytics import YOLO
import os
import shutil
import random
from ultralytics import YOLO
# from sklearn.model_selection import train_test_split
from consts import IMAGE_DATA_PATH, PT_FILES_PATH


def train(epochs=50, initial_pt_file_name='yolov8n.pt', yaml_file='data.yaml', result_pt_file_name='model_trained.pt'):
    # Initialize YOLO model
    model = YOLO(PT_FILES_PATH+initial_pt_file_name)  # Using a pretrained YOLOv8n model
    
    # Training
    model.train(data=yaml_file, epochs=epochs, imgsz=640, batch=8, workers=2)
    
    # Save the model
    model.save(PT_FILES_PATH+result_pt_file_name)


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
    model = YOLO(PT_FILES_PATH+'model_initial.pt')
    model.train(data=f'data_split_{i}.yaml', epochs=50, imgsz=640, batch=8, workers=2)
    model.save(f'PT_FILES_PATH+model_split_{i}.pt')
'''

def finetunning(epochs=50):
    train(epochs=epochs, initial_pt_file_name='model_trained.pt', yaml_file='data.yaml', result_pt_file_name='model_finetuned.pt')


''' Combine true and pseudo data -

from ultralytics import YOLO
import os

# Create a new data.yaml including pseudo-labeled data
with open('data_final.yaml', 'w') as f:
    f.write("""
train: data_combined/images
val: data_combined/images
nc: 3
names: ['Empty', 'Tweezers', 'Needle_driver']
""")

# Combine original and pseudo-labeled data
def combine_data(original_image_path, original_label_path, pseudo_image_path, pseudo_label_path, combined_path):
    image_combined_path = os.path.join(combined_path, 'images')
    label_combined_path = os.path.join(combined_path, 'labels')
    os.makedirs(image_combined_path, exist_ok=True)
    os.makedirs(label_combined_path, exist_ok=True)

    for img_file in os.listdir(original_image_path):
        shutil.copy(os.path.join(original_image_path, img_file), image_combined_path)
    for lbl_file in os.listdir(original_label_path):
        shutil.copy(os.path.join(original_label_path, lbl_file), label_combined_path)
    for img_file in os.listdir(pseudo_image_path):
        shutil.copy(os.path.join(pseudo_image_path, img_file), image_combined_path)
    for lbl_file in os.listdir(pseudo_label_path):
        shutil.copy(os.path.join(pseudo_label_path, lbl_file), label_combined_path)

combine_data('/datashare/HW1/labeled_image_data/images/train',
             '/datashare/HW1/labeled_image_data/labels/train',
             'data/pseudo_labels/images',
             'data/pseudo_labels/labels',
             'data_combined')

# Fine-tuning the model using all data
model = YOLO(PT_FILES_PATH+'model_split_4.pt')
model.train(data='data_final.yaml', epochs=50, imgsz=640, batch=8, workers=2)

# Save the fine-tuned model
model.save(PT_FILES_PATH+'model_final.pt')

# Test on OOD video data
ood_videos_path = '/datashare/HW1/ood_video_data'
results = model.predict(ood_videos_path)

# Process and save the results as needed
'''

'''
Maybe use data loader:
# Load data
train_loader = get_dataloader(IMAGE_DATA_PATH+'images/train', IMAGE_DATA_PATH+'labels/train', shuffle=True)  # Need both the train and val
val_loader = get_dataloader(IMAGE_DATA_PATH+'images/val', IMAGE_DATA_PATH+'labels/val', shuffle=True)
pseudo_loader = get_dataloader('data/pseudo_labels/images', 'pseudo_labels/labels')  # Pseudo labels

# Combine loaders
combined_loader = torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset, pseudo_loader.dataset])
'''