import torch
from ultralytics import YOLO
from data_loader import get_dataloader
from consts import IMAGE_DATA_PATH

def finetunning():
    # # Load data
    # train_loader = get_dataloader(IMAGE_DATA_PATH+'images/train', IMAGE_DATA_PATH+'labels/train', shuffle=True)  # Need both the train and val
    # val_loader = get_dataloader(IMAGE_DATA_PATH+'images/val', IMAGE_DATA_PATH+'labels/val', shuffle=True)
    # pseudo_loader = get_dataloader('data/pseudo_labels/images', 'pseudo_labels/labels')  # Pseudo labels

    # # Combine loaders
    # combined_loader = torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset, pseudo_loader.dataset])

    # Initialize YOLO model
    model = YOLO('model_trained.pt')

    # Fine-tuning
    model.train(data='data.yaml', epochs=50, imgsz=640, batch=8, workers=2)

    # Save the fine-tuned model
    model.save('model_finetuned.pt')


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
model = YOLO('model_split_4.pt')
model.train(data='data_final.yaml', epochs=50, imgsz=640, batch=8, workers=2)

# Save the fine-tuned model
model.save('model_final.pt')

# Test on OOD video data
ood_videos_path = '/datashare/HW1/ood_video_data'
results = model.predict(ood_videos_path)

# Process and save the results as needed
'''