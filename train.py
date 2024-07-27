''' This script is used to train the model on the labeled data and then fine-tune it using the pseudo-labeled data. '''

from ultralytics import YOLO
import os
import cv2
from consts import IMAGE_DATA_PATH, ID_DATA_PATH, PT_FILES_PATH


def train(epochs=50, initial_pt_file_name='yolov8n.pt', yaml_file='yaml_files/images.yaml', result_pt_file_name='model_trained.pt'):
    # Initialize YOLO model
    model = YOLO(PT_FILES_PATH+initial_pt_file_name)
    
    # Training
    model.train(data=yaml_file, epochs=epochs, imgsz=640, batch=8, workers=2)
    
    # Save the model
    model.save(PT_FILES_PATH+result_pt_file_name)


def finetunning(epochs=50):
    train(epochs=epochs, initial_pt_file_name='model_trained.pt', yaml_file='yaml_files/id.yaml', result_pt_file_name='model_finetuned.pt')

