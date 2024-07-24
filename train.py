''' This script is used to train the model on the labeled data and then fine-tune it using the pseudo-labeled data. '''

from ultralytics import YOLO
import os
from ultralytics import YOLO
import cv2
from consts import IMAGE_DATA_PATH, ID_DATA_PATH, PT_FILES_PATH


def train(epochs=50, initial_pt_file_name='yolov8n.pt', yaml_file='yaml_files/images.yaml', result_pt_file_name='model_trained.pt'):
    # Initialize YOLO model
    model = YOLO(PT_FILES_PATH+initial_pt_file_name)  # Using a pretrained YOLOv8n model
    
    # Training
    model.train(data=yaml_file, epochs=epochs, imgsz=640, batch=8, workers=2)
    
    # Save the model
    model.save(PT_FILES_PATH+result_pt_file_name)


def copy_ID_frames_to_combined_folder(video_name, num_frames, combined_folder_path='image_and_ID_data/'):
    # Copy the ID frames to the combined folder
    # Open the video file
    cap = cv2.VideoCapture(ID_DATA_PATH + video_name)

    frame_count = 0
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the current frame as an image file
        frame_filename = os.path.join(combined_folder_path+"images/train/", f'{frame_count:06}.txt')
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    # Release the video capture object
    cap.release()


def finetunning(epochs=50):
    train(epochs=epochs, initial_pt_file_name='model_trained.pt', yaml_file='yaml_files/id.yaml', result_pt_file_name='model_finetuned.pt')

