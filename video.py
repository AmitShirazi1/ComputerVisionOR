''' Runs predictions on a video using openCV. '''

import argparse
import cv2
import torch
from ultralytics import YOLO

def predict_video(video_path, model_path='model_finetuned.pt'):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            labels = result['labels']
            boxes = result['boxes']

            for label, box in zip(labels, boxes):
                class_id = label
                x_center, y_center, w, h = box
                x_min = int((x_center - w / 2) * frame.shape[1])
                y_min = int((y_center - h / 2) * frame.shape[0])
                x_max = int((x_center + w / 2) * frame.shape[1])
                y_max = int((y_center + h / 2) * frame.shape[0])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Predictions', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict bounding boxes on a video.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video.')
    args = parser.parse_args()

    predict_video(args.video_path)
