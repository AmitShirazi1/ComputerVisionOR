''' Runs predictions on an image. '''

import argparse
import cv2
import torch
from ultralytics import YOLO

def predict(image_path, model_path='model_finetuned.pt'):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    results = model(image)

    for result in results:
        labels = result['labels']
        boxes = result['boxes']

        for label, box in zip(labels, boxes):
            class_id = label
            x_center, y_center, w, h = box
            x_min = int((x_center - w / 2) * image.shape[1])
            y_min = int((y_center - h / 2) * image.shape[0])
            x_max = int((x_center + w / 2) * image.shape[1])
            y_max = int((y_center + h / 2) * image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, f'{class_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Predictions', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict bounding boxes on an image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()

    predict(args.image_path)
