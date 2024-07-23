''' Runs predictions on an image. '''

import cv2
import argparse
import torch
from ultralytics import YOLO
import os
from consts import PT_FILES_PATH
from visualization import visualize_predictions_on_image
from input_processing import *


def predict_on_image(model, image_path, save_dir, confidence):
    image = cv2.imread(image_path)
    results = model(image)

    for result in results:
        with open(save_dir+'.txt', 'w') as f:
            labels = list()
            for box in result.boxes:
                box_conf = float(box.conf.item())
                if box_conf >= confidence:
                    label = int(box.cls.item())
                    labels.append(label)
                    
                    x_center, y_center, w, h = [i.item() for i in box.xywhn[0]]
                    f.write(f'{label} {x_center} {y_center} {w} {h}\n')
    return image, results


def load_model_and_predict(model_path, data_path, labels_folder, visual_folder, confidence=0.5, visualize=True):
    # Load trained model
    model = YOLO(PT_FILES_PATH+model_path)

    labels_dir = create_dir(labels_folder)

    images = os.listdir(data_path)
    for img in images:
        img_path = data_path + img
        img_name = img.split('.')[0]
        img_object, results = predict_on_image(model, img_path, labels_dir+img_name, confidence)
        if visualize:
            visual_dir = create_dir(visual_folder)
            visualize_predictions_on_image(model, img_object, visual_dir+img, results, confidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict bounding boxes on an image.')
    parser.add_argument('-p', '--image_path', type=str, required=False, help='Path to the input image.')
    parser.add_argument('-d', '--images_folder', type=str, required=False, help='Path to the folder containing one or more images.')
    parser.add_argument('-c', '--conf_level', type=float, required=False, default=0.5, help='Confidence threshold for predictions.')
    args = parser.parse_args()
    conf_level = input_error_check(args.image_path, args.images_folder, args.conf_level, 'an image')

    output_dir = '/images_outputs/'
    images_path, weights_file = choose_input_path(output_dir, args.image_path, args.images_folder, 'image/')

    load_model_and_predict(weights_file, images_path, output_dir+'predictions/', output_dir+'visualizations/', confidence=conf_level)
