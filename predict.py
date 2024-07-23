''' Runs predictions on an image. '''

import argparse
import cv2
import torch
from ultralytics import YOLO
import os
import shutil
from consts import PT_FILES_PATH, CONFIDENCE_LEVEL, create_dir
from visualization import visualize_predictions_on_image


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


def load_model_and_predict(model_path, data_path, labels_folder, confidence=0.5, visualize=True):
    # Load trained model
    model = YOLO(PT_FILES_PATH+model_path)

    labels_dir = create_dir(labels_folder)

    images = os.listdir(data_path)
    for img in images:
        img_path = data_path + img
        img_name = img.split('.')[0]
        img_object, results = predict_on_image(model, img_path, labels_dir+img_name, confidence)
        if visualize:
            visual_img = create_dir('/images_outputs/visualizations/')
            visualize_predictions_on_image(img_object, visual_img+img, results, confidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict bounding boxes on an image.')
    parser.add_argument('--image_path', type=str, required=False, help='Path to the input image.')
    parser.add_argument('--images_folder', type=str, required=False, help='Path to the folder containing one or more images.')
    args = parser.parse_args()
    if not args.image_path and not args.images_folder:
        raise ValueError('Please provide either an image path or an images folder.')
    if args.image_path and args.images_folder:
        raise ValueError('Please provide only one of the following: an image path or an images folder.')

    images_path = create_dir('/images_outputs/')
    if args.image_path:
        images_path = create_dir(images_path+'image/')
        # Copy the file
        shutil.copy(args.image_path, images_path)
    else:
        images_path = args.images_folder

    weights_file = 'model_finetuned.pt' if os.path.exists(PT_FILES_PATH+'model_finetuned.pt') else 'model_trained.pt'

    load_model_and_predict(weights_file, images_path, '/images_outputs/predictions/', confidence=CONFIDENCE_LEVEL)
