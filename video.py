''' Runs predictions on a video using openCV. '''

import argparse
import cv2
import torch
from ultralytics import YOLO
import os
from consts import ID_DATA_PATH, OOD_DATA_PATH


def visualize_predictions(frame, labels, boxes):
    for label, box in zip(labels, boxes):
        class_id = label
        x_min, y_min, x_max, y_max = [int(i) for i in box.xyxy[0]]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Predictions', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def predict_on_video(model, video_path, save_dir, confidence=0.5):
    results = model(video_path, stream=True, device=0)  # return a generator of Results objects
    frame_count = 0
    for result in results:
        frame_count += 1
        with open(os.path.join(save_dir, f'{frame_count:06}.txt'), 'w') as f:
            labels = list()
            for box in result.boxes:
                if box.conf >= confidence:
                    label = int(box.cls.item())
                    labels.append(label)
                    x_center, y_center, w, h = [int(i) for i in box.xywhn[0]]
                    f.write(f'{label} {x_center} {y_center} {w} {h}\n')

        visualize_predictions(result.imgs[0], labels, result.boxes)
        # TODO: See if I need to replace imgs with orig_img

    for i in range(1, frame_count + 1):    
        if os.stat(os.path.join(save_dir, f'{i:06}.txt')).st_size:
            print(f'Frame {i} has detections')


def load_model(model_path, data_path, labels_folder, confidence):
    # Load trained model
    model = YOLO(model_path)

    videos = os.listdir(data_path)
    labels_dir = os.getcwd() + labels_folder

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    predict_on_video(model, data_path+videos[0], labels_dir, confidence)
    # for vid in videos:
    #     predict_video(model, data_path+vid, labels_dir, confidence)


def predict_on_ID(confidence=0.5):
    load_model('model_trained.pt', ID_DATA_PATH, '/pseudo_labels', confidence)

def predict_on_OOD(confidence=0.5):
    load_model('model_finetuned.pt', OOD_DATA_PATH, '/predictions', confidence)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict bounding boxes on a video.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video.')
    args = parser.parse_args()

    predict_video(args.video_path)
