''' Runs predictions on a video using openCV. '''

import argparse
import torch
from ultralytics import YOLO
import os
import shutil
from consts import ID_DATA_PATH, OOD_DATA_PATH, PT_FILES_PATH, DEFAULT_CONFIDENCE, create_dir
from visualization import visualize_predictions_on_video
from input_processing import *


def predict_on_video(model, video_path, save_dir, confidence):
    results = model(video_path, stream=True, device=0)  # return a generator of Results objects
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    frame_count = 0
    for result in results:
        if frame_count == 0:
            frame_h, frame_w = result.orig_shape
        frame_count += 1

        with open(os.path.join(save_dir, f'{frame_count:06}.txt'), 'w') as f:
            labels = list()
            for box in result.boxes:
                box_conf = float(box.conf.item())
                if box_conf >= confidence:
                    label = int(box.cls.item())
                    labels.append(label)
                    x_center, y_center, w, h = [i.item() for i in box.xywhn[0]]
                    f.write(f'{label} {x_center} {y_center} {w} {h}\n')

    for i in range(1, frame_count + 1):    
        if os.stat(os.path.join(save_dir, f'{i:06}.txt')).st_size:
            print(f'Frame {i} has detections')
    print('frames in video:', frame_count)
    return frame_w, frame_h


def load_model_and_predict(model_path, data_path, labels_folder, visual_folder, confidence=0.5, visualize=False):
    # Load trained model
    model = YOLO(PT_FILES_PATH+model_path)

    labels_dir = create_dir(labels_folder)

    videos = os.listdir(data_path)
    for vid in videos:
        vid_path = data_path + vid
        vid_name = vid.split('.')[0]
        w, h = predict_on_video(model, vid_path, labels_dir+vid_name, confidence)
        if visualize:
            visual_dir = create_dir(visual_folder)
            visualize_predictions_on_video(model, vid_path, visual_dir+vid, w, h, confidence)


def predict_on_ID(confidence=0.5):
    load_model_and_predict('model_trained.pt', ID_DATA_PATH, '/pseudo_labels_ID/', '', confidence)

def predict_on_OOD(confidence=0.5, visualize=False):
    load_model_and_predict('model_finetuned.pt', OOD_DATA_PATH, '/predictions_OOD/', '/visualizations_OOD/', confidence, visualize)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict bounding boxes on a video.')
    parser.add_argument('-p', '--video_path', type=str, required=False, help='Path to the input video.')
    parser.add_argument('-d', '--videos_folder', type=str, required=False, help='Path to a folder containing one or more vidoes.')
    parser.add_argument('-c', '--conf_level', type=float, required=False, default=0.5, help='Confidence threshold for predictions.')
    args = parser.parse_args()
    conf_level = input_error_check(args.video_path, args.videos_folder, args.conf_level, 'a video')

    output_dir = '/videos_outputs/'
    videos_path, weights_file = choose_input_path(output_dir, args.video_path, args.videos_folder, 'video/')

    load_model_and_predict(weights_file, videos_path, output_dir+'predictions/', output_dir+'visualizations/', confidence=conf_level, visualize=True)
