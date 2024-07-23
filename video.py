''' Runs predictions on a video using openCV. '''

import argparse
import torch
from ultralytics import YOLO
import os
import shutil
from consts import ID_DATA_PATH, OOD_DATA_PATH, PT_FILES_PATH, CONFIDENCE_LEVEL, create_dir
from visualization import visualize_predictions_on_video


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
    return results, frame_w, frame_h


def load_model_and_predict(model_path, data_path, labels_folder, confidence=0.5, visualize=False):
    # Load trained model
    model = YOLO(PT_FILES_PATH+model_path)

    labels_dir = create_dir(labels_folder)

    videos = os.listdir(data_path)
    for vid in videos:
        vid_path = data_path + vid
        vid_name = vid.split('.')[0]
        results, w, h = predict_on_video(model, vid_path, labels_dir+vid_name, confidence)
        if visualize:
            visual_dir = create_dir('/videos_outputs/visualizations/')
            visualize_predictions_on_video(visual_dir+vid, results, w, h, confidence)


def predict_on_ID(confidence=0.5):
    load_model_and_predict('model_trained.pt', ID_DATA_PATH, '/pseudo_labels_ID/', confidence)

def predict_on_OOD(confidence=0.5):
    load_model_and_predict('model_finetuned.pt', OOD_DATA_PATH, '/predictions_OOD/', confidence)
    

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Predict bounding boxes on a video.')
    # parser.add_argument('--video_path', type=str, required=False, help='Path to the input video.')
    # parser.add_argument('--videos_folder', type=str, required=False, help='Path to a folder containing one or more vidoes.')
    # args = parser.parse_args()
    # if not args.video_path and not args.videos_folder:
    #     print('Please provide a video path or a folder containing videos.')
    #     exit()
    # elif args.video_path and args.videos_folder:
    #     print('Please provide only one of the following: a video path or a folder containing videos.')
    #     exit()

    # videos_path = create_dir('/videos_outputs/')
    # if args.video_path:
    #     videos_path = create_dir(videos_path+'video/')
    #     # Copy the file
    #     shutil.copy(args.video_path, videos_path)
    # else:
    #     videos_path = args.videos_folder
    videos_path = '/datashare/HW1/id_video_data/'

    weights_file = 'model_finetuned.pt' if os.path.exists(PT_FILES_PATH+'model_finetuned.pt') else 'model_trained.pt'

    load_model_and_predict(weights_file, videos_path, '/videos_outputs/predictions/', confidence=CONFIDENCE_LEVEL, visualize=True)
