''' Runs predictions on a video using openCV. '''

import argparse
from ultralytics import YOLO
import os
import cv2
from consts import ID_DATA_PATH, OOD_DATA_PATH, PT_FILES_PATH, create_dir
from visualization import visualize_predictions_on_video
from input_processing import *


def predict_on_video(model, video_path, save_dir, vid_name, confidence):
    results = model(video_path, stream=True, device=0)  # Return a generator of Results objects
    
    def create_save_dirs(save_dir, type):
        save_dir = save_dir[0] + type + save_dir[1]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir
    save_dir_image = create_save_dirs(save_dir, 'images/')
    save_dir_label = create_save_dirs(save_dir, 'labels/')
    
    frame_count = 0
    for result in results:
        # Get the frame dimensions
        if frame_count == 0:
            frame_h, frame_w = result.orig_shape
        frame_count += 1
        cv2.imwrite(os.path.join(save_dir_image, f'{frame_count:06}_{vid_name}.jpg'), result.orig_img)  # Save the frame as an image file

        # Save the predicted labels to a file
        with open(os.path.join(save_dir_label, f'{frame_count:06}_{vid_name}.txt'), 'w') as f:
            labels = list()
            for box in result.boxes:
                box_conf = float(box.conf.item())
                if box_conf >= confidence:
                    label = int(box.cls.item())
                    labels.append(label)
                    x_center, y_center, w, h = [i.item() for i in box.xywhn[0]]
                    f.write(f'{label} {x_center} {y_center} {w} {h}\n')  # Write the predictions in YOLO format

    return frame_w, frame_h


def load_model_and_predict(model_path, data_path, save_dir, visual_folder, confidence=0.5, visualize=False):
    model = YOLO(PT_FILES_PATH+model_path)  # Load trained model

    videos = os.listdir(data_path)  # Get the list of videos in the folder
    for vid in videos:
        vid_path = data_path + vid
        vid_name = vid.split('.')[0]
        w, h = predict_on_video(model, vid_path, save_dir, vid_name, confidence)
        if visualize:
            visual_dir = create_dir(visual_folder)  # Create a directory to save the visualizations
            visualize_predictions_on_video(model, vid_path, visual_dir+vid, w, h, confidence)


def predict_on_ID(confidence=0.5):
    load_model_and_predict('model_trained.pt', ID_DATA_PATH, ('image_and_ID_data/', 'train/'), '', confidence)

def predict_on_OOD(confidence=0.5, visualize=False):
    load_model_and_predict('model_finetuned.pt', OOD_DATA_PATH, ('OOD/', ''), '/OOD/visualizations/', confidence, visualize)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict bounding boxes on a video.')
    parser.add_argument('-v', '--video_path', type=str, required=False, help='Path to the input video.')
    parser.add_argument('-d', '--videos_folder', type=str, required=False, help='Path to a folder containing one or more vidoes.')
    parser.add_argument('-c', '--conf_level', type=float, required=False, default=0.5, help='Confidence threshold for predictions.')
    args = parser.parse_args()
    conf_level = input_error_check(args.video_path, args.videos_folder, args.conf_level, 'a video')

    output_dir = '/videos_outputs/'
    videos_path, weights_file = choose_input_path(output_dir, args.video_path, args.videos_folder, 'video/')

    load_model_and_predict(weights_file, videos_path, (output_dir, ''), output_dir+'visualizations/', confidence=conf_level, visualize=True)
