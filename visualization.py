''' This file contains functions to visualize the predictions made by the model on images and videos. '''

import cv2
from consts import CLASSES

def visualize_result(frame, result, confidence):
    for box in result.boxes:
        box_conf = float(box.conf.item())
        if box_conf >= confidence:
            label = int(box.cls.item())

            x_min, y_min, x_max, y_max = [int(i) for i in box.xyxy[0]]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 204, 229), 5)
            cv2.putText(frame, f'{label}. {CLASSES[label]}, conf={box_conf:.04}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 204, 229), 2)


def visualize_predictions_on_image(model, image, save_file, results, confidence):
    results = model(image)
    for result in results:
        visualize_result(image, result, confidence)
    cv2.imwrite(save_file, image)


def visualize_predictions_on_video(model, video_path, save_file, w, h, confidence):
    results = model(video_path, stream=True, device=0)
    out = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
    
    for result in results:
        frame = result.orig_img
        visualize_result(frame, result, confidence)
    
        out.write(frame)
    out.release()

