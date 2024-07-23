''' This file contains functions to visualize the predictions made by the model on images and videos. '''

import cv2


def visualize_result(frame, result, confidence):
    for box in result.boxes:
        box_conf = float(box.conf.item())
        if box_conf >= confidence:
            label = int(box.cls.item())

            x_min, y_min, x_max, y_max = [int(i) for i in box.xyxy[0]]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}, {box_conf}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def visualize_predictions_on_image(image, save_file, results, confidence):
    out = cv2.imwrite(save_file, image)
    for result in results:
        visualize_result(image, result, confidence)
    out.write(image)
    out.release()


def visualize_predictions_on_video(save_file, results, w, h, confidence):
    out = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
    
    for result in results:
        frame = result.orig_img
        visualize_result(frame, result, confidence)
    
        out.write(frame)
    out.release()

def visualize_predictions(video_path, output_path, model, classid_classname):
    results = model(video_path, stream=True, device=0)
    first_frame = next(results)
    frame = first_frame.orig_img
    height, width = frame.shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    
    for result in results:
        frame = result.orig_img
        for box in result.boxes:
            x1, y1, x2, y2 = [int(box.xyxy[0][i].item()) for i in range(4)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            cv2.putText(frame, f'{classid_classname[class_id]} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # (255, 105, 180) (255, 0, 127)
        out.write(frame)
    out.release()