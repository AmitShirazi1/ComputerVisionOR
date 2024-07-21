import os
import torch
from ultralytics import YOLO
import cv2
from consts import ID_DATA_PATH


def generate_pseudo_labels(model, video_path, save_dir):
    results = model(video_path, stream=True, device=0)  # return a generator of Results objects
    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nresults:')
    print('type -', type(results))
    frame_count = 0
    for result in results:
        frame_count += 1
        with open(os.path.join(save_dir, f'{frame_count:06}.txt'), 'w') as f:
            for box in result.boxes:
                print('\nbox type -', type(box))
                print('box:\n', box)
                if box.conf:# >= 0.5:
                    # x1, y1, x2, y2 = detection.xyxy[0][0], detection.xyxy[0][1], detection.xyxy[0][2], detection.xyxy[0][3]
                    # x_center = ((x1 + x2) / 2).item()
                    # y_center = ((y1 + y2) / 2).item()
                    # width = ((x2 - x1)).item()
                    # height = ((y2 - y1)).item()
                    label = int(box.cls.item())
            
                    x_center, y_center, w, h = [box.xyxy[0][i].item() for i in range(4)]
                    f.write(f'{label} {x_center} {y_center} {w} {h}\n')
                    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
                    # video_pseudo_labels.append((cls, x_center, y_center, width, height))
                    # print((cls, x_center, y_center, width, height))
        if os.stat(os.path.join(save_dir, f'{frame_count:06}.txt')).st_size:
            print(f'Frame {frame_count} has detections')

    # cap = cv2.VideoCapture(video_path)
    # frame_count = 0

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     results = model(frame)
        
        # for result in results:
            # print('\n1 result type -', type(result))
            # print('1 result:\n', result)
            # labels = result['labels']
            # print('\nlabels type -', type(labels))
            # print('labels:\n', labels)
    #         boxes = result.boxes
    #         print('\nboxes type -', type(boxes))
    #         print('boxes:\n', boxes)

    #         with open(os.path.join(save_dir, f'{frame_count:06}.txt'), 'w') as f:
    #             for label, box in zip(labels, boxes):
    #                 x_center, y_center, w, h = box
    #                 f.write(f'{label} {x_center} {y_center} {w} {h}\n')
    #             print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        
    #     frame_count += 1

    # cap.release()


def predict_on_ID():
    # Load trained model
    model = YOLO('model_trained.pt')

    videos = os.listdir(ID_DATA_PATH)
    pseudo_label_dir = os.getcwd() + '/pseudo_labels'

    if not os.path.exists(pseudo_label_dir):
        os.makedirs(pseudo_label_dir)

    generate_pseudo_labels(model, ID_DATA_PATH+videos[0], pseudo_label_dir)
    # for vid in videos:
    #     generate_pseudo_labels(model, ID_DATA_PATH+vid, pseudo_label_dir)

