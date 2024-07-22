import os
import torch
from ultralytics import YOLO
import cv2
from consts import ID_DATA_PATH


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



