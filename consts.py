IMAGE_DATA_PATH = '/datashare/HW1/labeled_image_data/'
ID_DATA_PATH = '/datashare/HW1/id_video_data/'
OOD_DATA_PATH = '/datashare/HW1/ood_video_data/'

PT_FILES_PATH = './yolo_pt_files/'
CONFIDENCE_LEVEL = 0

import os
def create_dir(relative_dir):
    full_dir = os.getcwd() + relative_dir
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    return full_dir