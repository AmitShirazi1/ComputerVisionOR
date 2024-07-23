import shutil
import os
from consts import PT_FILES_PATH, DEFAULT_CONFIDENCE, create_dir


def input_error_check(path_of_file, folder_of_files, conf_level, file_type):
    if not path_of_file and not folder_of_files:
        raise ValueError("Please provide either a path to a file or "+file_type+"s folder.")
    if path_of_file and folder_of_files:
        raise ValueError("Please provide only one of the following: "+file_type+" path or "+file_type+"s folder.")
    
    if conf_level:
        if conf_level < 0 or conf_level > 1:
            print('Confidence level must be between 0 and 1.')
            exit()
    else:
        conf_level = DEFAULT_CONFIDENCE

    return conf_level


def choose_input_path(output_dir, path_of_file, folder_of_files, file_type):
    input_path = create_dir(output_dir)
    if path_of_file:
        input_path = create_dir(input_path + file_type)
        # Copy the file
        shutil.copy(path_of_file, input_path)
    else:
        input_path = folder_of_files

    weights_file = 'model_finetuned.pt' if os.path.exists(PT_FILES_PATH+'model_finetuned.pt') else 'model_trained.pt'
    return input_path, weights_file