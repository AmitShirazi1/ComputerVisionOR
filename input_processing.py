import shutil
import os
from consts import PT_FILES_PATH, DEFAULT_CONFIDENCE, create_dir


def input_error_check(path_of_file, folder_of_files, conf_level, file_type):
    """ Checks if the input arguments are valid, and if so, returns the confidence level. """
    if not path_of_file and not folder_of_files:
        print("Please provide either a path to a file or "+file_type+"s folder.")
        exit()
    if path_of_file and folder_of_files:
        print("Please provide only one of the following: "+file_type+" path or "+file_type+"s folder.")
        exit()
    
    if conf_level:
        if conf_level < 0 or conf_level > 1:
            print('Confidence level must be between 0 and 1.')
            exit()
    else:
        conf_level = DEFAULT_CONFIDENCE

    return conf_level


def choose_input_path(output_dir, path_of_file, folder_of_files, file_type):
    """ Returns the path of the input file and the name of the weights file. """
    create_dir(output_dir)  # Make sure the output directory (to which the predictions will be saved) exists.
    if path_of_file:
        # If the user provided a file path, copy the file to the output directory
        input_path = create_dir(output_dir + file_type)
        shutil.copy(path_of_file, input_path)  # Copy the file

    else:
        # If the user provided a folder path, make sure it ends with a '/'
        if not folder_of_files.endswith('/'):
            folder_of_files += '/'
        input_path = folder_of_files

    # Choose the weights file
    weights_file = 'model_finetuned.pt' if os.path.exists(PT_FILES_PATH+'model_finetuned.pt') else 'model_trained.pt'
    return input_path, weights_file