# Surgical Tool Detection using YOLO with Semi-Supervised Learning

This project involves the use of YOLO (You Only Look Once) to train a model on an image dataset, predict pseudo labels on In-Distribution (ID) videos, fine-tune the model, and then predict on Out-Of-Distribution (OOD) videos. The project leverages semi-supervised learning techniques to improve the detection accuracy of surgical tools.

## Project Structure

- **predict.py**: Functions to run predictions on images.
- **video.py**: Functions to run predictions on videos.
- **train.py**: Functions to train the model on labeled data and fine-tune using pseudo-labeled data.
- **visualization.py**: Functions to visualize predictions on images and videos.
- **main.py**: Main script to execute training, pseudo-labeling, fine-tuning, and OOD prediction.
- **input_processing.py**: Contains functions for input validation and path selection.
- **yaml_files**
  - `id.yaml`: Configuration file for the In-Distribution dataset.
  - `images.yaml`: Configuration file for the labeled image dataset.
- **consts.py**: Contains constant paths and utility functions.
- **requirements.txt**: Lists the required Python packages.


## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/AmitShirazi1/ComputerVisionOR.git
   cd ComputerVisionOR
   ```

2. Recommended -    
   Create a virtual environment:
   ```sh
   python -m venv cv_hw1
   source cv_hw1/bin/activate
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model

Train the model on the labeled image dataset:

```sh
python main.py
```

This will:
- Train the model for 50 epochs on the labeled image dataset.
- Predict pseudo labels on the ID videos.
- Fine-tune the model for 25 epochs using the pseudo-labeled data.
- Predict on the OOD videos with visualization.
(Note that you can change the number of epochs to your liking, it is set in the main.py file.)

### 2. Predicting on Images

To run predictions on images:

```sh
python predict.py -i <image_path> -c <confidence_level>
```

or

```sh
python predict.py -d <images_folder> -c <confidence_level>
```

### 3. Predicting on Videos

To run predictions on videos:

```sh
python video.py -v <video_path> -c <confidence_level>
```

or

```sh
python video.py -d <videos_folder> -c <confidence_level>
```

## File Descriptions

- **yaml_files/id.yaml**: Configuration for the In-Distribution dataset.
- **yaml_files/images.yaml**: Configuration for the labeled image data.
- **consts.py**: Contains constants like paths and confidence levels.
- **input_processing.py**: Handles input validation and directory creation.
- **main.py**: Main script to control the workflow.
- **predict.py**: Script for running predictions on images.
- **train.py**: Contains training and fine-tuning functions.
- **video.py**: Script for running predictions on videos.
- **visualization.py**: Contains functions for visualizing predictions.

## Contributing

Feel free to open issues or submit pull requests for improvements and bug fixes.

## License

This project is currently not licensed for public use.

---
