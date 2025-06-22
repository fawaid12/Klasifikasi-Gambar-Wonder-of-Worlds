
# Wonders of the World Image Classification

This project is a Convolutional Neural Network (CNN)-based image classification pipeline for recognizing the top 4 classes of the Wonders of the World dataset.

## Features

- Image preprocessing and augmentation using `ImageDataGenerator`
- Transfer learning with MobileNet
- Evaluation with confusion matrix and classification report
- Conversion to TensorFlow Lite and TensorFlow.js for deployment
- Training callbacks including EarlyStopping and ReduceLROnPlateau

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
project/
│
├── data/                 # Original dataset
├── processed_data/       # Preprocessed and split dataset
├── models/               # Saved model files
├── tflite_model/         # TensorFlow Lite exported model
├── tfjs_model/           # TensorFlow.js exported model
├── notebooks/            # Jupyter notebooks
├── requirements.txt
└── README.md
```

## Usage

### 1. Prepare Data

Make sure the dataset is downloaded and placed under `data/`. Then split it:

```python
import splitfolders
splitfolders.ratio('data/', output='processed_data', seed=1570, ratio=(.7, .15, .15))
```

### 2. Train Model

Run the training notebook to train the CNN model on top 4 classes using model sequential and MobileNet.

### 3. Evaluate

The model is evaluated using accuracy, confusion matrix, and classification report.

### 4. Export Model

Use TensorFlow APIs to convert the model into `.tflite` and `.tfjs` formats for deployment.

## Model Conversion Commands

```python
# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model('models/final_model')
tflite_model = converter.convert()
with open('tflite_model/model.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert to TensorFlow.js
!tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model submission/saved_model submission/tfjs_model
```

## License

This project is open-source and free to use under the MIT License.
