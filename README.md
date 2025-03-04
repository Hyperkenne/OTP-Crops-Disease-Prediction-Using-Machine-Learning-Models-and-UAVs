# OTP-Crops-Disease-Prediction-Using-Machine-Learning-Models-and-UAVs
# OTP Crop Disease Prediction

## Overview

This project predicts crop diseases using a **Convolutional Neural Network (CNN)** with **ResNet50**, leveraging **VS Code** for development.

## Dataset

- The dataset contains images of various crops categorized by disease type.
- UAVs (Unmanned Aerial Vehicles) capture real-time field images.



## Installation

### 1. Clone the Repository

```sh
$ git clone https://github.com/your-username/OTP-Crop-Disease-Prediction.git
$ cd OTP-Crop-Disease-Prediction
```

### 2. Create a Virtual Environment (Optional but Recommended)

```sh
$ python -m venv venv
$ source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies

```sh
$ pip install -r requirements.txt
```

## Data Preprocessing

- Extract the dataset:
  ```python
  import zipfile
  zip_path = 'dataset.zip'  # Change to actual dataset path
  extract_dir = 'dataset'
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      zip_ref.extractall(extract_dir)
  ```
- Resize and normalize images:
  ```python
  import cv2
  img_size = (224, 224)
  img = cv2.imread('dataset/sample.jpg')
  img = cv2.resize(img, img_size)
  img = img / 255.0
  ```

## Model Training

- Train the model using **model\_training.py**:
  ```sh
  $ python model_training.py
  ```
- The model is built using **ResNet50**:
  ```python
  from tensorflow.keras.applications import ResNet50
  base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
  ```
- Training is performed using:
  ```python
  model.fit(train_generator, validation_data=val_generator, epochs=80)
  ```

## Model Evaluation

- Compute confusion matrix:
  ```python
  from sklearn.metrics import confusion_matrix
  y_pred = model.predict(val_generator)
  conf_matrix = confusion_matrix(y_true, y_pred)
  ```
- Save confusion matrix as an Excel file:
  ```python
  import pandas as pd
  pd.DataFrame(conf_matrix).to_excel("confusion_matrix.xlsx")
  ```

## Running Predictions

Use **evaluation.py** to test the model on new images:

```sh
$ python evaluation.py
```

## How to Use

1. Upload UAV-captured images.
2. Run the trained model to predict disease categories.
3. Use the results to suggest treatment plans.

## Future Improvements

- Integrate the model into a **mobile app** for real-time disease detection.
- Expand dataset with more crop varieties.
- Use **Firebase** to store and retrieve predictions.

## Contributing

Pull requests are welcome! If you want to improve the model or dataset, feel free to contribute.

## License

MIT License

