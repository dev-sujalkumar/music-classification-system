# Music Classification System

A deep learning project for automatic music genre classification using audio feature extraction and neural networks. The repository demonstrates how to extract MFCC features from audio files and train both CNN and CRNN models to classify music into genres.

## Overview

This project uses the GTZAN-style music dataset and converts each audio clip into Mel Frequency Cepstral Coefficients (MFCCs), which are commonly used for speech and music classification tasks. The extracted features are then passed into neural network models for genre prediction.

The system includes:

- Audio feature extraction and visualization
- MFCC dataset generation to JSON
- CNN-based classification model
- CRNN-based classification model
- Training, validation, and evaluation scripts

## Project Structure

- `audio_features.py` — visualizes waveform, FFT spectrum, spectrogram, and MFCC features for a sample audio file
- `extract_features.py` — extracts MFCC features from the dataset and stores them in `data.json`
- `train_cnn.py` — trains and evaluates a CNN classifier
- `train_crnn.py` — trains and evaluates a CRNN model with convolutional and LSTM layers

## Requirements

Install the required Python dependencies before running the project:

```bash
pip install numpy librosa matplotlib scikit-learn tensorflow
```

If you are using a GPU-enabled TensorFlow environment, the training may run faster; otherwise, CPU is also supported.

## Dataset

Place your dataset folder in the project root as:

```text
genres/
    blues/
    classical/
    country/
    disco/
    hiphop/
    jazz/
    metal/
    pop/
    reggae/
    rock/
```

The code expects the dataset structure used by the GTZAN dataset, where each genre is stored in a separate subfolder.

## How to Run

### 1. Extract features

From the project root:

```bash
python extract_features.py
```

This will create a `data.json` file containing MFCC arrays and their labels.

### 2. Train the CNN model

```bash
python train_cnn.py
```

This script:

- loads the MFCC dataset
- splits it into train/validation/test sets
- builds the CNN model
- trains the model
- plots training curves
- evaluates performance

### 3. Train the CRNN model

```bash
python train_crnn.py
```

This script builds a hybrid CNN + LSTM model for sequence-aware genre classification.

## Model Details

### CNN Model

The CNN model uses:

- 2D convolution layers
- max-pooling layers
- batch normalization
- dense layers
- softmax output for 10 genres

### CRNN Model

The CRNN model adds:

- convolutional feature extraction
- reshaping into sequential format
- LSTM layers to capture temporal information from MFCC sequences

## Output

During training, the project prints:

- training and validation accuracy/loss
- test accuracy
- example prediction for a sample audio segment
- genre-wise evaluation plots

## Notes

- The scripts use a fixed sample rate of 22050 Hz and 30-second track duration.
- Audio is segmented into smaller windows during MFCC extraction to increase sample count.
- The code is intended as a learning project and can be extended for better accuracy with hyperparameter tuning, data augmentation, and deeper models.

## Future Improvements

Possible enhancements include:

- adding a preprocessing pipeline for noise reduction
- experimenting with more audio features such as chroma, spectral contrast, and tempo
- improving model accuracy with transfer learning or stronger architectures
- adding command-line arguments for dataset path and training configuration

## License

This project is available for educational and research use.

## Author

Developed as a music genre classification project using Python, Librosa, and TensorFlow.
