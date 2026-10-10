# Radio Signal Quality Classification

Deep learning project for radio signal quality estimation using spectrogram-based CNN classification. Uses the DeepSig RadioML GOLD_XYZ_OSC.0001_1024 dataset to classify IQ samples into three quality classes (Poor, Average, Good) based on Signal-to-Noise Ratio (SNR).

## Overview

This project implements multiple CNN architectures to classify radio signal quality from IQ samples converted to spectrograms. The classification is based on SNR thresholds:
- **Poor**: SNR < 0 dB
- **Average**: 0 ≤ SNR ≤ 10 dB
- **Good**: SNR > 10 dB

## Features

- **Fixed spectrogram computation**: Uses `scipy.stft` with proper complex signal handling
- **Per-sample normalization**: Prevents data leakage between train/test sets
- **Data augmentation**: Random horizontal flip and brightness jitter
- **Multiple model architectures**:
  - Simple CNN from scratch
  - Improved CNN with BatchNorm & Dropout
  - MobileNetV2 transfer learning with two-phase fine-tuning
- **Balanced dataset**: 6,000 samples per class (18,000 total)
- **Cosine decay learning rate schedule** for fine-tuning phase

## Dataset

- **Source**: DeepSig RadioML / GOLD_XYZ_OSC.0001_1024
- **Format**: HDF5 file containing IQ samples of shape (N, 2, 1024)
- **Metadata**: SNR and modulation index for each sample
- **Total samples**: 2,556,004 (balanced subset of 18,000 used for training)

## Installation

### Requirements

- Python 3.9+
- TensorFlow 2.16+
- NumPy
- SciPy
- OpenCV
- scikit-learn
- Matplotlib
- Seaborn
- h5py

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd DeepLearning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
- Place `GOLD_XYZ_OSC.0001_1024.hdf5` in the `archive/` directory
- Dataset available from [DeepSig RadioML](https://www.deepsig.ai/datasets)

## Usage

### Running the Notebook

Open and run the Jupyter notebook:

```bash
jupyter notebook deepsig_improved_notebook\ \(3\).ipynb
```

### Key Components

#### Spectrogram Generation

The `fast_spectrogram()` function converts IQ samples to log-magnitude spectrograms:
- Combines I and Q channels into complex signal
- Applies STFT with proper complex handling
- Performs per-sample normalization to [0, 1]
- Resizes to 128×128 pixels

#### Model Training

Each model is trained with:
- 80/20 train-test split (stratified)
- Data augmentation (training only)
- Early stopping (patience=10)
- ReduceLROnPlateau callback

#### Evaluation

Models are evaluated using:
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Training/validation accuracy and loss curves

## Model Architectures

### Model 1: Simple CNN

- 3 convolutional blocks (16, 32, 64 filters)
- Global average pooling
- Dense classifier head (256 units)
- ~40K parameters

### Model 2: Improved CNN

- 3 convolutional blocks with BatchNorm & Dropout
- Progressive regularization (0.25 → 0.3 → 0.4)
- Deeper classifier head with BatchNorm
- Better generalization through regularization

### Model 3: MobileNetV2 Transfer Learning

- Pre-trained ImageNet weights
- Two-phase fine-tuning:
  - Phase 1: Frozen base, train top layers
  - Phase 2: Unfreeze top 50 layers, lower learning rate
- Cosine decay LR schedule
- Deeper classifier head (256 → 128 → 3)

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Simple CNN | 85% | 0.85 | 0.85 | 0.85 |
| Improved CNN | TBD | TBD | TBD | TBD |
| MobileNetV2 | TBD | TBD | TBD | TBD |

### Key Improvements

1. **Fixed spectrogram**: Proper complex signal handling preserves full signal information
2. **Per-sample normalization**: Eliminates data leakage from test set
3. **Increased samples**: 6,000 per class (up from 4,000) improves generalization
4. **Data augmentation**: Reduces overfitting on small dataset
5. **Extended early stopping**: Patience increased from 4 to 10 epochs
6. **Two-phase fine-tuning**: Better transfer learning performance

## Project Structure

```
DeepLearning/
├── archive/
│   ├── GOLD_XYZ_OSC.0001_1024.hdf5    # Dataset file
│   ├── classes-fixed.json
│   └── classes.txt
├── deepsig_improved_notebook (3).ipynb  # Main notebook
└── README.md
```

## Technical Details

### Spectrogram Computation

The spectrogram generation uses:
- `scipy.signal.stft` for correct complex signal processing
- Log-magnitude compression via `np.log1p()`
- Per-sample min-max normalization to [0, 1]
- OpenCV resize to 128×128

### Data Augmentation

- **RandomFlip (horizontal)**: Valid due to time-symmetry of spectrograms
- **RandomBrightness (factor=0.15)**: Simulates signal power variations

### Training Configuration

- **Batch size**: 64
- **Max epochs**: 50
- **Optimizer**: Adam
- **Loss**: Sparse categorical crossentropy
- **Metrics**: Accuracy

## References

- DeepSig RadioML Dataset: https://www.deepsig.ai/datasets
- TensorFlow: https://www.tensorflow.org/
- SciPy Signal Processing: https://docs.scipy.org/doc/scipy/reference/signal.html

## License

This project uses the DeepSig dataset. Please refer to the dataset's license in `archive/LICENSE.TXT`.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
