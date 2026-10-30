# Visual Defect Detective System
### Steel Surface Defect Classification Using an Artificial Neural Network (ANN)

An AI-powered steel surface inspection system that classifies steel images into two categories: **DEFECT** and **CLEAN**. The project uses an Artificial Neural Network (ANN), engineered image features, and an ensemble learning approach to improve classification performance.

## Project Overview

Quality inspection is an important part of steel manufacturing. Surface defects such as scratches, patches, pitting, and inclusions can affect product quality. Manual inspection can be time-consuming and inconsistent.

The Visual Defect Detective System provides an automated approach to steel surface defect classification using machine learning and image processing techniques.

The system analyzes steel surface images, extracts relevant visual features, and predicts whether an image contains a defect.

## Key Features

- Automated steel surface defect classification.
- Binary classification: DEFECT or CLEAN.
- Image preprocessing and feature extraction.
- Background removal and edge-strength analysis.
- Data augmentation using horizontal and vertical flips.
- Fully connected Artificial Neural Network.
- Ensemble of three independently trained ANN models.
- Early stopping and learning-rate scheduling.
- Performance evaluation using classification metrics.
- Interactive Streamlit dashboard.
- Image upload and prediction functionality.
- Steel image validation before classification.

## Dataset

**Dataset:** Severstal Steel Defect Detection

**Source:** Kaggle

https://www.kaggle.com/c/severstal-steel-defect-detection

Dataset information:

| Description | Details |
|---|---|
| Total labeled images | 12,568 |
| Defective images | 6,666 |
| Clean images | 5,902 |
| Original image dimensions | 1600 × 256 |
| Classification type | Binary classification |

The original dataset contains four defect categories. However, this project focuses on image-level binary classification, determining whether a steel surface image contains a defect.

The dataset is divided into three subsets:

| Dataset Split | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

**Note:** The original dataset is not included in this repository. Users must obtain it separately from Kaggle.

## Methodology

The project follows these major stages:

1. **Data Acquisition:** Obtain steel surface images and their corresponding labels from the Severstal dataset.
2. **Label Creation:** Assign DEFECT and CLEAN labels to the images.
3. **Preprocessing and Feature Extraction:** Resize, normalize, remove background variations, and extract image features.
4. **Data Splitting:** Divide the dataset into training, validation, and testing subsets.
5. **Model Training:** Train three fully connected ANN models using the selected hyperparameters.
6. **Model Evaluation:** Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix.
7. **Dashboard Deployment:** Integrate the trained model into an interactive dashboard for image classification.

## Feature Engineering

Instead of directly using raw image pixels, the final model uses **1,483 engineered features per image**.

These features include:

- Block-based statistical features.
- Row and column intensity profiles.
- Global image statistics.
- Brightness distribution.
- Texture information.
- Edge-strength measurements.

Images are resized to 128 × 512 pixels before feature extraction.

## ANN Architecture

The final model uses a fully connected neural network with the following architecture:

```text
Input Layer:       1483 Neurons
       |
Hidden Layer 1:    512 Neurons
       |
Hidden Layer 2:    256 Neurons
       |
Hidden Layer 3:     64 Neurons
       |
Output Layer:        1 Neuron
       |
   Classification
   DEFECT / CLEAN
```

The network uses ReLU activation functions in its hidden layers and a sigmoid activation function in the output layer.

Three independently trained networks are combined using probability averaging to generate the final prediction.

## Model Training Configuration

| Parameter | Value |
|---|---:|
| Maximum Epochs | 60 |
| Batch Size | 256 |
| Learning Rate | 0.001 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Ensemble Size | 3 |
| Early Stopping Patience | 8 |
| Learning Rate Scheduling Patience | 3 |
| Dropout | 0.4, 0.4, 0.3 |

Additional techniques such as batch normalization, data augmentation, and early stopping are used to control overfitting and improve generalization.

## Performance Results

The project compares two model versions:

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|---|---:|---:|---:|
| Version 1: Raw Pixels | 84.3% | 80.7% | 81.0% |
| Version 2: Engineered Features and Ensemble | 96.0% | 90.5% | 91.6% |

<details>
<summary>Performance highlights</summary>

- Final test accuracy: **91.6%**
- Validation accuracy: **90.5%**
- Training accuracy: **96.0%**
- Project target accuracy: **87%**
- Improvement over baseline test accuracy: **10.6 percentage points**

</details>

The final model exceeds the project's target test accuracy of 87%.

## Dashboard

The project provides an interactive dashboard with four main sections:

**1. Data**
- Dataset information.
- Class distribution.
- Image counts.
- Sample images.

**2. Train**
- Adjustable training configurations.
- Model training controls.
- Live training progress.

**3. Results**
- Training, validation, and test accuracy.
- Learning curves.
- Overfitting analysis.
- Confusion matrix.
- Classification report.

**4. Detective**
- Upload a steel surface image.
- Validate the input image.
- Predict DEFECT or CLEAN.
- Display defect prediction probability.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | ANN development and training |
| NumPy | Numerical operations |
| OpenCV | Image preprocessing |
| Scikit-learn | Dataset splitting and evaluation |
| Matplotlib | Performance visualization |
| Streamlit | Interactive web dashboard |
| Gradio | Google Colab dashboard |

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/visual-defect-detective.git
```

### 2. Navigate to the project directory

```bash
cd visual-defect-detective
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run defect_detective_v3.py
```

The application will open in your browser through the local Streamlit server.

## Google Colab

A separate Google Colab notebook is also provided:

`Visual_Defect_Detective_Colab_v2.ipynb`

It implements the project pipeline and provides a Gradio dashboard for interactive predictions. The Colab version does not include the Streamlit steel validation feature.

## Project Structure

```text
visual-defect-detective/
│
├── defect_detective_v3.py
│
├── Visual_Defect_Detective_Colab_v2.ipynb
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
└── defect_detective_output/
    ├── Saved model files
    ├── Evaluation metrics
    ├── Learning curves
    └── Cached features
```

*The actual contents of the output directory depend on the files generated during training.*

## Limitations

- The system performs binary classification rather than identifying individual defect types.
- It does not locate or segment defective regions.
- Very small defects may be lost during image resizing.
- The dataset comes from a single manufacturer, which may limit generalization to other production environments.
- A fully connected ANN with engineered features may not capture all spatial patterns as effectively as a CNN.

## Future Improvements

- Implement defect-type classification.
- Add defect localization and segmentation.
- Compare ANN performance with CNN architectures.
- Improve feature extraction techniques.
- Evaluate the model using datasets from different steel manufacturing environments.

## Academic Project

**Project Title:** Visual Defect Detective System: Steel Surface Defect Classification Using an Artificial Neural Network

**Course:** Artificial Neural Networks (ANN)

**Program:** BS Artificial Intelligence (BS AI)

**Institution:** SZABIST Institute of Science & Technology, Islamabad

**Department:** Robotics & Artificial Intelligence

---

**Developed as an academic machine learning project focused on automated industrial surface inspection.**