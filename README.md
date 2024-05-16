### README.md

# Affective Computing Model for Natural Interaction Based on Large-Scale Self-Built Dataset

This repository contains the code associated with the research paper titled **"Affective Computing Model for Natural Interaction Based on Large-Scale Self-Built Dataset."** The paper proposes building a large facial expression dataset from internet-sourced data and utilizing downsampling and data augmentation to address category imbalance in facial expression recognition. Utilizing a DenseNet-based model trained on this new dataset, the research achieved a state-of-the-art accuracy of 97.5% on the CK+ benchmark, demonstrating the effectiveness of the approach compared to existing methods using smaller or unbalanced datasets.

## Project Overview

The primary components of this project include:
- Data loading and processing for various datasets (CelebA, CK+, etc.)
- Facial expression recognition using a DenseNet-based model
- Data augmentation and downsampling techniques to handle category imbalance
- Evaluation of model performance

## Project Directory Structure

```plaintext
/
- 01maskload.py
- 03testcelebA.py
- blazeface.py
- celebAload.py
- classify.py
- cliputils.py
- compose.py
- dialogload.py
- dialogplot.py
- eval_detect.py
- eval_lfw.py
```

## File Descriptions

### 01maskload.py
This script loads and processes images with and without masks, performs mask recognition experiments, calculates similarity matrices, and evaluates accuracy.

### 03testcelebA.py
This script tests the CelebA dataset, calculates image and text similarity matrices, and evaluates model accuracy.

### blazeface.py
This script implements the BlazeFace face detection model, including model loading, image preprocessing, face detection, and display of results.

### celebAload.py
This script handles the CelebA dataset, including sample loading, attribute concatenation, and prompt generation. It generates captions for CelebA images using a language model and saves them to a file.

### classify.py
This script loads a TensorFlow model for image classification and provides methods for single image and batch classification.

### cliputils.py
This script uses the CLIP model to calculate similarity matrices between images and captions, evaluates model performance, and provides utilities for drawing matrix figures.

### compose.py
This script reads frames from a video, performs face alignment and detection, and saves the results as images. It also provides functionality to merge images into a video using FFmpeg.

### dialogload.py
This script loads and samples data from the CelebA Dialog JSON dataset, constructing image path and caption arrays.

### dialogplot.py
This script plots and saves images and their corresponding captions from the CelebA Dialog dataset.

### eval_detect.py
This script evaluates the performance of a face detection model, including forward pass times and frames per second metrics.

### eval_lfw.py
This script evaluates the performance of a face recognition model on the LFW dataset, including accuracy calculations and forward pass times.

## Usage

Make sure you have downloaded and prepared the required datasets (e.g., CelebA, CK+, LFW).

### Running Specific Scripts
- **Mask Recognition**:
  ```bash
  python 01maskload.py
  ```
- **CelebA Dataset Testing**:
  ```bash
  python 03testcelebA.py
  ```
- **BlazeFace Face Detection**:
  ```bash
  python blazeface.py
  ```
- **Data Loading and Processing**:
  ```bash
  python celebAload.py
  ```
- **Image Classification**:
  ```bash
  python classify.py
  ```
- **CLIP Model Utilities**:
  ```bash
  python cliputils.py
  ```
- **Video Processing**:
  ```bash
  python compose.py
  ```
- **Dialog Data Loading**:
  ```bash
  python dialogload.py
  ```
- **Dialog Data Plotting**:
  ```bash
  python dialogplot.py
  ```
- **Model Evaluation**:
  ```bash
  python eval_detect.py
  python eval_lfw.py
  ```