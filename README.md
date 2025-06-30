# Neural Decoding with MindForce

**MindForce** is a custom-designed neural interface device capable of acquiring peripheral nerve signals—specifically from the **radial and ulnar nerves**—at the wrist level. This project focuses on decoding these signals to classify human hand gestures using deep learning.

## Overview

This repository presents a **gesture classification framework** that processes nerve signals extracted from the wrist using **MindForce**. The core goal is to enable **robust neural decoding** across time (inter-session) and across individuals (inter-subject).

## Pipeline

The overall pipeline consists of the following stages:

1. **Data Acquisition**

   * Signals are recorded from the **radial and ulnar nerves** using the MindForce device.
   * Each session consists of multiple gesture trials, recorded at a fixed sampling rate.

2. **Windowing**

   * Continuous signals are segmented into overlapping windows (e.g., 250–500 ms) with a fixed, overlapped stride.
   * Windowing improves temporal resolution and enables frame-based classification.

3. **Feature Extraction**

   * Multiple features are extracted per window to capture discriminative signal characteristics. Current features include:

     * Time-domain: RMS, MAV, ZC, WL
     * Frequency-domain: MNF, MDF, Spectral Entropy
     * Wavelet-based features (optional)

4. **Neural Classification**

   * Extracted features are fed into a **Deep Neural Network (DNN)**.
   * Current architecture includes:

     * Fully-connected layers with dropout
     * Optional temporal encoders (e.g., LSTM or CNN) depending on experimental variant

## Research Focus

### Inter-Session Generalization (Current Target)

We aim to develop models that **generalize across sessions** for the same subject. This setting captures temporal variations such as:

* Electrode shift
* Muscle fatigue
* Skin impedance changes

Models are trained on data from selected sessions and tested on temporally distant sessions from the same subject.

### Inter-Subject Generalization (Future Work)

Later phases of the project will extend to **inter-subject generalization**, targeting subject-independent decoding. This scenario requires the model to learn **subject-invariant features**, which is more challenging due to anatomical and physiological variability.

## Goals

* Develop a neural decoder that can generalize across sessions (temporal robustness)
* Extend to cross-subject decoding for real-world usability
* Analyze the effectiveness of various features and model architectures
* Investigate domain adaptation and personalization methods

## Repository Structure

```
📂 MindForce_NeuralDecoding/
├── data/                # Raw and preprocessed MindForce recordings
├── features/            # Feature extraction scripts
├── models/              # DNN training and evaluation code
├── utils/               # General utility functions (windowing, normalization)
├── results/             # Saved model weights, evaluation logs, figures
└── README.md            # Project description
```

## Gesture Class
1. Index finger flexion
2. Index finger pinch
3. Ring finger pinch
4. Wrist roll (Inward)
5. Six
6. Fist grasp
7. Stretch palm


## Acknowledgements

This project is developed in the Neural Interfaces Lab. MindForce hardware was custom-designed for this study.
