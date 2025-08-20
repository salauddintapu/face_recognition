# FACE RECOGNITION
## Introduction
This project is a real-time face recognition system built with the power of **MTCNN (Multi-task Cascaded Convolutional Networks)** for accurate face detection and **ArcFace** for highly discriminative face recognition.

The system is capable of detecting faces from images or live video streams and recognizing individuals with high precision, even under challenging conditions such as varying lighting, angles, and facial expressions.

By combining MTCNN’s robust face localization with ArcFace’s state-of-the-art feature embeddings, this project achieves:

- Accurate face detection with bounding boxes and facial landmarks.
- Discriminative face recognition using deep feature embeddings.
- Real-time performance for practical applications.

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Getting Started](#getting-started)
4. [Usage](#usage)
5. [API References](#api-references)
6. [License](#license)

## System Architecture
![System Architecture](assets/face_recognition.svg)

## Getting Started
### Prerequisites
- `Python 3.8.0`
- Dependencies are listed in `requirements.txt`

### Installation
```bash
git clone https://github.com/salauddintapu/face_recognition.git
cd face_recognition
```
**Note:**
- It is recommended to use a virtual environment (`venv` or `conda`).
- If using conda, after creating the environment, install bcolz first:

```bash
conda create -n myenv python==3.8.0
conda install -c conda-forge bcolz
pip install -r requirements.txt
```