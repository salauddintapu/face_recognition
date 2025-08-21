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
6. [Acknowledgements](#acknowledgements)

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
<br>

> **⚠️ Note:** It is recommended to use a virtual environment (`venv` or `conda`). If using conda, after creating the environment, install bcolz first:

```bash
conda create -n myenv python==3.8.0
conda install -c conda-forge bcolz
pip install -r requirements.txt
```

## Usage
This section provides step-by-step instructions on how to use the face recognition api.

### Step 1: Prepare the Environment
Prepare the environment by following the steps described in [Getting Started](#getting-started) section. Keep in mind that after setting up the `venv`, you must install `bcolz` first. Installing `Python 3.8.0` and `bcolz` may take some time. Then install the dependencies listed in `requirements.txt`

### Step 2: Configuration Instructions
- In the `app.py`, change `localhost` to machine's IP address if you are using a remote server.
- If `cuda` is avaiable the api will use your machine's GPU otherwise it will run on `cpu`.

### Step 3: Run
To run the API, open your terminal and run,
```bash
conda activate myenv
python app.py
```
### Prepare Your Facebank
To prepare your facebank you have to collect face data and store them in `arcface/data/facebank` directory. You must follow the following structure while storing data:
```
facebank/
├── name-of-person1/images of person1
├── name-of-person2/images of person2
├── name-of-person3/images of person3
```
> **⚠️ Note:** Use cropped faces of the persons you want to recognize. Name the folder after the name of the person and store images of that person in the named folder.

## API References

### Endpoint: `/rec`

#### Payload Example Using cURL
```bash
curl --location 'http://localhost:5000/id_info' \
--header 'Content-Type: application/json' \
--data '{
    "file": "image in base64 format"
}'
```

#### Parameters
| Parameter | Type   | Description                                                         |
|-----------|--------|---------------------------------------------------------------------|
| `file`      | `base64` | Convert your `image` to `base64` format. API uses `opencv` to process it. |

#### Response
```json
{
    "names": "None or list of names",
    "bboxes": "None or list of bbox of face coordinates",
    "scores": "None or list of confidence scores"
}
```

## Acknowledgements

I would like to acknowledge the following resources and works that made this project possible:  

- [MTCNN Face Detector](https://kpzhang93.github.io/MTCNN_face_detection_alignment/)
- [MTCNN Implementation](https://github.com/ipazc/mtcnn)
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- [InsightFace Implementation](https://github.com/deepinsight/insightface) 
- [OpenCV](https://opencv.org/)
- The open-source community for providing accessible libraries and frameworks that accelerate research and development in computer vision.  
