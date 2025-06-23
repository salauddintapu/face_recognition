import numpy as np
from PIL import Image
from mtcnn.src import detect_faces

img = Image.open('office1.jpg')
bbox, face_points = detect_faces(img)
print(bbox)