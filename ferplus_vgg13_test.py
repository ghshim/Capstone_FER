import os
import os.path
from PIL import Image
import cv2

import torch
import torch.optim as optim
import torch.utils.data as datay
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg13
import torch.nn as nn
import math

import torchvision
from torchvision import transforms

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocess import preprocess

### about printing on the image
'''
print another label on the image if there are other labels with a high probability except top_class
'''
def print_label(frame, output, top_class):
    global classes
    # feedback
    count = 2
    for i in range(len(classes)):
        if output[0][i] > 0.5 and classes[i] != top_class:
            x = 100 * count
            string = str(classes[i]) + ':' + str(output[0][i])
            cv2.putText(frame, string, (100, x), cv2.FONT_ITALIC, 2, (0,0,0), 2)
            count += 1
    return frame

'''
change array to tensor for CNN input
'''
def array2tesnor(image):
    # transform for image
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor_image = trans(image).unsqueeze(dim=0)

    return tensor_image



# for model
MODEL_DIR = '../pretrained'
MODEL_NAME = 'FERPlusmodel.pth'
device = "cpu"  #cpu

# emotion classes
classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

# model load
print("Loading model...")
model = vgg13()
model.classifier = nn.Sequential(nn.Linear(7 * 7 * 512, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(1024, 8))

model.to(device)
model.load_state_dict(torch.load(f'{MODEL_DIR}/{MODEL_NAME}', map_location=device))
print("model loaded...")

# real-time emotion recognition
webcam = cv2.VideoCapture(cv2.CAP_V4L+0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    
    # preprocess_frame = preprocess.image_preprocessing(frame) # input for cnn
    preprocess_frame = frame

    if preprocess_frame is None:
        result = 'Not Defined'
        cv2.putText(frame, result, (50, 100), cv2.FONT_ITALIC, 2, (0,0,255), 2)
    
    else:
    
        tensor = array2tesnor(frame).to(device)
        output = model(tensor)
        ps = torch.exp(output)

        prob = torch.nn.functional.softmax(output, dim=1)
        top_p, top_class = prob.topk(1, dim=1) # extract top class index and probability
        result = classes[top_class]
        frame = print_label(frame, output, result) # feedback

        # put final class on the frame
        cv2.putText(frame, result, (50, 100), cv2.FONT_ITALIC, 2, (0,0,255), 2)
        
    if status:
        cv2.imshow("test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()