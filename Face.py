from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from scipy.spatial.distance import cosine
import cv2
from sys import argv
import ssl
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
os.environ['TORCH_HOME'] = script_directory

ssl._create_default_https_context = ssl._create_unverified_context

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

a,b = argv[1],argv[2]
image1 = cv2.imread(str(a))
image2 = cv2.imread(str(b))
with torch.no_grad():
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    faces1 = mtcnn(image1_rgb)
    faces2 = mtcnn(image2_rgb)
    if faces1 is not None and faces2 is not None:
        embedding1 = resnet(faces1.unsqueeze(0))[0]
        embedding2 = resnet(faces2.unsqueeze(0))[0]
        distance = cosine(embedding1.detach().numpy(), embedding2.detach().numpy())
        threshold = 0.4
        if distance < threshold:
            print('Face Matched')
        else:
            print('Face Not Matched')
    else:
        print('No Face Detected in one of the images.')