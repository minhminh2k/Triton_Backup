import numpy as np
from torchvision import transforms
from tritonclient.utils import triton_to_np_dtype
import cv2
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from PIL import Image
import matplotlib.pyplot as plt
import torch

import os
import glob

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

import time

start_time = time.time()

# Setting up client HTTP and GRPC
# client = httpclient.InferenceServerClient(url="localhost:8000")
client = grpcclient.InferenceServerClient(url="localhost:8001")

# Preprocess using Albumentations
def preprocess(image_path):
    image= Image.open(image_path).convert("RGB")
    image = np.array(image)
    transform = Compose(
        [
            A.Resize(768, 768),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return transform(image=image)["image"].numpy()

# Preprocess Image
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
           
        ])
    return preprocess(img).numpy()

'''
os.makedirs("./output_segment/", exist_ok=True)

# Image folder
# folder_path = "./input_test_v2_8000/"
folder_path = "./input_test_v2_ship_40/"
output_path = "./output_segment/"
'''

os.makedirs("./output/", exist_ok=True)

folder_path = "./input/"
output_path = "./output/"

# List of images
image_files = [os.path.basename(file) for file in glob.glob(os.path.join(folder_path, "*.jpg"))]
print(len(image_files))

# Batch Size
batch_size = 4

# Number 
count = 0


image_files_ship = []

for i in range(0, len(image_files), batch_size):
   
    if i + batch_size < len(image_files):
        batch_images = np.array(list((preprocess(f"{folder_path}{image_files[j]}") for j in range(i, i + batch_size))))
    else:
        batch_images = np.array(list((preprocess(f"{folder_path}{image_files[j]}") for j in range(i, len(image_files)))))
        
    # GRPC
    inputs = grpcclient.InferInput("input_image_resnet34", batch_images.shape, datatype="FP32")
    inputs.set_data_from_numpy(batch_images)
    outputs = grpcclient.InferRequestedOutput("CLASSIFICATION_OUTPUT")
        
    # Querying the server
    results = client.infer(model_name="resnet34", inputs=[inputs])

    inference_output_resnet = results.as_numpy('CLASSIFICATION_OUTPUT')

    batch_images_ship = [] 
   
   
    for output, image in zip(inference_output_resnet, batch_images):
        if output[0] > 0:
            image_files_ship.append(image_files[count])
            batch_images_ship.append(image)
        
        # print(f"{image_files[count]}:", output)
        count += 1
      
    batch_images_ship = np.array(batch_images_ship)
        
    if batch_images_ship.size > 0:
        # print(np.array(batch_images_ship).dtype)
        # batch_images_ship = np.array(batch_images_ship, np.float32)
        # Inference Unet Models
        # GRPC
        inputs_unet = grpcclient.InferInput("input_image_unet34", batch_images_ship.shape, datatype="FP32")
        inputs_unet.set_data_from_numpy(batch_images_ship)
        output_unet = grpcclient.InferRequestedOutput("SEGMENTATION_OUTPUT")
                    
        # Querying the server
        results_unet = client.infer(model_name="unet34", inputs=[inputs_unet])

        inference_output_unet = results_unet.as_numpy('SEGMENTATION_OUTPUT').squeeze()

        # inference_output_unet = inference_output_unet * 255

        for mask, image_path in zip(inference_output_unet, image_files_ship):
            mask = torch.sigmoid(torch.from_numpy(mask.copy()))
            mask = (
                (mask >= 0.5).cpu().numpy().astype(np.uint8)
            )
            
            cv2.imwrite(f"{output_path}{image_path}", mask) # * 255 easier to show
            
            
    batch_images_ship = None
    batch_images = None
    image_files_ship = []


end_time = time.time()

execution_time = end_time - start_time

print(f"Time: {execution_time} seconds")