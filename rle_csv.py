import numpy as np
import pandas as pd
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
from skimage.morphology import label 

import os
import glob

# Create csv
csv_file = './data.csv'
if not os.path.exists(csv_file):
    # Create datadrame
    data = {'ImageId': [], 'EncodedPixels': []}
    df = pd.DataFrame(data)
    # Save
    df.to_csv(csv_file, index=False)

# Folder Path
os.makedirs("./mask_output/", exist_ok=True)

# Image folder
input_path = "./mask_cvat/"
output_path = "./mask_output/"

ship_dir = "./"

input_image = "./input"

# List of images
image_files = [os.path.basename(file) for file in sorted(glob.glob(os.path.join(input_path, "*.jpg")) + glob.glob(os.path.join(input_path, '*.png')))]
# print(image_files)
 
def rle_decode(mask_rle, shape=(768, 768)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    # if mask_rle == 'nan':
    #     return np.zeros(shape[0] * shape[1], dtype=np.uint8).reshape(shape).T
    
    s = mask_rle.split()
    starts, lengths = (np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2]))
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
    
def multi_rle_encode(img):
    labels = label(img)  # connected components marked with same value (label)
    unique_labels = np.unique(labels[labels>0]) # len(unique_labels) = number of ships
    return [rle_encode(labels==k) for k in unique_labels]  # a list of encoded pixels

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def mask_overlay(image, mask, color=(0, 1, 0)):
    """Helper function to visualize mask on the top of the image."""
    mask = mask.squeeze()  # mask could be (1, 768, 768) or (768, 768)
    mask = np.dstack((mask, mask, mask)) * np.array(color, dtype=np.uint8) * 255
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.0)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img

def imshow(img, mask, title=None):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(mask_overlay(img, mask))
    if title is not None:
        plt.title(title)
    plt.show()

def grayscale_mask(path):
    mask_rgb = cv2.imread(path)

    # Convert to grayscale
    mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2GRAY)
    
    # Thresholding = 1
    _, single_channel_mask = cv2.threshold(mask_gray, 1, 1, cv2.THRESH_BINARY)
    # single_channel_mask = single_channel_mask / 255
    # single_channel_mask = np.where(single_channel_mask > 1, 1, 0)
    
    return single_channel_mask
    
def image_show(mask):
    plt.imshow(rle_decode(mask))
    plt.show()

masks = np.array(list(grayscale_mask(f"{input_path}{image}") for image in image_files))
# print(mask.shape)

mask_rle = list(multi_rle_encode(mask) for mask in masks)

# Read CSV 
df = pd.read_csv(csv_file)
# print(len(df))

# EncodedPixels = df['EncodedPixels']

adding = []
for image, str_rle in zip(image_files, mask_rle):
    if image not in df['ImageId'].values:
        if len(str_rle) > 0:
            for rle in str_rle:
                adding.append({'ImageId': image, 'EncodedPixels': rle})
        else: adding.append({'ImageId': image, 'EncodedPixels': None})

df_new = pd.DataFrame(adding)
df = pd.concat([df, df_new], ignore_index = True)
    
df.to_csv(csv_file, index=False)

'''
Read csv to decode masks
'''

masks = pd.read_csv(os.path.join(ship_dir, 'data.csv'))
# masks = masks.dropna(subset=['EncodedPixels'])

unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')

# List ImageId unique
image_ids = masks["ImageId"].unique()
# print(image_ids)

# Dataframe ImageId unique
image_ids_df = unique_img_ids['ImageId']
# print(image_ids_df)

masks = masks[masks["ImageId"].isin(image_ids)].reset_index(
                drop=True
            )

os.makedirs("./mask_output/", exist_ok=True)

filenames = [
            os.path.join(ship_dir, "mask_cvat", image_id) for image_id in image_ids
        ]
assert (
    len(filenames) == masks["ImageId"].nunique()
), "The number of filenames does not match the number of unique ImageIds"

for image_id in filenames:
    file_id = image_id.split("/")[-1]
    # print(file_id)
    print(image_id)
    mask = masks[masks["ImageId"] == file_id]["EncodedPixels"]
    # Image open
    # image_array = Image.open(os.path.join(input_image, file_id)).convert("RGB")
    
    # Cv2 open
    image_array = cv2.imread(os.path.join(input_image, file_id))
    
    mask = masks_as_image(mask)
    image_label = 0 if mask.sum() == 0 else 1
    # print(image_label)
    # imshow(np.array(image_array, dtype=np.uint8), mask)
    
    # Save using cv2
    cv2.imwrite(f"{output_path}{file_id}", mask_overlay(np.array(image_array, dtype=np.uint8), mask))


# # Encode and Decode to test
image_test = multi_rle_encode(grayscale_mask("./mask_cvat/ship14.jpg"))
# image_hello = cv2.imread("./input/ship14.jpg")
image_hello = Image.open("input/ship14.jpg").convert("RGB")

plt.subplot(1, 2, 1)
plt.imshow(masks_as_image(image_test))
plt.title("Mask")

plt.subplot(1, 2, 2) 
plt.imshow(image_hello)
plt.title("Image")

plt.show()

imshow(np.array(image_hello, dtype=np.uint8), masks_as_image(image_test))

# cv2.imwrite("./000155de5.jpg", mask_overlay(np.array(image_hello, dtype=np.uint8), masks_as_image(image_test)))