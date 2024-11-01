import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse

import torch
import os

import bentoml
from src.models.components.lossbinary import LossBinary
from src.models.components.lovasz_loss import LovaszLoss
from src.models.components.unet import UNet
from src.models.components.unet34 import Unet34
from src.models.unet_module import UNetLitModule
from src.models.classifier_module import ResNetLitModule
from src.models.components.resnet34 import ResNet34_Binary



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # UNET Model
    # model = UNetLitModule.load_from_checkpoint(
    #     "unet34.ckpt",
    #     net=UNet(),
    #     criterion=LossBinary(),
    #     map_location=torch.device(device),
    # )
    
    # Make dirs
    # os.makedirs("model_repository/unet/1/", exist_ok=True)
    # model = model.to_torchscript("model_repository/unet34/1/model.pt")
    
    # Check type model
    # model = torch.jit.load('model.pt')
    # print(type(model))
    
    # Resnet34 Model
    classifier_model = ResNetLitModule.load_from_checkpoint(
        "resnet34.ckpt",
        net=ResNet34_Binary(),
        map_location=torch.device(device),
    )
    
    os.makedirs("model_repository/resnet34/1/", exist_ok=True)
    classifier_model = classifier_model.to_torchscript('model_repository/resnet34/1/model.pt')
    
    classifier_model = torch.jit.load('resnet34.pt')
    print(type(classifier_model))


if __name__ == "__main__":
    main()
