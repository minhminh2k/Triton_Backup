import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse

import torch

import bentoml
from src.models.components.lossbinary import LossBinary
from src.models.components.lovasz_loss import LovaszLoss
from src.models.components.unet import UNet
from src.models.components.unet34 import Unet34
from src.models.unet_module import UNetLitModule

# Create PyTorch Model Object
model = UNetLitModule(net = UNet(), optimizer = torch.optim.Adam,
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,criterion = LossBinary())

# Load model weights from external file
state = torch.load("model.pt")

print(type(state))

state = {key.replace("module.", ""): value for key, value in state.items()}

model.load_state_dict(state)
print(type(model))

model = model.to_torchscript()

# Create ONNX file by tracing model
trace_input = torch.randn(1, 3, 256, 256)

# torch.onnx.export(model, trace_input, "model.onnx", verbose=True)
# traced_model = torch.jit.trace(model, torch.randn(1,3,256,256).to("cuda"))
# torch.jit.save(traced_model, "model.pt")

torch.jit.save(model.state_dict(), "model.pt")
