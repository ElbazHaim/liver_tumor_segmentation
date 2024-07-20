import torch
from monai.networks.nets.unetr import UNETR

model = torch.jit.load(
    "/home/haim/code/tumors/liver_tumors/models/UNETR_model_best_acc.pt"
)
print(model)
