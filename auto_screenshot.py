import os
import time

import dxcam
import torch

from detect_me import run_for_me
from models.common import DetectMultiBackend

camera = dxcam.create()
# c=camera.get_latest_frame()
# print(c)
camera.start()
weights = "./pts/2023-4-5-2.pt"
weights = os.path.abspath(weights)
device = torch.device('cuda:0')
model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
print(f"model:{model}")
while True:
    image = camera.get_latest_frame()
    print(type(image))
    # print(image)
    run_for_me(image,model)
    # break