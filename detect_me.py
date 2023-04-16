import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (Profile, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.torch_utils import smart_inference_mode


@smart_inference_mode()
def run(
        weights=r'./pts/2023-4-5-2.pt',  # model path or triton URL
        source=r'./images_delect/1.png',  # file/dir/URL/glob/screen/0(webcam)
        data='',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
):
    source = str(source)

    # Load model
    device = torch.device('cuda:0')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt

    # Dataloader
    bs = 1  # batch_size

    path = source
    im0s = cv2.imread(path)  # BGR
    print(type(im0s))
    im = letterbox(im0s, imgsz, stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    with dt[0]:

        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

    max_like = 0
    max_people = None
    for i, det in enumerate(pred):  # per image
        seen += 1
        p, im0 = path, im0s.copy()

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                print(f"xywh:{xywh}")
                print(f"conf:{conf}")
                if conf > max_like:
                    max_like = conf
                    max_people = xywh
    print(f"max_like:{max_like}")
    print(f"max_people:{max_people}")
    return max_people


def run_for_me(
        array,  # file/dir/URL/glob/screen/0(webcam)
        model,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
):
    bs = 1  # batch_size
    pt = model.pt
    # path = source
    im0s = array  # BGR
    print(type(im0s))
    im = letterbox(im0s, imgsz, stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    with dt[0]:

        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

    max_like = 0
    max_people = None
    for i, det in enumerate(pred):  # per image
        seen += 1
        im0 = im0s.copy()

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                print(f"xywh:{xywh}")
                print(f"conf:{conf}")
                if conf > max_like:
                    max_like = conf
                    max_people = xywh
    print(f"max_like:{max_like}")
    print(f"max_people:{max_people}")
    return max_people


# @smart_inference_mode()
def run_for_me2():
    weights = "./pts/2023-4-5-2.pt"
    weights = os.path.abspath(weights)
    device = torch.device('cuda:0')
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    # print(f"model:{model}")

    ret = run_for_me(cv2.imread('./images_delect/1.png'), model)
    print(ret)


if __name__ == '__main__':
    run()
    print("_" * 50)
    run_for_me2()
