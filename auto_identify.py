import os
import sys
from pathlib import Path

import win32api
import win32con

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import (Profile, non_max_suppression, scale_boxes, xyxy2xywh)
import os
from concurrent.futures import ThreadPoolExecutor

import dxcam
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox


class Mine:
    def __init__(self, pt_path="./pts/2023-4-5-2.pt"):
        self.pt_path = pt_path
        self.executor = ThreadPoolExecutor(max_workers=5)  # 初始化线程池
        self.model = self.__load_mode()  # 加载模型

        self.camera = dxcam.create()
        self.__start_camera()  # 启动截图
        self.screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)  # 获得屏幕分辨率X轴

        self.screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)  # 获得屏幕分辨率Y轴

    def __start_camera(self):
        self.camera.start()
        print("启动camera")

    def __identity(self):
        print("开始识别")
        while True:
            img = self.camera.get_latest_frame()
            # print(img)
            x, y = self.__detection_target(img, self.model)
            if x and y:
                print(f"x,y:{x},{y}")

    def identity(self):
        self.executor.submit(self.__identity)

    def __load_mode(self):
        weights = self.pt_path
        weights = os.path.abspath(weights)
        device = torch.device('cuda:0')
        model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
        print(f"加载模型完成:{weights}")
        return model

    def stop_camera(self):
        self.camera.stop()

    def __detection_target(self,
                           array,  # file/dir/URL/glob/screen/0(webcam)
                           model,
                           imgsz=(640, 640),  # inference size (height, width)
                           conf_thres=0.25,  # confidence threshold
                           iou_thres=0.45,  # NMS IOU threshold
                           max_det=1000,  # maximum detections per image
                           ):
        bs = 1  # batch_size
        pt = model.pt
        im0s = array  # BGR
        # print(type(im0s))
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
        # print(f"pred:{pred}")
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0 = im0s.copy()

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # print(f"xywh:{xywh}")
                    # print(f"conf:{conf}")
                    if conf > max_like:
                        max_like = conf
                        max_people = xywh
        # print(f"max_like:{max_like}")
        # print(f"max_people:{max_people}")
        if max_people:
            # print(f"max_people:{max_people}")
            x = int(self.screen_width * max_people[0])
            y = int(self.screen_height * max_people[1])
            return x, y
        return None, None


    def my_tk(self):
        pass

def main():
    me = Mine()

    me.identity()
    # me.executor.submit(me.get_img_by_camera, )
    print("结束")
    # time.sleep(10)


if __name__ == '__main__':
    main()
