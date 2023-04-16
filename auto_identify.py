import ctypes
import os
import sys
from pathlib import Path
from tkinter import Tk, Frame, Label

import win32api
import win32con
from pynput import keyboard, mouse
from pynput.mouse import Button

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
    bg = 'yellow'
    fg = 'red'
    left = 'left'
    font_name = '黑体'
    font_size = 16  # 字体大小
    font_choose = 'bold'

    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)  # 获得屏幕分辨率X轴
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)  # 获得屏幕分辨率Y轴

    def __init__(self, pt_path="./pts/2023-4-5-2.pt"):
        self.pt_path = pt_path

        self.executor = ThreadPoolExecutor(max_workers=5)  # 初始化线程池
        self.model = self.__load_mode()  # 加载模型 todo 测试后恢复

        self.lib = None
        self.handler = None
        self.lib, self.handler = self.load_usb()

        self.camera = dxcam.create()
        self.__start_camera()  # 启动截图

        self.win = self.create_win()

        self.switch = False
        self.switch_label = self.__pack_switch_label()

        self.people_x = None
        self.people_y = None
        self.people_label = self.__pack_people_label()

        self.listener_keyboard()  # 监听键盘事件
        self.listener_mouse()#监听鼠标事件

    def load_usb(self):
        path = "box64.dll"
        path = os.path.join(os.path.dirname(__file__), path)
        lib = ctypes.windll.LoadLibrary(path)

        lib.M_Open.restype = ctypes.c_uint64
        ret = lib.M_Open(1)
        if ret in [-1, 18446744073709551615]:
            print('未检测到 USB 芯片!')
            os._exit(0)
        handler = ctypes.c_uint64(ret)
        result = lib.M_ResolutionUsed(handler, self.screen_width, self.screen_height)
        if result != 0:
            print('设置分辨率失败!')
            os._exit(0)
        print("加载USB成功!!!")
        return lib, handler

    def __show_switch(self):
        return f"开启状态:{'开启' if self.switch else '关闭'}"

    def __show_people(self):
        return f"人物位置:({self.people_x},{self.people_y})"

    def __start_camera(self):
        self.camera.start()
        print("启动camera")

    def __identity(self):
        print("开始识别")
        while True:
            img = self.camera.get_latest_frame()
            x, y = self.__detection_target(img, self.model)
            if x and y:
                # print(f"x,y:{x},{y}")
                self.people_x = x
                self.people_y = y
                self.people_label.config(text=self.__show_people())

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
        print("camera stopped")

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

    def create_win(self):
        win = Tk()
        width = 1200  # tkinter宽度
        distance_middle = 300  # 中间偏右多少
        x = int((self.screen_width - width) / 2) + distance_middle
        win.geometry(f'{width}x30+{x}+0')  # 设置宽度300,高度300,距离左上角x轴距离为500,y轴距离为100
        win.attributes('-alpha', 0.3)  # 设置透明度,数值是0-1之间的小数,包含0和1
        win.attributes("-fullscreen", False)  # 设置全屏
        win.attributes("-topmost", True)  # 设置窗体置于最顶层
        win.update()  # 刷新窗口,否则获取的宽度和高度不准
        win.overrideredirect(True)  # 去除窗口边框
        return win

    def start_win(self):
        self.win.mainloop()

    def __pack_switch_label(self):
        win = self.win
        frame = Frame(win, bg=self.bg)
        frame.pack(fill='both', side=self.left)
        label = Label(
            master=win,  # 父容器
            text=self.__show_switch(),  # 文本
            bg=self.bg,  # 背景颜色
            fg=self.fg,  # 文本颜色
            font=(self.font_name, self.font_size, self.font_choose),
        )
        label.pack(side=self.left)
        return label

    def __pack_people_label(self):
        win = self.win
        frame = Frame(win, bg=self.bg)
        frame.pack(fill='both', side=self.left)
        label = Label(
            master=win,  # 父容器
            text=self.__show_people(),  # 文本
            bg=self.bg,  # 背景颜色
            fg=self.fg,  # 文本颜色
            font=(self.font_name, self.font_size, self.font_choose),
        )
        label.pack(side=self.left)
        return label

    def keyboard_press(self, key):
        if hasattr(key, 'vk') and key.vk == 97:  # 小键盘1
            self.switch = not self.switch
            self.switch_label.config(text=self.__show_switch())
        elif hasattr(key, 'vk') and key.vk == 103:  # 小键盘7
            self.stop_camera()
            os._exit(0)  # 强制所有线程都退出

    def __listener_keyboard(self):
        listener = keyboard.Listener(on_press=self.keyboard_press)
        listener.start()
        listener.join()

    def listener_keyboard(self):
        self.executor.submit(self.__listener_keyboard)
    def __listener_mouse(self):
        listener = mouse.Listener(on_click=self.mouse_click)
        listener.start()
        listener.join()

    def mouse_click(self,x, y, button: Button, pressed):
        """
        鼠标点击事件
        :param x: 横坐标
        :param y: 纵坐标
        :param button: 按钮枚举对象 Button.left 鼠标左键 Button.right 鼠标右键 Button.middle 鼠标中键
        :param pressed: 按下或者是释放,按下是True释放是False
        :return:
        """
        if pressed and button == Button.left:
            # print("鼠标左键按下")
            pass

        elif not pressed and button == Button.left:
            # print("鼠标左键释放")
            pass


    def listener_mouse(self):
        self.executor.submit(self.__listener_mouse)


def main():
    me = Mine()
    me.identity()
    me.start_win()
    print("结束")


if __name__ == '__main__':
    main()
