"""
python .\train.py --batch-size 36 --epochs 100 --data D:\Debug-Conda\yolov5\data\people.yaml --weight .\weights\yolov5s.pt --cache ram --device 0


python .\detect.py --source D:\GitCode\yolo5-me\images_delect\96.png --weights D:\GitCode\yolo5-me\pts\2023-4-5-2.pt   --device 0 --save-txt

"""
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print("-"*50)
import win32api
import win32con

#0 0.626562 0.283333 0.0132813 0.0555556
width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)   #获得屏幕分辨率X轴

height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)   #获得屏幕分辨率Y轴
print(f'width:{width}')
print(f'height:{height}')
print(width*0.626562)
print(height*0.283333)
