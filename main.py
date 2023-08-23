import sys
import time
import ctypes
import cv2
import keyboard
import numpy as np
import argparse
import pyautogui
from mss import mss
import onnxruntime as rt
import threading
import pygetwindow


# 定义鼠标事件类型常量
MOUSEEVENTF_MOVE = 0x0001

# 定义鼠标输入结构体
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

# 定义输入联合体
class InputUnion(ctypes.Union):
    _fields_ = [("mi", MouseInput),
                ("ki", ctypes.c_ulonglong),
                ("hi", ctypes.c_ulonglong)]

# 定义输入结构体
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("union", InputUnion)]

class main():
    def __init__(self, confThreshold=0.3, nmsThreshold=0.4):
        self.classes = list(map(lambda x: x.strip(), open('coco.names',
                                                          'r').readlines()))  ###这个是在coco数据集上训练的模型做opencv部署的，如果你在自己的数据集上训练出的模型做opencv部署，那么需要修改self.classes
        self.inpWidth = 512
        self.inpHeight = 512
        providers = ['CPUExecutionProvider']
        # 新增代码，此处默认使用CPU，可以根据需要修改，比如使用GPU，修改为providers = ['CUDAExecutionProvider']，
        # providers = ['OpenVINOExecutionProvider']
        # providers = ['TensorrtExecutionProvider']
        # providers = ['DnnlExecutionProvider']

        #注意，需要自行下载对应驱动！！！！！！！！

        self.sess = rt.InferenceSession('model.onnx', providers=providers, provider_options=[{'device_type' : 'GPU_FP32'}])

        self.input_name = self.sess.get_inputs()[0].name # 新增代码
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.H, self.W = 32, 32
        self.grid = self._make_grid(self.W, self.H)
        self.detectTime = 0  # 新增代码

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        # 获取屏幕分辨率
        screen_width, screen_height = pyautogui.size()
        # 计算屏幕中心点坐标
        screen_center_x = screen_width // 2
        screen_center_y = screen_height // 2

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId] * detection[0]
            if confidence > self.confThreshold:
                center_x = int(detection[1] * frameWidth)
                center_y = int(detection[2] * frameHeight)
                width = int(detection[3] * frameWidth)
                height = int(detection[4] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        if len(indices) == 0:
            return frame
        indices = np.array(indices).flatten()

        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            if classIds[i] == 0:  # 检查类别 ID 是否为人类
                x_center = left + width // 2  # 计算边界框的中心点坐标
                y_center = top + height // 2  # 计算边界框的中心点坐标

                # 计算目标与屏幕中心点的方向和距离
                delta_x = x_center - screen_center_x
                delta_y = y_center - screen_center_y

                # 计算实际应该鼠标移动的距离
                delta_x = delta_x / (1 + sensitivity / 100.0)
                delta_y = delta_y / (1 + sensitivity / 100.0)

                # 创建鼠标输入结构体实例
                mouse_input = MouseInput(int(delta_x), int(delta_y), 0, MOUSEEVENTF_MOVE, 0, None)

                # 创建输入联合体实例
                input_union = InputUnion()
                input_union.mi = mouse_input

                # 创建输入结构体实例
                input = Input(ctypes.c_ulong(0), input_union)

                # 调用 SendInput 函数模拟鼠标移动
                ctypes.windll.user32.SendInput(1, ctypes.pointer(input), ctypes.sizeof(input))


            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return frame

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def detect(self, srcimg):

        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight))
        t1 = cv2.getTickCount()  # 新增代码
        pred = self.sess.run(None, {self.input_name: blob})[0]  # 新增代码
        t2 = cv2.getTickCount()  # 新增代码
        self.detectTime = (t2 - t1) / cv2.getTickFrequency()  # 新增代码
        pred[:, 3:5] = self.sigmoid(pred[:, 3:5])  ###w,h
        pred[:, 1:3] = (np.tanh(pred[:, 1:3]) + self.grid) / np.tile(np.array([self.W,self.H]), (pred.shape[0], 1)) ###cx,cy

        srcimg = self.postprocess(srcimg, pred)
        print('本轮计算时间: %.2f ms' % (self.detectTime * 1000))  # 新增代码：将计算时间打印到控制台
        return srcimg


import threading
import time

paused = False


def start_recognition():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confThreshold', default=0.8, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.35, type=float, help='nms iou thresh')
    args = parser.parse_args()
    model = main(confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)
    sct = mss()
    monitor = sct.monitors[1]
    while True:
        if not paused:
            frame = sct.grab(monitor)
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = model.detect(frame)


def toggle_pause():
    global paused
    paused = not paused


def check_key():
    while True:
        keyboard.wait('backspace')
        toggle_pause()


if __name__ == '__main__':
    print("按下backspace后暂停检测，再次按下后继续")
    # 询问用户游戏的灵敏度
    sensitivity = float(input("请输入游戏的灵敏度（0-100）："))


    t1 = threading.Thread(target=start_recognition)
    t1.start()

    t2 = threading.Thread(target=check_key)
    t2.start()

    t1.join()
    t2.join()