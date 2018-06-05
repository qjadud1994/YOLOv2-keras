# YOLO v2

**YOLO** is a real-time object detection model based on deep learning.
yolov2 (or YOLO9000) was implemented as keras (tensorflow backend).


## Backbone
1. DarkNet19
2. DarkNet tiny
3. MobileNet v1
- - -

## Todo list:
- [x] DarkNet19, DarkNet tiny, MobileNet v1
- [ ] SqueezeNet, ResNet, DenseNet backends
- [x] Multiscale training
- [x] mAP Evaluation
- [x] PostProcessing for real-time process
- [ ] Warmup training




### Result
![enter image description here](https://github.com/qjadud1994/YOLOv2-keras/blob/master/result/yolo%20test.jpg)
It works well and is processed in real time on the GTX-1080.

And you can see the results of YOLO in the video.
Click on the link below to see three images combined.
From the left is YOLO using DarkNet19, DarkNet tiny, MobileNet.
[YOLOv2 Demo](https://github.com/qjadud1994/YOLOv2-keras/blob/master/result/yolo%20demo.avi)

- - -

## File Description

os : Ubuntu 16.04.4 LTS
GPU : GeForce GTX 1080 (8GB)
Python : 3.5.2
Tensorflow : 1.5.0
Keras : 2.1.3
CUDA, CUDNN : 9.0, 7.0
|       File         |Description                                                   |
|----------------|--------------------------------------------------|
|Depthwise_conv .py  |  For MobileNet            |
|Losses. py |  YOLO v2 Loss function            |
|Model. py | YOLO v2 Model <br> (DarkNet19, DarkNet tiny, MobileNet) |
|YOLO_eval. py | Performance evaluation (mAP and recall)  |
|YOLO_parameter. py | Parameters used in YOLO v2 |
|YOLO_pred. py | Run YOLO v2 on video  |
|YOLO_train. py | YOLO v2 training |
|YOLO_utils. py | Utils used in YOLO v2|

- - -

## Training Result:

| Test  | with this implementation | on released weights |
|:---------------:|:-------------:|:-------------:|
| VOC2007 test    | mAP 66.2% <br> Recall 79.2%|    mAP 67.6% <br> Racall 77.5% |