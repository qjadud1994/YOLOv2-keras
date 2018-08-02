# YOLO v2

**YOLO** is a real-time object detection model based on deep learning. <br>
yolov2 (or YOLO9000) was implemented as keras (tensorflow backend).


## Backbone
1. DarkNet19
2. DarkNet tiny
3. MobileNet v1
- - -

## Todo list:
- [x] DarkNet19, DarkNet tiny, MobileNet v1
- [ ] SqueezeNet, ResNet, DenseNet backbone
- [x] Multiscale training
- [x] mAP Evaluation
- [x] PostProcessing for real-time process
- [ ] Warmup training
- [ ] revise GT generator (I think this is the cause of the lower performance than the existing model)


## How to Training

I have created an annotation file of the form <br/><br/>

[file_name / img_wigth / img_height / xmin / ymin / xmax / ymax / class]<br/>
2007_001185.jpg 500 375 197 199 289 323 cat <br/>
2007_001185.jpg 500 375 78 78 289 375 person <br/>
2007_001185.jpg 500 375 204 223 500 375 diningtable <br/>
2007_001185.jpg 500 375 452 131 500 253 bottle <br/>
2007_001763.jpg 500 375 281 119 500 375 cat <br/>
2007_001763.jpg 500 375 1 24 330 366 dog <br/>
2007_001763.jpg 500 375 1 48 500 375 sofa <br/>
2007_001763.jpg 500 375 83 1 195 16 tvmonitor <br/>

[annotation example](https://github.com/qjadud1994/YOLOv2-keras/blob/master/result/annotation.txt)<br/>

Then change the image directory and annotation directory in [YOLO_parameter.py](https://github.com/qjadud1994/YOLOv2-keras/blob/master/YOLO_parameter.py)  and run [YOLO_train.py](https://github.com/qjadud1994/YOLOv2-keras/blob/master/YOLO_train.py)


### Result
![enter image description here](https://github.com/qjadud1994/YOLOv2-keras/blob/master/result/yolo%20test.jpg)

It works well and is processed in real time on the GTX-1080.
<br><br><br>
And you can see the results of YOLO in the **video**.<br>
**Click** on the link below to see three images combined. <br>
From the left is YOLO using DarkNet19, DarkNet tiny, MobileNet. <br>

[YOLOv2 Demo](https://youtu.be/s3KO7YEkniQ)

- - -

## File Description

os : Ubuntu 16.04.4 LTS <br>
GPU : GeForce GTX 1080 (8GB) <br>
Python : 3.5.2 <br>
Tensorflow : 1.5.0 <br>
Keras : 2.1.3 <br>
CUDA, CUDNN : 9.0, 7.0 <br>

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
