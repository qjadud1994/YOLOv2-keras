
VOC_LABELS = ['motorbike', 'car', 'person', 'bus', 'bird', 'horse', 'bicycle', 'chair', 'aeroplane', 'diningtable', 'pottedplant', 'cat', 'dog', 'boat', 'sheep', 'sofa', 'cow', 'bottle', 'tvmonitor', 'train']
COCO_LABELS = ['truck', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench',  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'bed',  'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

LABELS = VOC_LABELS + COCO_LABELS

img_dir = '/media/user/Store2/DB/'

voc_dir = "/media/user/Store2/DB/voc/VOCdevkit/VOC2012/"
voc_detection_train_list = voc_dir + 'voc_train_annotation2.txt'
voc_detection_val_list = voc_dir + 'voc_val_annotation2.txt'

voc_detection_test_list = '/media/user/Store2/DB/VOCtest_06-Nov-2007/VOC2007/voc_test_annotation.txt'
voc_testimg_dir = '/media/user/Store2/DB/VOCtest_06-Nov-2007/VOC2007/JPEGImages/'

coco_dir = "/media/user/Store2/DB/coco/"
coco_detection_train_list = coco_dir + 'coco_train_annotations2.txt'
coco_detection_val_list = coco_dir + 'coco_val_annotations2.txt'

input_shape = (416, 416, 3)
NORM_H, NORM_W = 416, 416
GRID_H, GRID_W = 13 , 13
BATCH_SIZE = 16
BOX = 5
CLASS = len(LABELS)
THRESHOLD = 0.40

SCORE_THRESHOLD = 0.005
IOU_THRESHOLD = 0.5
MAX_BOXES = 100

ANCHORS = [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]]
SCALE_NOOB, SCALE_CONF, SCALE_COOR, SCALE_PROB = 0.5, 5.0, 5.0, 1.0

#ANCHORS = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
#SCALE_NOOB, SCALE_CONF, SCALE_COOR, SCALE_PROB = 1.0, 5.0, 1.0, 1.0

MULTI_SCALE_INPUT = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
MULTI_SCALE_OUTPUT = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
