import cv2
import numpy as np
from keras.models import Input, Model
from Model import *
from YOLO_parameter import *
from YOLO_utils import parse_annotation, compute_ap, compute_overlap

# make yolo model
inputs = Input((416, 416, 3))

darknet = DarkNet_deep(inputs, trainable=True)
#darknet = DarkNet_tiny(inputs, trainable=True)
#darknet = DarkNet_mobile(inputs, trainable=True)

yolo_output = multi_yolo(darknet)

yolo_post = OutputInterpreter(anchors=ANCHORS, num_classes=CLASS)(yolo_output)
boxes, scores, classes = PostProcessor(score_threshold=SCORE_THRESHOLD,
                                       iou_threshold=IOU_THRESHOLD,
                                       max_boxes=MAX_BOXES,
                                       name="NonMaxSuppression")(yolo_post)

# make yolo model
yolo_model = Model(inputs, [boxes, scores, classes])

yolo_model.load_weights("./weights/DeepYOLO.hdf5")

# get test imageset
eval_imgs = parse_annotation(voc_detection_test_list)

all_detections = [[None for i in range(CLASS)] for j in range(len(eval_imgs))]
all_annotations = [[None for i in range(CLASS)] for j in range(len(eval_imgs))]

for i in range(len(eval_imgs)):
    path = eval_imgs[i]['filename']
    raw_image = cv2.imread(voc_testimg_dir + path)
    raw_height, raw_width, _ = raw_image.shape

    # make the boxes and the labels
    try_image = cv2.resize(raw_image, (416, 416))
    try_image = cv2.cvtColor(try_image, cv2.COLOR_BGR2RGB)
    try_image = try_image / 255.
    try_image = np.expand_dims(try_image, axis=0)

    pred_boxes, score, pred_labels = yolo_model.predict_on_batch(try_image)

    if len(pred_boxes) > 0:  # pred_boxes = [ymin, xmin, ymax, xmax]
        pred_boxes = np.array(
            [[box[1] * raw_width, box[0] * raw_height,
              box[3] * raw_width, box[2] * raw_height,
              score[a]] for a, box in enumerate(pred_boxes)])
    else:
        pred_boxes = np.array([[]])

    # sort the boxes and the labels according to scores
    score_sort = np.argsort(-score)
    pred_labels = pred_labels[score_sort]
    pred_boxes = pred_boxes[score_sort]

    # copy detections to all_detections
    for label in range(CLASS):
        all_detections[i][label] = pred_boxes[pred_labels == label, :]

    annotations = eval_imgs[i]['object'][:]

    annots = []
    for obj in annotations:
        annot = [int(obj['xmin']), int(obj['ymin']), int(obj['xmax']),
                 int(obj['ymax']), LABELS.index(obj['name'])]
        annots += [annot]

    annotations = np.array(annots)

    # copy detections to all_annotations
    for label in range(CLASS):
        all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

# compute mAP by comparing all detections and all annotations
average_precisions = {}
recalls = {}

for label in range(CLASS):
    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    scores = np.zeros((0,))
    num_annotations = 0.0

    for i in range(len(eval_imgs)):
        detections = all_detections[i][label]
        annotations = all_annotations[i][label]
        num_annotations += annotations.shape[0]
        detected_annotations = []

        for d in detections:
            scores = np.append(scores, d[4])

            if annotations.shape[0] == 0:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)
                continue

            overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]

            if max_overlap >= IOU_THRESHOLD and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
            else:
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)

    # no annotations -> AP for this class is 0 (is this correct?)
    if num_annotations == 0:
        average_precisions[label] = 0
        continue

    # sort by score
    indices = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives / num_annotations
    recalls[label] = recall
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = compute_ap(recall, precision)
    average_precisions[label] = average_precision


print("average_precisions : ", average_precisions)
print("recalls : ", recalls)
