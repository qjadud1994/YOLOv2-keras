import cv2, time
import numpy as np
from keras.models import Input, Model
from Model import *
from YOLO_parameter import *
from YOLO_utils import draw_boxes

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

cap = cv2.VideoCapture("./video/drive1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    original = cv2.resize(frame, (416, 416))

    try_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    try_image = try_image / 255.
    try_image = np.expand_dims(try_image, axis=0)
    start = time.time()
    boxes, scores, classes = yolo_model.predict_on_batch(try_image)

    classes = [LABELS[idx] for idx in classes]
    boxes = [box * np.array([416, 416, 416, 416]) for box in boxes]
    yolo_result = draw_boxes(original, boxes, classes, scores)

    end = time.time()

    yolo_result = cv2.putText(yolo_result, "FPS : %.1f" % (1.0 / (end - start)),
                              (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(20, 20, 20), 4)

    cv2.imshow('detect', yolo_result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()