import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import time
from keras.models import Input, Model
from keras.utils import generic_utils
from keras.optimizers import Adam

from Model import *
from YOLO_parameter import *
from YOLO_utils import parse_annotation, multi_data_gen
from Losses import multi_yolo_loss

# train dataset (voc + coco)
voc_train_set = parse_annotation(voc_detection_train_list)
coco_train_set = parse_annotation(coco_detection_train_list)
yolo_train_set = voc_train_set + coco_train_set

# val dataset (voc + coco)
voc_val_set = parse_annotation(voc_detection_val_list)
coco_val_set = parse_annotation(coco_detection_val_list)
yolo_val_set = voc_val_set + coco_val_set

# make data generator
yolo_train_img = multi_data_gen(yolo_train_set, BATCH_SIZE)
yolo_val_img = multi_data_gen(yolo_val_set, BATCH_SIZE)


print("create model...")

inputs = Input((None, None, 3))

darknet = DarkNet_deep(inputs, trainable=True)
#darknet = DarkNet_tiny(inputs, trainable=True)
#darknet = DarkNet_mobile(inputs)

yolo_output = multi_yolo(darknet)

# make yolo model
yolo_model = Model(inputs, yolo_output)
yolo_model.summary()
####

# transfer learning -> Converts the last convolution to random weight.
'''
yolo_model.load_weights("./weights/DeepYOLO.hdf5")

layer = model.layers[-3]
weights = layer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape) / (GRID_H * GRID_W)
new_bias = np.random.normal(size=weights[1].shape) / (GRID_H * GRID_W)

layer.set_weights([new_kernel, new_bias])
'''

# compile
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
yolo_model.compile(loss=multi_yolo_loss, optimizer=adam, metrics=["accuracy"])

# set training parameters
epoch_length = len(yolo_train_set) // BATCH_SIZE
num_epochs = 400
iter_num = 0
early_stop_count=0

losses = np.zeros(epoch_length)
accuracy = np.zeros(epoch_length)
start_time = time.time()
model_path = "./weights/DeepYOLO.hdf5"

best_acc = -np.Inf
best_loss = np.Inf

print('Starting training')
for epoch_num in range(num_epochs):

    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    while True:
        X, Y = next(yolo_train_img)
        loss_yolo, acc_yolo = yolo_model.train_on_batch(X, Y)

        losses[iter_num] = loss_yolo
        accuracy[iter_num] = acc_yolo
        iter_num += 1

        progbar.update(iter_num,
                       [('yolo_loss', np.round(np.mean(losses[:iter_num]), 4)),
                        ('yolo_acc', np.round(np.mean(accuracy[:iter_num]), 4))])

        if iter_num == epoch_length:
            loss_yolo = np.mean(losses)
            acc_yolo = np.mean(accuracy)
            print('Loss YOLO: {} , Acc YOLO: {} , Elapsed time: {}'.
                  format(np.round(loss_yolo, 4), np.round(acc_yolo, 4), round(time.time()-start_time), 4))

            iter_num = 0

            yolo_val_outs = yolo_model.evaluate_generator(
                yolo_val_img,
                len(yolo_val_set) // BATCH_SIZE,
                max_queue_size=3)

            print('val_loss YOLO: {}  ,  val_acc YOLO: {}'.
                  format(np.round(yolo_val_outs[0], 4), np.round(yolo_val_outs[1], 4)))

            start_time = time.time()
            curr_acc = yolo_val_outs[1]

            if curr_acc > best_acc:
                print('val acc increased from {} to {}, saving weights {}'.
                      format(np.round(best_acc,4), np.round(curr_acc,4), model_path))
                best_acc = curr_acc
                yolo_model.save_weights(model_path)
                early_stop_count=0
            else:
                early_stop_count+=1
                if early_stop_count > 30:
                    print("early stop!")
                    exit()
            break

print('Training complete, exiting.')
