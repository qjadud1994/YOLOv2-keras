import tensorflow as tf
import numpy as np
import keras.backend as K
from YOLO_parameter import *


def multi_yolo_loss(y_true, y_pred):
    GRID =  K.cast(K.shape(y_true)[1], dtype='float32')

    ### yolo loss
    # adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[:, :, :, :, :2])

    # adjust w and h
    pred_box_wh = tf.exp(y_pred[:, :, :, :, 2:4]) * tf.reshape(ANCHORS, [1, 1, 1, BOX, 2])
    pred_box_wh = tf.sqrt(pred_box_wh / tf.reshape([GRID, GRID], [1, 1, 1, 1, 2]))

    # adjust confidence
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)

    # adjust probability
    pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])

    y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)

    ### Adjust ground truth
    # adjust x and y
    center_xy = .5 * (y_true[:, :, :, :, 0:2] + y_true[:, :, :, :, 2:4])
    center_xy = center_xy / tf.reshape([GRID*32.0 / GRID, GRID*32.0 / GRID], [1, 1, 1, 1, 2])
    true_box_xy = center_xy - tf.floor(center_xy)

    # adjust w and h
    true_box_wh = (y_true[:, :, :, :, 2:4] - y_true[:, :, :, :, 0:2])
    true_box_wh = tf.sqrt(true_box_wh / tf.reshape([GRID*32.0, GRID*32.0], [1, 1, 1, 1, 2]))

    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * tf.reshape([GRID, GRID], [1, 1, 1, 1, 2])
    pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

    true_tem_wh = tf.pow(true_box_wh, 2) * tf.reshape([GRID, GRID], [1, 1, 1, 1, 2])
    true_box_area = true_tem_wh[:, :, :, :, 0] * true_tem_wh[:, :, :, :, 1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh

    intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

    iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
    best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
    best_box = tf.to_float(best_box)
    true_box_conf = tf.expand_dims(best_box * y_true[:, :, :, :, 4], -1)

    # adjust confidence
    true_box_prob = y_true[:, :, :, :, 5:]

    y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)

    ### Compute the weights
    weight_coor = tf.concat(4 * [true_box_conf], 4)
    weight_coor = SCALE_COOR * weight_coor

    weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_CONF * true_box_conf

    weight_prob = tf.concat(CLASS * [true_box_conf], 4)
    weight_prob = SCALE_PROB * weight_prob

    weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)

    ### Finalize the loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight

    loss_shape = tf.shape(loss)
    loss = tf.reshape(loss, [-1, loss_shape[1] * loss_shape[2] * loss_shape[3] * loss_shape[4]])

    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)

    return loss
