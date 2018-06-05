import tensorflow as tf
from keras.layers import Input, Add, Activation, BatchNormalization, Reshape, Lambda
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers import Layer
from keras.layers.merge import concatenate
from keras import backend as K
from Depthwise_conv import DepthwiseConv2D
from YOLO_parameter import CLASS, BOX


class BatchNorm(BatchNormalization):    # for fixed Batchnorm
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)


def Conv_layer(x, filters, kernel_size, strides, num, trainable):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', name='conv_' + num,
               use_bias=False, kernel_initializer='he_normal', trainable=trainable)(x)
    #x = BatchNorm(axis=-1, name='norm_' + num)(x)
    x = BatchNormalization(axis=-1, name='norm_' + num)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def DarkNet_deep(input_image, trainable=True):

    # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
    def space_to_depth_x2(x):
        return tf.space_to_depth(x, block_size=2)

    # Layer 1
    x = Conv_layer(input_image, 32, (3, 3), strides=(1, 1), num='1', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv_layer(x, 64, (3, 3), strides=(1, 1), num='2', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv_layer(x, 128, (3, 3), strides=(1, 1), num='3', trainable=trainable)
    x = Conv_layer(x, 64, (1, 1), strides=(1, 1), num='4', trainable=trainable)
    x = Conv_layer(x, 128, (3, 3), strides=(1, 1), num='5', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv_layer(x, 256, (3, 3), strides=(1, 1), num='6', trainable=trainable)
    x = Conv_layer(x, 128, (1, 1), strides=(1, 1), num='7', trainable=trainable)
    x = Conv_layer(x, 256, (3, 3), strides=(1, 1), num='8', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv_layer(x, 512, (3, 3), strides=(1, 1), num='9', trainable=trainable)
    x = Conv_layer(x, 256, (1, 1), strides=(1, 1), num='10', trainable=trainable)
    x = Conv_layer(x, 512, (3, 3), strides=(1, 1), num='11', trainable=trainable)
    x = Conv_layer(x, 256, (1, 1), strides=(1, 1), num='12', trainable=trainable)
    x = Conv_layer(x, 512, (3, 3), strides=(1, 1), num='13', trainable=trainable)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='14', trainable=trainable)
    x = Conv_layer(x, 512, (1, 1), strides=(1, 1), num='15', trainable=trainable)
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='16', trainable=trainable)
    x = Conv_layer(x, 512, (1, 1), strides=(1, 1), num='17', trainable=trainable)
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='18', trainable=trainable)
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='19', trainable=trainable)
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='20', trainable=trainable)

    # Layer 21
    skip_connection = Conv_layer(skip_connection, 64, (1, 1), strides=(1, 1), num='21', trainable=trainable)
    skip_connection = Lambda(space_to_depth_x2, name='space_to_depth')(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='22', trainable=trainable)

    return x


def DarkNet_tiny(input_tensor, trainable=True):
    x = Conv_layer(input_tensor, 16, (3, 3), strides=(1, 1), num='1', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv_layer(x, 32, (3, 3), strides=(1, 1), num='2', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv_layer(x, 64, (3, 3), strides=(1, 1), num='3', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv_layer(x, 128, (3, 3), strides=(1, 1), num='4', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv_layer(x, 256, (3, 3), strides=(1, 1), num='5', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv_layer(x, 512, (3, 3), strides=(1, 1), num='6', trainable=trainable)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='7', trainable=trainable)
    x = Conv_layer(x, 1024, (3, 3), strides=(1, 1), num='8', trainable=trainable)
    return x


def relu6(x):
    return K.relu(x, max_value=6)

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv%d' % block_id)(inputs)
    x = BatchNorm(axis=channel_axis, name='conv%d_bn' % block_id)(x)
    return Activation(relu6, name='conv%d_relu' % block_id)(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNorm(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNorm(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def DarkNet_mobile(img_input, alpha=1.0, depth_multiplier=1):

    x = _conv_block(img_input, 32, alpha)
    x = MaxPooling2D(pool_size = (2,2))(x)

    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    return x


def multi_yolo(x):
    x = Conv2D(5 * (CLASS + BOX), (1, 1), strides=(1, 1), name='conv_23')(x)
    x = Activation('linear', name="linear_activation")(x)

    shape = K.shape(x)
    final = Reshape((shape[1], shape[2], 5, CLASS + BOX), name="final_reshape")(x)
    return final


class OutputInterpreter(Layer):
    """
    Convert output features into predictions
    """

    def __init__(self, anchors, num_classes, **kwargs):
        super(OutputInterpreter, self).__init__(**kwargs)
        self.anchors = anchors
        self.num_classes = num_classes

    def build(self, input_shape):
        super(OutputInterpreter, self).build(input_shape)

    def call(self, output_features, **kwargs):
        shape = tf.shape(output_features)
        batch, height, width = shape[0], shape[1], shape[2]

        #  Create offset map
        cx = tf.reshape(tf.tile(tf.range(width), [height]), [-1, height, width, 1])
        cy = tf.tile(tf.expand_dims(tf.range(height), -1), [1, width])
        cy = tf.reshape(cy, [-1, height, width, 1])
        c_xy = tf.to_float(tf.stack([cx, cy], -1))
        anchors_tensor = tf.to_float(K.reshape(self.anchors, [1, 1, 1, len(self.anchors), 2]))
        output_size = tf.to_float(K.reshape([width, height], [1, 1, 1, 1, 2]))

        outputs = K.reshape(output_features, [batch, height, width, len(self.anchors), self.num_classes + 5])

        # Interpret outputs
        box_xy = K.sigmoid(outputs[..., 0:2]) + c_xy
        box_wh = K.exp(outputs[..., 2:4]) * anchors_tensor
        box_confidence = K.sigmoid(outputs[..., 4:5])
        box_class_probs = K.softmax(outputs[..., 5:])

        # Convert coordinates to relative coordinates (percentage)
        box_xy = box_xy / output_size
        box_wh = box_wh / output_size

        # Calculate corner points of bounding boxes
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)

        # Y1, X1, Y2, X2
        boxes = K.concatenate([box_mins[..., 1:2],
                               box_mins[..., 0:1],  # Y1 X1
                               box_maxes[..., 1:2],
                               box_maxes[..., 0:1]], axis=-1)  # Y2 X2

        outputs = K.concatenate([boxes, box_confidence, box_class_probs], axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], input_shape[1], input_shape[2], len(self.anchors), 5 + self.num_classes])

    def get_config(self):
        config = {'anchors': self.anchors,
                  'num_classes': self.num_classes}
        base_config = super(OutputInterpreter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PostProcessor(Layer):
    """
    Perform Non-Max Suppression to calculate prediction
    """

    def __init__(self, score_threshold, iou_threshold, max_boxes=1000, **kwargs):
        super(PostProcessor, self).__init__(**kwargs)

        self.max_boxes = max_boxes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def build(self, input_shape):
        super(PostProcessor, self).build(input_shape)

    def call(self, inputs, **kwargs):
        boxes = inputs[..., 0:4]
        box_confidence = inputs[..., 4:5]
        box_class_probs = inputs[..., 5:]

        box_scores = box_confidence * box_class_probs
        box_classes = K.argmax(box_scores, -1)

        box_class_scores = K.max(box_scores, -1)
        prediction_mask = (box_class_scores >= self.score_threshold)

        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)

        nms_index = tf.image.non_max_suppression(boxes,
                                                 scores,
                                                 max_output_size=self.max_boxes,
                                                 iou_threshold=self.iou_threshold)
        boxes = K.gather(boxes, nms_index)
        scores = tf.gather(scores, nms_index)
        classes = tf.gather(classes, nms_index)

        return [boxes, scores, classes]

    def compute_output_shape(self, input_shape):
        return [(None, 4), (None, 1), (None, 1)]

    def get_config(self):
        config = {'score_threshold': self.score_threshold,
                  'iou_threshold': self.iou_threshold,
                  'max_boxes': self.max_boxes}
        base_config = super(PostProcessor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
