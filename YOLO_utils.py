import numpy as np
import copy, random
import cv2
from PIL import Image, ImageDraw, ImageFont
from YOLO_parameter import *


def parse_annotation(ann_dir):
    f = open(ann_dir, 'r')
    _f = f.read()
    f_content = _f.split('\n')

    all_img = []
    current = ""

    for ann in f_content:
        img_data = ann.split(' ')
        if img_data == ['']:
            break
        try:
            file_name, width, height, xmin, ymin, xmax, ymax, label = img_data
        except:
            file_name, width, height, xmin, ymin, xmax, ymax, label1, label2 = img_data
            label = label1 + "_" + label2
        if not current == file_name:
            img = {'height': float(width), 'width': float(height), 'object': [], 'filename': file_name}
            current = file_name
            all_img.append(img)

        img['object'].append({'xmin': float(xmin), 'ymin': float(ymin),
                          'name': label, 'xmax': float(xmax),
                          'ymax': float(ymax)})

    return all_img


def draw_boxes(img, bboxes, classes, scores):
    if len(bboxes) == 0:
        return img

    height, width, _ = img.shape
    image = Image.fromarray(img)
    font = ImageFont.truetype(
        font='../new_YOLO/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.4).astype('int32'))

    thickness = (image.size[0] + image.size[1]) // 300
    draw = ImageDraw.Draw(image)

    for box, category, score in zip(bboxes, classes, scores):
        y1, x1, y2, x2 = [int(i) for i in box]
        p1 = (x1, y1)
        p2 = (x2, y2)

        label = '{} {:.1f}%   '.format(category.title(), score * 100)
        label_size = draw.textsize(label)
        text_origin = np.array([p1[0], p1[1] - label_size[1]])

        color = np.array([0, 255, 0])
        for i in range(thickness):
            draw.rectangle(
                [p1[0] + i, p1[1] + i, p2[0] - i, p2[1] - i],
                outline=tuple(color))

        draw.rectangle(
            [tuple(text_origin),
             tuple(text_origin + label_size)],
            fill=tuple(color))

        draw.text(
            tuple(text_origin),
            label, fill=(0, 0, 0),
            font=font)

    del draw
    return np.array(image)


def multi_aug_img(train_instance, s):
    path = train_instance['filename']
    all_obj = copy.deepcopy(train_instance['object'][:])
    img = cv2.imread(img_dir + path)
    h, w, c = img.shape

    # scale the image
    scale = np.random.uniform() / 10. + 1.
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    # translate the image
    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    img = img[offy: (offy + h), offx: (offx + w)]

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5: img = cv2.flip(img, 1)

    # re-color
    #t = [np.random.uniform()]
    #t += [np.random.uniform()]
    #t += [np.random.uniform()]
    #t = np.array(t)

    #img = img * (1 + t)
    img = img / 255.

    # resize the image to standard size
    img = cv2.resize(img, (MULTI_SCALE_INPUT[s], MULTI_SCALE_INPUT[s]))
    img = img[:, :, ::-1]

    # fix object's position and size
    for obj in all_obj:
        for attr in ['xmin', 'xmax']:
            obj[attr] = int(obj[attr] * scale - offx)
            obj[attr] = int(obj[attr] * float(MULTI_SCALE_INPUT[s]) / w)
            obj[attr] = max(min(obj[attr], MULTI_SCALE_INPUT[s]), 0)

        for attr in ['ymin', 'ymax']:
            obj[attr] = int(obj[attr] * scale - offy)
            obj[attr] = int(obj[attr] * float(MULTI_SCALE_INPUT[s]) / h)
            obj[attr] = max(min(obj[attr], MULTI_SCALE_INPUT[s]), 0)

        if flip > 0.5:
            xmin = obj['xmin']
            obj['xmin'] = MULTI_SCALE_INPUT[s] - obj['xmax']
            obj['xmax'] = MULTI_SCALE_INPUT[s] - xmin

    return img, all_obj


def multi_data_gen(all_img, batch_size):
    num_img = len(all_img)
    shuffled_indices = np.random.permutation(np.arange(num_img))
    l_bound = 0
    r_bound = batch_size if batch_size < num_img else num_img

    while True:
            s = random.randint(0,9)
            if l_bound == r_bound:
                l_bound = 0
                r_bound = batch_size if batch_size < num_img else num_img
                shuffled_indices = np.random.permutation(np.arange(num_img))

            batch_size2 = r_bound - l_bound
            currt_inst = 0
            x_batch = np.zeros((batch_size2, MULTI_SCALE_INPUT[s], MULTI_SCALE_INPUT[s], 3))
            y_batch = np.zeros((batch_size2, MULTI_SCALE_OUTPUT[s], MULTI_SCALE_OUTPUT[s], BOX, 5 + CLASS))

            for index in shuffled_indices[l_bound:r_bound]:
                train_instance = all_img[index]
                img, all_obj = multi_aug_img(train_instance, s)

                # construct output from object's position and size
                for obj in all_obj:
                    center_x = .5 * (obj['xmin'] + obj['xmax'])  # xmin, xmax
                    center_x = center_x / (float(MULTI_SCALE_INPUT[s]) / MULTI_SCALE_OUTPUT[s])
                    center_y = .5 * (obj['ymin'] + obj['ymax'])  # ymin, ymax
                    center_y = center_y / (float(MULTI_SCALE_INPUT[s]) / MULTI_SCALE_OUTPUT[s])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < MULTI_SCALE_OUTPUT[s] and grid_y < MULTI_SCALE_OUTPUT[s]:
                        obj_indx = LABELS.index(obj['name'])
                        box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]

                        y_batch[currt_inst, grid_y, grid_x, :, 0:4] = BOX * [box]
                        y_batch[currt_inst, grid_y, grid_x, :, 4] = BOX * [1.]
                        y_batch[currt_inst, grid_y, grid_x, :, 5:] = BOX * [[0.] * CLASS]
                        y_batch[currt_inst, grid_y, grid_x, :, 5 + obj_indx] = 1.0

                # concatenate batch input from the image
                x_batch[currt_inst] = img
                currt_inst += 1

                del img, all_obj
            yield x_batch, y_batch

            l_bound = r_bound
            r_bound = r_bound + batch_size
            if r_bound > num_img: r_bound = num_img


def compute_ap(recall, precision):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_overlap(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua
