"""
Main file to train Yolov3 on costum dataset
"""
from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_utils import *

def YoloV3(size, channels, anchors, masks, classes=80, training=False):
	x = inputs = Input([size, size, channels])

	x_36, x_61, x = Darknet(name='yolo_darknet')(x)

	x = YoloConv(512, name='yolo_conv_0')(x)
	output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

	x = YoloConv(256, name='yolo_conv_1')((x, x_61))
	output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

	x = YoloConv(128, name='yolo_conv_2')((x, x_36))
	output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

	if training:
		return Model(inputs, (output_0, output_1, output_2), name='yolov3')

	boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
					name='yolo_boxes_0')(output_0)
	boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
					name='yolo_boxes_1')(output_1)
	boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
					name='yolo_boxes_2')(output_2)

	outputs = Lambda(lambda x: nonMaximumSuppression(x, anchors, masks, classes),
					name='nonMaximumSuppression')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

	return Model(inputs, outputs, name='yolov3')

if __name__ == '__main__':

	yolo_iou_threshold   = 0.6           # iou threshold
	yolo_score_threshold = 0.6           # score threshold

	weightsyolov3 = 'yolov3.weights'     # path to weights file
	weights= 'checkpoints/yolov3.tf'     # path to checkpoints file
	size = 416                           # resize images to\
	checkpoints = 'checkpoints/yolov3.tf'
	num_classes = 80                     # number of classes in the model

	yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
							(59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
	yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

	yolo = YoloV3(size=None, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=1, training=True)
	# print(yolo.summary())

	## TRAINING
	# set up
	optimizer = tf.keras.optimizers.Adam(lr=0.001)
	loss = [YoloLoss(yolo_anchors[mask], classes=1) for mask in yolo_anchor_masks]
	yolo.compile(optimizer=optimizer, loss=loss)
	