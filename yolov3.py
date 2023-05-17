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
import matplotlib.pyplot as plt

# def YoloV3(size, channels, anchors, masks, classes=80, training=False):
# 	x = inputs = Input([size, size, channels])

# 	x_36, x_61, x = Darknet(name='yolo_darknet')(x)

# 	x = YoloConv(512, name='yolo_conv_0')(x)
# 	output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

# 	x = YoloConv(256, name='yolo_conv_1')((x, x_61))
# 	output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

# 	x = YoloConv(128, name='yolo_conv_2')((x, x_36))
# 	output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

# 	if training:
# 		return Model(inputs, (output_0, output_1, output_2), name='yolov3')

# 	boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
# 					name='yolo_boxes_0')(output_0)
# 	boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
# 					name='yolo_boxes_1')(output_1)
# 	boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
# 					name='yolo_boxes_2')(output_2)

# 	outputs = Lambda(lambda x: nonMaximumSuppression(x, anchors, masks, classes),
# 					name='nonMaximumSuppression')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

# 	return Model(inputs, outputs, name='yolov3')



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

	## YOLO MODEL
	yolo = YoloV3(size=size, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=1, training=True)

	## pretrained
	model_pretrained = YoloV3(size=size, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=80, training=True)
	model_pretrained.load_weights('Pretreined_yolo/checkpoints/yolov3.tf')
	yolo.get_layer('yolo_darknet').set_weights(model_pretrained.get_layer('yolo_darknet').get_weights())
	freeze_all(yolo.get_layer('yolo_darknet'))
	print(yolo.summary())

	## TRAINING
	# set up
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
	loss = [YoloLoss(yolo_anchors[mask], classes=1) for mask in yolo_anchor_masks]
	yolo.compile(optimizer=optimizer, loss=loss)
	
	# load dataset
	train_dataset = tf.data.TFRecordDataset('dataset.tfrecord')
	train_dataset = train_dataset.map(parse_tfrecords)
	train_dataset = train_dataset.shuffle(buffer_size=4)
	train_dataset = train_dataset.batch(1)
	train_dataset = train_dataset.map(lambda x, y: (
		transform_images(x, 416),
		transform_targets(y, yolo_anchors, yolo_anchor_masks, 416)))
	train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	# fitting
	start_time = time.time()
	history = yolo.fit(train_dataset, epochs=20, verbose=1)
	end_time = time.time() - start_time
	print(f'Total Training Time: {end_time}')

	## prediction
	name = 'Test/IMG01493.JPG'
	img = tf.image.decode_image(open(name, 'rb').read(), channels=3)
	img = tf.expand_dims(img, 0)
	img = preprocess_image(img, size)

	print(len(yolo(img)))
	print(yolo(img)[0].shape)
	print(yolo(img)[1].shape)
	print(yolo(img)[2].shape)


	## postprocess yolo output
	boxes_0 = yolo_boxes(yolo(img)[0], yolo_anchors[yolo_anchor_masks[0]], classes=1)
	boxes_1 = yolo_boxes(yolo(img)[1], yolo_anchors[yolo_anchor_masks[1]], classes=1)
	boxes_2 = yolo_boxes(yolo(img)[2], yolo_anchors[yolo_anchor_masks[2]], classes=1)

	boxes, scores, classes, nums = nonMaximumSuppression((boxes_0[:3], boxes_1[:3], boxes_2[:3]), yolo_anchors, yolo_anchor_masks, classes=1,
														 yolo_iou_threshold=0.1, yolo_score_threshold=0.1)
	print(boxes.shape)
	print(scores.shape)
	print(classes.shape)
	print(nums.shape)

	for i,j in zip(boxes[0], scores[0]):
		print(i,j)

	img = cv2.imread(name)
	img = draw_outputs(img, (boxes, scores, classes, nums))
	cv2.imwrite(f'{name}_out.jpg', img)

	
	

