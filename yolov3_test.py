"""
Main file to test the yolo output
"""
import argparse

import tensorflow as tf
import numpy as np
import cv2
import time
from yolov3_utils import *


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main file to train yolov3')
	parser.add_argument("-buffer_size", default=16, type=int, help="shufle buffer size")
	parser.add_argument("-batch", default=8, type=int, help="batch")
	parser.add_argument("-size", default=416, type=float, help="square img size")
	parser.add_argument("-num_classes", default=1, type=int, help="number of classes")
	parser.add_argument("-epochs", default=10, type=int, help="epochs")
	parser.add_argument("-learning_rate", default=0.001, type=float, help="learning_rate")
	args = parser.parse_args()

	##Paramenters
	yolo_iou_threshold   = 0.05           # iou threshold
	yolo_score_threshold = 0.05           # score threshold

	checkpoint = 'checkpoints/yolov3_train_10.tf'

	BUFFER_SIZE = args.buffer_size
	BATCH = args.batch
	SIZE = args.size                    
	
	yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
							(59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
	yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
	num_classes = args.num_classes      # number of classes in the model

	
	## Load trained model
	yolo = YoloV3(size=SIZE, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=num_classes, training=True)
	yolo.load_weights(checkpoint).expect_partial()
	print(yolo.summary())

	## prediction
	test_dataset = tf.data.TFRecordDataset('TFRecords/test.tfrecord')
	test_dataset = test_dataset.map(parse_tfrecords)
	test_dataset = test_dataset.batch(BATCH)

	plot_dataset(test_dataset)
	plt.show()

	# prediction = yolo.predict(test_dataset, verbose=1)
	

	# name = 'Test/IMG01493.JPG'
	# img = tf.image.decode_image(open(name, 'rb').read(), channels=3)
	# img = tf.expand_dims(img, 0)
	# img = preprocess_image(img, SIZE)

	# pred = yolo(img)
	# print(len(pred))

	# ## postprocess yolo output
	# boxes_0 = yolo_boxes(pred[0], yolo_anchors[yolo_anchor_masks[0]], classes=1)
	# boxes_1 = yolo_boxes(pred[1], yolo_anchors[yolo_anchor_masks[1]], classes=1)
	# boxes_2 = yolo_boxes(pred[2], yolo_anchors[yolo_anchor_masks[2]], classes=1)

	# boxes, scores, classes, nums = nonMaximumSuppression((boxes_0[:3], boxes_1[:3], boxes_2[:3]), yolo_anchors, yolo_anchor_masks, classes=1,
	# 													yolo_iou_threshold=0.1, yolo_score_threshold=0.1)
	# print(boxes.shape)
	# print(scores.shape)
	# print(classes.shape)
	# print(nums.shape)

	# for i,j in zip(boxes[0], scores[0]):
	# 	print(i.numpy(),j.numpy())

	# img = cv2.imread(name)
	# img = draw_outputs(img, (boxes, scores, classes, nums))
	# cv2.imwrite('out.jpg', img)