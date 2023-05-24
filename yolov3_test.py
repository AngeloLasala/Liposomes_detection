"""
Main file to test the yolo output
"""
import argparse

import tensorflow as tf
import numpy as np
import cv2
import time
from yolov3_utils import *

def analysis_test(test_dataset):
	"""
	Analyse and compare the results between real bbox and predicted ones.
	- Number of detected liposomes
	- dimension of detected liposomes
	- plot images

	Parameters
	----------
	test_dataset : tensorflow dataset
		test dataset of images and labels

	Returns
	-------
	"""
	for img, label in test_dataset.take(1):
		# take images and label
		img_real, label_real = img.numpy()[0]/255, label.numpy()[0]
		
		# Real images
		boxes, lab = label_real[:, 0:4], label_real[:,-1]
		wh = np.flip(img_real.shape[0:2])
		print(img.shape)
		
		real_nums = (len(boxes))
		for i in range(real_nums):
			x1y1 = tuple((np.array(boxes[i][0:2])*wh).astype(np.int32))
			x2y2 = tuple((np.array(boxes[i][2:4])*wh).astype(np.int32))
			img_real = cv2.rectangle(img_real, x1y1, x2y2, (255, 0, 0), 2)
			# img = cv2.putText(img, ' {:.4f}'.format(scores[i]),
			# x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

		# prediction
		pred = yolo(img.numpy()/255)
		boxes_0 = yolo_boxes(pred[0], yolo_anchors[yolo_anchor_masks[0]], classes=1)
		boxes_1 = yolo_boxes(pred[1], yolo_anchors[yolo_anchor_masks[1]], classes=1)
		boxes_2 = yolo_boxes(pred[2], yolo_anchors[yolo_anchor_masks[2]], classes=1)

		boxes, scores, classes, nums = nonMaximumSuppression((boxes_0[:3], boxes_1[:3], boxes_2[:3]), yolo_anchors, yolo_anchor_masks, classes=1,
															yolo_iou_threshold=0.1, yolo_score_threshold=0.1)

		# print(nums.numpy())
		# for i,j in zip(boxes[0], scores[0]):
		# 	print(i.numpy(), j.numpy())

		img_pred = draw_outputs(img.numpy()[0]/255, (boxes, scores, classes, nums))

		## Plot and print the results
		print('NUMBER OF LIPOSOMES')
		print(f'real number: {real_nums} - predicted number {nums[0]}')
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))

		ax[0].set_title('Real Boundig Box')
		ax[0].imshow(img_real)
		ax[0].axis('off')

		ax[1].set_title('Predicted Boundig Box')
		ax[1].imshow(img_pred)
		ax[1].axis('off')

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

	checkpoint = 'checkpoints/yolov3_train_20.tf'

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

	analysis_test(test_dataset)
		
	plt.show()

	# print(scores.shape)
	# print(classes.shape)
	# print(nums.shape)

	# cv2.imwrite('out.jpg', img)
	