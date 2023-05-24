"""
Main file to train Yolov3 on costum dataset
"""
import argparse

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
	yolo_iou_threshold   = 0.6           # iou threshold
	yolo_score_threshold = 0.6           # score threshold

	weightsyolov3 = 'Pretreined_yolo/checkpoints/yolov3.tf'  # path to weights file
	weights= 'checkpoints/yolov3.tf'                         # path to checkpoints file                          
	checkpoints = 'checkpoints/yolov3.tf'

	BUFFER_SIZE = args.buffer_size
	BATCH = args.batch
	SIZE = args.size                    
	
	yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
							(59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
	yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
	num_classes = args.num_classes      # number of classes in the model

	## Load train - validation dataset
	train_dataset = tf.data.TFRecordDataset('TFRecords/train.tfrecord')
	train_dataset = train_dataset.map(parse_tfrecords)
	# for img, label in train_dataset.take(1):
	# 	img, label = img.numpy()/255, label.numpy()
	# 	plt.figure(figsize=(12, 12))
	# 	# for i in range(4):
	# 	# 	img_d, label_d = data_augumentation(img, label)
	# 	# 	plt.subplot(2, 2, i + 1)
	# 	# 	plt.imshow(img_d)
	# 	# 	plt.axis('off')
	# 	plt.figure()
	# 	plt.imshow(img)
	# 	plt.axis('off')
	# plt.show()
	train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE)
	train_dataset = train_dataset.batch(BATCH)
	train_dataset = train_dataset.map(lambda x, y: (
		transform_images(x, SIZE),
		transform_targets(y, yolo_anchors, yolo_anchor_masks, SIZE)))
	train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	
	val_dataset = tf.data.TFRecordDataset('TFRecords/validation.tfrecord')
	val_dataset = val_dataset.map(parse_tfrecords)
	val_dataset = val_dataset.batch(BATCH)
	val_dataset = val_dataset.map(lambda x, y: (
		transform_images(x, SIZE),
		transform_targets(y, yolo_anchors, yolo_anchor_masks, SIZE)))

	
	## YOLO MODEL
	yolo = YoloV3(size=SIZE, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=num_classes, training=True)
	## pretrained
	model_pretrained = YoloV3(size=SIZE, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=80, training=True)
	model_pretrained.load_weights(weightsyolov3)
	yolo.get_layer('yolo_darknet').set_weights(model_pretrained.get_layer('yolo_darknet').get_weights())
	freeze_all(yolo.get_layer('yolo_darknet'))
	print(yolo.summary())

	## TRAINING
	# set up
	callbacks = [
		# ReduceLROnPlateau(verbose=1),
		# EarlyStopping(patience=3, verbose=1),
		ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf', verbose=1, save_weights_only=True, save_freq=5*BATCH*8),
		TensorBoard(log_dir='logs')
	]
	learning_rate = args.learning_rate

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	loss = [YoloLoss(yolo_anchors[mask], classes=num_classes) for mask in yolo_anchor_masks]
	yolo.compile(optimizer=optimizer, loss=loss)
	
	# fitting
	start_time = time.time()
	history = yolo.fit(train_dataset, 
					callbacks=callbacks,
					epochs=args.epochs, 
					validation_data=val_dataset,
					verbose=1)
	end_time = time.time() - start_time
	print(f'Total Training Time: {end_time}')



