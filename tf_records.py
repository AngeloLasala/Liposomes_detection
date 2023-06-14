"""
Convert VOC dataset to tensorflow TF Records to object detections
"""

import time
import os
import hashlib
import argparse

import tensorflow as tf
import lxml.etree
import tqdm
from PIL import Image
import matplotlib.pyplot as plt


def build_example(image_path, xml_path, label_dict=None):
	## get the filename
	filename = os.path.basename(image_path).encode('utf8')

	## get image path and open it
	img_path = os.path.join(image_path)
	img_raw = open(img_path, 'rb').read()
	key = hashlib.sha256(img_raw).hexdigest()
	img = Image.open(img_path)
	width, height = img.size  # img width and height

	## get xml infprmatio
	xml_path = os.path.join(xml_path)
	with tf.io.gfile.GFile(xml_path, 'rb') as fid:
		xml_str = fid.read()
	xml = lxml.etree.fromstring(xml_str)

	xmin = []
	ymin = []
	xmax = []
	ymax = []
	classes = []
	classes_text = []
	truncated = []
	views = []
	difficult_obj = []

	
	for obj in xml.xpath('//object'): #loop over element in the images, aka bounding box
		# print(obj)
		xmin.append(float(obj.xpath('bndbox/xmin')[0].text) / width)
		ymin.append(float(obj.xpath('bndbox/ymin')[0].text) / height)
		xmax.append(float(obj.xpath('bndbox/xmax')[0].text) / width)
		ymax.append(float(obj.xpath('bndbox/ymax')[0].text) / height)
		classes_text.append(obj.xpath('name')[0].text.encode('utf8'))
		if label_dict==None:
			classes.append(1)
		else:
			classes.append(label_dict[obj.xpath('name')[0].text.encode('utf8')]) 
		truncated.append(int(obj.xpath('truncated')[0].text))
		views.append(obj.xpath('pose')[0].text.encode('utf8'))

	print(f' xmin: {xmin}')
	print(f' xmax: {xmax}')
	print(f' ymin: {ymin}')
	print(f' ymax: {ymax}')
	print(f' classes_text: {classes_text}')
	print(f' classes: {classes}')
	print(f' truncated: {truncated}')
	print(f' views: {views}')
	print('---------------------------------------------------------------------')

	example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
		'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
		'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
			filename])),
		'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
			filename])),
		'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
		'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
		'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['png'.encode('utf8')])),
		'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
		'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
		'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
		'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
		'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
		'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
		'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
		'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
		'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
	}))
	return example

def main(data_dir, output, label_dict):
	"""
	Main function to covert VOC dataset into TFRecords.
	"""
	writer = tf.io.TFRecordWriter('TFRecords' + '/' + output)
	image_list = [data_dir + '/' + 'images' + '/' + i for i in os.listdir(data_dir + '/' + 'images')]
	xml_path = [data_dir + '/' + 'Annotations' + '/' + i for i in os.listdir(data_dir + '/' + 'Annotations')]
	for img in tqdm.tqdm(image_list):
		xml = 'Annotations/Annotations/' + img.split('/')[-1].split('.')[0] + '.xml'
		print(img)
		tf_example = build_example(img, xml, label_dict)
		writer.write(tf_example.SerializeToString())
	writer.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Converter from VOC dataset in TFRecords')
	parser.add_argument("data_directory", type=str, help="VOC dataset directory")
	args = parser.parse_args()
	
	label_dict = {b'Empty_GUV':0, b'Fused_GUV':1, b'Blurry_GUV':2, b'Full_GUV':3, 
                  b'Faint_GUV':4, b'Deformed_GUV':5, b'Edge_GUV':6}
	
	for k in ['train', 'test', 'validation']:
		main(args.data_directory + '/' + k, k + '_label.tfrecord', label_dict)
