"""
Expolore the feature of TFRdataset 
"""
import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
import time
from yolov3_utils import *

def main_analize(dataset, color_dict, num):
    """
    Return the radius and the count by label of given dataset
    """
    real_radius = []
    count_label_list = []
    for ii, (img, label, width, height) in enumerate(dataset.take(num)):
        # take images and label
        img_real, label_real, width, height = img.numpy()[0], label.numpy()[0], width.numpy(), height.numpy()

        # Real images
        boxes, lab = label_real[:, 0:4], label_real[:,-1]
        wh = np.flip(img_real.shape[0:2])
        
        real_nums = (len(boxes))
        radius_i = []
        count_label = {0.:0, 1.:0, 2.:0, 3.:0, 4.:0, 5.:0, 6.:0}
        for i in range(real_nums):
            x1y1 = tuple((np.array(boxes[i][0:2])*wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4])*wh).astype(np.int32))
            count_label[lab[i]] += 1
            radius_1, radius_2 = np.abs(x1y1[0]-x2y2[0]), np.abs(x1y1[1]-x2y2[1])
            radius_mean, radius_error = (radius_1 + radius_2)/2.0, np.abs((radius_1 - radius_2)/2.0)
            radius_i.append(radius_mean)
            # print(radius_mean, radius_error)
            img_real = cv2.rectangle(img_real, x1y1, x2y2, color_dict[lab[i]], 20)
        real_radius.append(radius_i)
        count_label_list.append(count_label)

        # plt.figure()
        # plt.imshow(img_real)
        # plt.axis('off')
        # plt.show()

    return real_radius, count_label_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset explorer')
    args = parser.parse_args()

    label_dict = {'Empty_GUV':0, 'Fused_GUV':1, 'Blurry_GUV':2, 'Full_GUV':3, 
                  'Faint_GUV':4, 'Deformed_GUV':5, 'Edge_GUV':6}

    color_dict = {0.:(220,20,60),   # crimsom (red)
                  1.:(0,0,139),     # darkblue
                  2.:(0,100,0),     # darkgreen 
                  3.:(0,191,255),   # deepskyblue
                  4.:(255,215,0),   # gold
                  5.:(255,105,180), # hotpink
                  6.:(255,69,0)}    # orange

    split_dict = {}
    for i in ['train', 'validation', 'test']:    
        test_dataset = tf.data.TFRecordDataset(f'TFRecords/{i}_label.tfrecord')
        test_dataset = test_dataset.map(parse_tfrecords_test)
        test_dataset = test_dataset.batch(1)

        num_image = len(os.listdir('Annotations/train/images'))
        print(num_image)
        radius, count_label_list = main_analize(test_dataset, color_dict, num=num_image)
        
        tot_count_dict = {0.:0, 1.:0, 2.:0, 3.:0, 4.:0, 5.:0, 6.:0}

        for count_dict in count_label_list:
            for key in color_dict.keys():
                tot_count_dict[key] += count_dict[key]
        split_dict[i] = tot_count_dict

    print(split_dict)

    hist_dict = {}
    for key in split_dict.keys():
        values = list(split_dict[key].values())
        values_percentage = np.array(values)/(np.array(values).sum())*100
        hist_dict[key] = values_percentage

    x_ticks = np.array(range(len(tot_count_dict)))
    plt.figure(num='Labels distribuction', tight_layout=True)
    plt.bar(x_ticks - 0.3, hist_dict['train'], 0.3, tick_label=list(label_dict.keys()), label='train')
    plt.bar(x_ticks, hist_dict['validation'], 0.3, tick_label=list(label_dict.keys()), label='validation')
    plt.bar(x_ticks + 0.3, hist_dict['test'], 0.3, tick_label=list(label_dict.keys()), label='test')
    plt.xticks(x_ticks, list(label_dict.keys()))
    plt.legend()

    plt.show()