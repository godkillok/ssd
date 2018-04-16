
# coding: utf-8

# In[1]:


import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim


# In[2]:



import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[3]:


import sys
sys.path.append('../')


# In[4]:


from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization


# In[5]:


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# ## SSD 300 Model
# 
# The SSD 300 network takes 300x300 image inputs. In order to feed any image, the latter is resize to this input shape (i.e.`Resize.WARP_RESIZE`). Note that even though it may change the ratio width / height, the SSD model performs well on resized images (and it is the default behaviour in the original Caffe implementation).
# 
# SSD anchors correspond to the default bounding boxes encoded in the network. The SSD net output provides offset on the coordinates and dimensions of these anchors.

# In[6]:


# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# ## Post-processing pipeline
# 
# The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
# 
# * Select boxes above a classification threshold;
# * Clip boxes to the image shape;
# * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# * If necessary, resize bounding boxes to original image shape.

# In[7]:


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.

    #------将一张图片输入ssd网络后，rpredictions,预测的21个类别，会输出6层预测，（第一层预测为1（batch size）*38(宽的网格数)*
    # 38（长的网格数）*4（每个网格4个anchor）*21(21个种类)；第2为(1, 19, 19, 6, 21)....第6层(1, 1, 1, 4, 21)）

    #-----rlocalisations 预测的ancher的 cx cy w h的值，所以也是6层，第一层:
    # (1, 38, 38, 4, 4)，注意这个其实不是ancher的大小真正的值，这个只是相对
    # 初始ancher的变形的一些参数，所以要得到acher的大小，需要转换具体在np_methods的
    # ssd_bboxes_decode方法中计算，在代码这个过程成为decode
    #至于为啥ssd出来的location 不是中心点的x,y和w,h,而是他们的变形参数。
    # 因为需要考虑一个因素就是不会去预测正确的ancher与初始ancher的绝对值差别多少，
    # 而是预测一定的变形参数，主要是考虑到泛化性，具体可以百度 bounding boxing regression。
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    #
    # Get classes and bboxes from the net outputs.
    #将21类预测score大于阈值，将符合的ancher 选出
    #rclasses=(1,符合条件的ancher数), rscores=(1,符合条件的ancher数), rbboxes=(符合条件的ancher数,4)
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    # 主要剪掉一些超出图片最大边框的ancher
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    # 排序选出top_k个预测score最大的ancher box
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    # 根据ancher的non max suppression 保留重叠面积小，或者是不同种类的acher.(主要是去除那些预测种类一样，又重叠面积大的ancher)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# In[21]:


# Test on some demo image and visualize output.
path = '../demo/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[-5])
rclasses, rscores, rbboxes =  process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

