#!/usr/bin/env python3
''' imageGrabber.py

    This is a ros node for grabbing an image stream from
    a drone and publishing the image frames to a ros topic

    Author: Arden Knoll
'''
import rospy
#import socket
#import codecs
import cv2
import numpy as np
#import h264decoder
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class infoPub():

    def __init__(self):
        rospy.init_node('image_stream', anonymous=False)   # Can have only one leader node
        self.img_sub = rospy.Subscriber('/image_stream/sharp_image', Image, self.republish_info)
        self.info_pub = rospy.Publisher('/image_stream/sharp_camera_info', CameraInfo,  queue_size=10)
        self.br = CvBridge()
        self.cam_info = CameraInfo()

        self.cam_info.height = 576
        self.cam_info.width = 720
        self.cam_info.distortion_model = "plumb_bob"
        self.cam_info.D = [0.8337068337092105, -0.7296451172680306, 0.015187197230019187, 0.08793175446175946, 0.0]
        self.cam_info.K = [1989.1574391172464, 0.0, 554.6150636910545, 0.0, 2068.7938072844036, 280.42584157479126, 0.0, 0.0, 1.0]
        self.cam_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.cam_info.P = [2012.7718505859375, 0.0, 562.0574733415124, 0.0, 0.0, 2135.0751953125, 281.9590617435897, 0.0, 0.0, 0.0, 1.0, 0.0]


    def republish_info(self, img_msg):
        self.cam_info.header.stamp = img_msg.header.stamp
        self.info_pub.publish(self.cam_info)

if __name__ == '__main__':
    stream = infoPub()
    rospy.spin()
