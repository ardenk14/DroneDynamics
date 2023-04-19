#!/usr/bin/env python
''' imageGrabber.py

    This is a ros node for grabbing an image stream from
    a drone and publishing the image frames to a ros topic

    Author: Arden Knoll
'''
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int8MultiArray
import sys
sys.path.append('/home/ardenk14/GhostDeblurGAN')
from predict import Predictor

class image_test():

    def __init__(self):
        rospy.init_node('image_display', anonymous=False)
        self.br = CvBridge()
        self.cam_info = CameraInfo()
        self.frame = None
        self.cmd = Int8MultiArray() 
        self.cmd.data = [127, 0, 0, 0]
        self.predictor = Predictor(weights_path='/home/ardenk14/GhostDeblurGAN/trained_weights/fpn_ghostnet_gm_hin.h5', cuda= True)
        self.pub = rospy.Publisher('/image_stream/sharp_image', Image,  queue_size=10)
        self.info_pub = rospy.Publisher('/image_stream/camera_info', CameraInfo,  queue_size=10)

        self.cam_info.height = 576
        self.cam_info.width = 720
        self.cam_info.distortion_model = "plumb_bob"
        self.cam_info.D = [0.8337068337092105, -0.7296451172680306, 0.015187197230019187, 0.08793175446175946, 0.0]
        self.cam_info.K = [1989.1574391172464, 0.0, 554.6150636910545, 0.0, 2068.7938072844036, 280.42584157479126, 0.0, 0.0, 1.0]
        self.cam_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.cam_info.P = [2012.7718505859375, 0.0, 562.0574733415124, 0.0, 0.0, 2135.0751953125, 281.9590617435897, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.D = np.array([0.8337068337092105, -0.7296451172680306, 0.015187197230019187, 0.08793175446175946, 0.0])
        self.K = np.array([[1989.1574391172464, 0.0, 554.6150636910545], [0.0, 2068.7938072844036, 280.42584157479126], [0.0, 0.0, 1.0]])

        self.sub = rospy.Subscriber('/image_stream/image', Image, self.show_img) 
        print("INITIALIZED")  

    def show_img(self, img_msg):
        #print("RECEIVED")
        try:
            self.frame = self.br.imgmsg_to_cv2(img_msg)
        except CvBridgeError as e:
            print(e)
            return
        #print("CONVERTED")
        #print("FRAME: ", self.frame.shape)
        if self.frame is not None:
            #print("INSIDE")
            h,  w = self.frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w,h), 1, (w,h))
            #print("NEW CAM: ", newcameramtx)
            self.frame = cv2.undistort(self.frame, self.K, self.D, None, newcameramtx)


            #pred = self.frame
            img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            pred = self.predictor(img, None)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)


            """hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
      
            # Threshold of blue in HSV space
            lower_blue = np.array([0, 35, 0])
            upper_blue = np.array([180, 255, 100])
        
            # preparing the mask to overlay
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # The black region in the mask has the value of 0,
            # so when multiplied with original image removes all non-blue regions
            result = cv2.bitwise_and(self.frame, self.frame, mask = mask)"""
            cv2.imshow('frame', pred)
            new_img_msg = self.br.cv2_to_imgmsg(pred)
            new_img_msg.encoding = 'bgr8'
            new_img_msg.header.stamp = img_msg.header.stamp
            self.cam_info.header.stamp = img_msg.header.stamp
            self.pub.publish(new_img_msg)
            self.info_pub.publish(self.cam_info)
        #print("DISPLAYED")

        #cmd = Int8MultiArray()

        k = cv2.waitKey(2) & 0xFF
        if k == ord('q'):
            rospy.signal_shutdown('Quitting')
        """elif k == ord('w'):
            self.cmd.data[0] = 0xFF
        elif k == ord('s'):
            self.cmd.data[0] = 0xA0
        elif k == ord('x'):
            self.cmd.data[0] = 0x00"""

        #self.pub.publish(self.cmd)



if __name__ == '__main__':

    image_test()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()