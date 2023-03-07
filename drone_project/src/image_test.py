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

class image_test():

    def __init__(self):
        rospy.init_node('image_display', anonymous=False)
        self.br = CvBridge()
        self.frame = None
        self.cmd = Int8MultiArray() 
        self.cmd.data = [127, 0, 0, 0]
        self.pub = rospy.Publisher('/drone_commands', Int8MultiArray, queue_size=10)
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
            cv2.imshow('frame', self.frame)
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

        self.pub.publish(self.cmd)



if __name__ == '__main__':

    image_test()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()