#!/usr/bin/env python3
''' controller.py
    
    Author: Arden Knoll
'''
import rospy
#import socket
#import codecs
#import cv2
import numpy as np
#import h264decoder
#from sensor_msgs.msg import Image, CameraInfo
#from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import UInt8MultiArray
import pygame
MAX_RANGE=255
MIN_RANGE=0

class ps4_controller():

    def __init__(self):
        rospy.init_node('controller', anonymous=False)   # Can have only one leader node
        self.img_pub = rospy.Publisher('/drone_commands', UInt8MultiArray,  queue_size=10)

        # Set up the drawing window
        
        # Run until the user asks to quit
        running = True
        #num_devices = tch.get_num_devices()
        #print(num_devices)
        #device = tch.get_device(0)
        #print(device)
        #screen = pygame.display.set_mode([500, 500])

        pygame.joystick.init()
        joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        print("JOYSTICKS: ", joysticks)
        while not rospy.is_shutdown():
            #m.header.stamp = rospy.Time.now()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("QUIT")
                    #done=True
                if event.type == pygame.JOYBUTTONDOWN:
                    print("Button DOWN")
                    #done=False
                if event.type == pygame.JOYBUTTONUP:
                    print("BUTTON UP")
                    #done=False
            #axis = [5,3,4,0]
            axis = [5,0,1,3]
            control_val=[]
            for i in range(len(axis)):
                axis_crt = joysticks[0].get_axis(axis[i])
                int_size = (MAX_RANGE-MIN_RANGE)/2 - 1
                
                if axis_crt==5:
                    pass
                else:
                    axis_crt = axis_crt**5
                
                if axis[i]==1:
                    cal_control_val = -(int)((axis_crt-1)*int_size+MIN_RANGE)
                else:
                    cal_control_val = (int)((axis_crt+1)*int_size+MIN_RANGE)
                control_val.append(cal_control_val)
                
                #m.axes[i] = axis

            c = UInt8MultiArray()
            c.data = control_val
            self.img_pub.publish(c)

            rospy.sleep(0.002)

if __name__ == '__main__':
    pygame.init()
    dr = ps4_controller()
    #try:
    #    rospy.spin()
    #except KeyboardInterrupt:
    #    pass
    # Done! Time to quit.
    pygame.quit()