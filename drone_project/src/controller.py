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

MAX_RANGE_THROTTLE = 255
MIN_RANGE_THROTTLE = 0

MAX_RANGE_PITCH = 220
MIN_RANGE_PITCH = 110

MAX_RANGE_RY = 158
MIN_RANGE_RY = 100
MAPPING_POWER = 5


class ps4_controller():

    def __init__(self):
        rospy.init_node('controller', anonymous=False)   # Can have only one leader node
        self.img_pub = rospy.Publisher('/drone_commands', UInt8MultiArray,  queue_size=10)
        self.setThrottle = False
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
                    #print("QUIT")
                    pass#done=True
                if event.type == pygame.JOYBUTTONDOWN:
                    #print("Button DOWN")
                    #done=False
                    if joysticks[0].get_button(1):
                        if self.setThrottle:
                            self.setThrottle = False
                        else:
                            self.setThrottle=True
                    
                if event.type == pygame.JOYBUTTONUP:
                    #print("BUTTON UP")
                    #done=False
                    pass
            #axis = [5,3,4,0]
            
            axis = [5,0,1,3]
            control_val=[]
            
            # Setting throttle once reached minimum height
            #throttle_setter = joysticks[0].get_axis(2)
            #if throttle_setter>=0:
                
                
            for i in range(len(axis)):
                axis_crt = joysticks[0].get_axis(axis[i])
                
                if axis[i]==2 and axis_crt >= 0.5:
                    self.setThrottle = True
                
                elif axis[i]==5:
                    MAX_RANGE = MAX_RANGE_THROTTLE
                    if self.setThrottle:
                        MIN_RANGE = 150
                    else:
                        MIN_RANGE = MIN_RANGE_THROTTLE
                elif axis[i]==1:
                    axis_crt = axis_crt**MAPPING_POWER
                    MAX_RANGE = MAX_RANGE_PITCH
                    MIN_RANGE = MIN_RANGE_PITCH
                else:
                    axis_crt = axis_crt**MAPPING_POWER
                    MAX_RANGE = MAX_RANGE_RY
                    MIN_RANGE = MIN_RANGE_RY
                
                int_size = (MAX_RANGE-MIN_RANGE)/2 - 1
                if axis[i]==1:
                    cal_control_val = (int)(-(axis_crt-1)*int_size)+MIN_RANGE
                else:
                    cal_control_val = (int)((axis_crt+1)*int_size+MIN_RANGE)
                control_val.append(cal_control_val)
            #print(control_val)
            #print(control_val)
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