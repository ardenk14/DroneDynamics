#!/usr/bin/env python
''' controlSender.py

    This is a ROS node for reading in drone control commands
    and sending them to the drone.

    Author: Arden Knoll
'''
import rospy
from drone import Drone
from std_msgs.msg import UInt8MultiArray
from threading import Lock

class drone_controller():

    def __init__(self):
        rospy.init_node('drone_controls', anonymous=False)
        self.drone = Drone()
        rospy.sleep(1)
        self.lock = Lock()
        self.sub = rospy.Subscriber('/drone_commands', UInt8MultiArray, self.send_cmd) 
        rospy.loginfo("Initialized")
        while True:
            self.lock.acquire()
            self.drone.send()
            #print("SENT")
            self.lock.release()
            rospy.sleep(0.002)

    def send_cmd(self, msg):
        # Set drone state to match that of the last message
        cmd = msg.data # Size of list is 4: roll, pitch, thrust, yaw
        self.lock.acquire()
        self.drone.set_cmd(cmd)
        self.lock.release()
        # Send command
        #self.drone.send()
        #rospy.loginfo("SENT CMD")


if __name__ == '__main__':
    dr = drone_controller()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


# Subscribe to drone command topic
# Update a shared variable between this thread and another


# Initialize drone object and ROS node

# while True:
#   drone.send()
#   rospy.sleep()