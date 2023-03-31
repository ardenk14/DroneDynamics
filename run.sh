#!/bin/bash
# Connect to drone wifi



#Run april Tags
gnome-terminal --title ="April Tags" -- source ~/catkin_ws/devel_isolated/apriltag_ros/setup.bash -- roslaunch apriltag_ros continuous_detection.launch
sleep 2
#Launch Controller through Roslaunch
gnome-terminal --title="Controller Launch" -- source ~/catkin_ws/devel/setup.bash -- roslaunch drone_project control_drone.launch






# 

