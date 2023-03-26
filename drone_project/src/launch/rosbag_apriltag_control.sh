#!/bin/bash
DATETIME=$date +"%Y%m%d_%H%M%S"
rosbag record -O flight_DATETIME /turtle1/cmd_vel /turtle1/pose