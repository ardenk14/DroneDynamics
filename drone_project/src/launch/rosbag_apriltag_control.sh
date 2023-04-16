#!/bin/bash
DATETIME=$date +"%Y%m%d_%H%M%S"
rosbag record -O flight_DATETIME /sent_drone_commands /image_stream/image /image_stream/camera_info