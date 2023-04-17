#!/usr/bin/env python
# Process a ROS bag file with April tag detections into positional data for training

"""
Data collection process: 
1. Record flight data rosbag with topics:
    /clock
    /image_stream/camera_info
    /image_stream/image
    /sent_drone_commands

2. Play back this rosbag but this time run the apriltags detector to get:
    /clock
    /tag_detections 
    /sent_drone_commands
(could merge setps 1 and 2)

3. Process this new rosbag with the below script, 
which converts the above two topics into a time-stamped csv file

4. Post-process the csv to remove sections of flight with no tag detections/during crashes
and interpolate values so each timepoint has both a 

Based on: http://wiki.ros.org/rosbag/Cookbook#Python

# USAGE EXAMPLE:
source devel_isolated/setup.bash  #to get apriltag message definitions
python3 process_bagfile.py bagfile_directory
python3 process_bagfile.py bagfile_directory bagfile_name1 bagfile_name2
python3 process_bagfile.py ~/Downloads flight_4_12_bag2.bag
"""

import sys
import os
import csv
import rosbag

# get bagfile directory
directory = sys.argv[1]
if not directory.endswith("/"):
  directory += "/"

# create list of bagfiles to scan
filenames = []
if len(sys.argv) > 2:
  for filename_arg in sys.argv[2:]:
    if not filename_arg.endswith(".bag"):
      filename_arg = filename_arg + ".bag"
    
    f = os.path.join(directory, filename_arg)
    if os.path.isfile(f) and filename_arg[-4:] == '.bag':
      print(f"Found bagfile: {f}")    
      filenames.append(filename_arg)
else:
  print(f"Reading all .bag files in {directory}")
  for filename_arg in os.listdir(directory):

    f = os.path.join(directory, filename_arg)
    if os.path.isfile(f) and filename_arg[-4:] == '.bag':
      print(f"Found bagfile: {f}")    
      filenames.append(filename_arg)


# make directory for results
results_dir = directory + "bagfile_csvs" #+ filename[:-4]
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

taggroup_dict = {
  

}
def get_taggroup_global_coords(id, taggroup_dict):
  """
  @args 
  id: list of all ids in this apriltag bundle
  @ return
  global_coord: global position of tag group in world frame [x,y,z,qx,qy,qz,qw]
  """
  global_coord = [0,0,0,0,0,0,0]


  return global_coord
  


for filename in filenames:
  bag = rosbag.Bag(directory + filename)
  csv_filepath = results_dir +"/"+filename+'_commands_states.csv'
  print(f"Writing tag_detections and sent_drone_commands in {filename} to CSV")
  
  with open(csv_filepath, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    data_writer.writerow(['time','commands[0]','commands[1]','commands[2]','commands[3]',"position.x", "position.y", "position.z", "orient.x", "orient.y", "orient.z", "orient.w"])
    #TODO: what are drone_commands [0],[1],etc?

    for topic, message, timestamp in bag.read_messages(topics=['/sent_drone_commands', '/tag_detections']):
      # print(f"topic: {topic}, time:{timestamp}")
      if topic=='/sent_drone_commands' or topic=='sent_drone_commands':
  #       /sent_drone_commands  std_msgs/UInt8MultiArray
          d = message.data
          data_writer.writerow([timestamp, d[0], d[1], d[2], d[3], "", "", "", "", "", "", ""])
      elif topic=='/tag_detections' or topic=='tag_detections':
        # message contains an array of pose wrt each visible group
        # AprilTagDetectionArray
          # std_msgs/Header header
          # AprilTagDetection[] detections
            # int32[] id
            # float64[] size
            # geometry_msgs/PoseWithCovarianceStamped pose
            #   std_msgs/Header header
            #   geometry_msgs/PoseWithCovariance pose
            #     float64[36] covariance
            #     geometry_msgs/Pose pose
            #       geometry_msgs/Point position
            #       geometry_msgs/Quaternion orientation
          try: 
            # assume we only have one tag group
            pos = message.detections[0].pose.pose.position #leave as wall wrt drone
            orient = message.detections[0].pose.pose.orientation
          except:
            pos = message.pose.pose.position
            orient = message.pose.pose.orientation
          
          data_writer.writerow([timestamp, "", "", "", "",pos.x, pos.y, pos.z, orient.x, orient.y, orient.z, orient.w])
  bag.close()
  print(f"Wrote {csv_filepath}")



# parameters for writerow function:
            # @param topics: list of topics or a single topic. if an empty list is given all topics will be read [optional]
          # @type  topics: list(str) or str
          # @param start_time: earliest timestamp of message to return [optional]
          # @type  start_time: U{genpy.Time}
          # @param end_time: latest timestamp of message to return [optional]
          # @type  end_time: U{genpy.Time}





