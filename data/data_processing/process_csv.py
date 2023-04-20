#!/usr/bin/env python
"""
Process a csv file containing April tag detections and states (position/orientation) and perform the following:
1) Find the velocity by taking the difference between state values 
2) Interpolate the states to get a state for each command


# USAGE EXAMPLE:
python3 process_csv.py csv_directory
python3 process_csv.py csv_directory csv_name1 csv_name2
python3 process_csv.py ~/Downloads/bagfile_csvs

IDEAS:
What if we keep only the rows with states?
The high sample rate might be too fast (not enough change between samples)
What is the effect of cubic spline vs interpolation?
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation , RotationSpline

LARGE_TIMEGAP_THRESHHOLD = 0.1 # [s] (note that csvs are in nanoseconds by default)

# get cvs directory
directory = sys.argv[1]
if not directory.endswith("/"):
  directory += "/"

# create list of .csv's to scan
filenames = []
if len(sys.argv) > 2:
  for filename_arg in sys.argv[2:]:
    if not filename_arg.endswith(".csv"):
      filename_arg = filename_arg + ".csv"
    
    f = os.path.join(directory, filename_arg)
    if os.path.isfile(f) and filename_arg[-4:] == 'csv':
      print(f"Found file: {f}")    
      filenames.append(filename_arg)
else:
  print(f"Reading all .csv files in {directory}")
  for filename_arg in os.listdir(directory):

    f = os.path.join(directory, filename_arg)
    if os.path.isfile(f) and filename_arg[-4:] == '.csv':
      print(f"Found csvfile: {f}")    
      filenames.append(filename_arg)

# make directory for results
results_dir = directory + "../processed_csvs"
if not os.path.exists(results_dir):
  os.makedirs(results_dir)


for filename in filenames:
    df = pd.read_csv(directory + filename)
    file_to_write = results_dir +"/processed_"+filename
    print(f"Adding velocity and interpolating the states in {file_to_write}")

    # # remove every nth command (but leave the states)
    # n = 2
    # df = df[df.index % n == 0 and df['position.x'].notna()]
    
    # Interpolate the states to get a state for each command
    # (commands are at faster rate than the states)
    # At each command, fill in the states by interpolating (by time) between the previous and next state

    df = df.set_index('time')     # set the index to time if doing interp by index
    # interpolate the commands columns
    df['commands[0]'] = df['commands[0]'].interpolate(method='index')
    df['commands[1]'] = df['commands[1]'].interpolate(method='index')
    df['commands[2]'] = df['commands[2]'].interpolate(method='index')
    df['commands[3]'] = df['commands[3]'].interpolate(method='index')
    # convert the commands columns to integers:
    df['commands[0]'] = df['commands[0]'].astype(int)
    df['commands[1]'] = df['commands[1]'].astype(int)
    df['commands[2]'] = df['commands[2]'].astype(int)
    df['commands[3]'] = df['commands[3]'].astype(int)

    # interpolate positions
    df['position.x'] = df['position.x'].interpolate(method='cubic') #method='index'
    df['position.y'] = df['position.y'].interpolate(method='cubic')
    df['position.z'] = df['position.z'].interpolate(method='cubic')
    # interpolate orientations (is this fair to do or should I use scipy?)
    df['orient.x'] = df['orient.x'].interpolate(method='cubic')
    df['orient.y'] = df['orient.y'].interpolate(method='cubic')
    df['orient.z'] = df['orient.z'].interpolate(method='cubic')
    df['orient.w'] = df['orient.w'].interpolate(method='cubic')

    df = df.reset_index() # reset the index to be the row number

    # Add velocity columns (time in seconds)
    df['vel.x'] = df['position.x'].diff() / (df['time'].diff()) 
    df['vel.y'] = df['position.y'].diff() / (df['time'].diff()) 
    df['vel.z'] = df['position.z'].diff() / (df['time'].diff()) 
    # can't do the same for angular velocity
    
    # # get rid of any remaining rows containing Nans (breaks quaternion to euler conversion)
    # (do this after taking diff())
    df = df.dropna()
    # convert quaternions to euler angles for orientation 
    orient = Rotation.from_quat(df[['orient.x', 'orient.y','orient.z','orient.w']])#.values)
    euler_orient = orient.as_euler('xyz', degrees=False)
    df[['roll', 'pitch', 'yaw']] = pd.DataFrame(euler_orient, columns=['roll', 'pitch', 'yaw']) #add to dataframe
    
    # get angular velocity: Use scipy to create rotation objects and then 
    # rotation spline to get ang_velocity from angle. 
    times = df['time'].values
    spline = RotationSpline(times, orient)
    ang_vel = spline(times, 1)
    df[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']] = pd.DataFrame(ang_vel, columns=['ang_vel_x', 'ang_vel_y', 'ang_vel_z'])

    # encode angle as first two columns of rotation matrix flattened, concat

    # time,commands[0],commands[1],commands[2],commands[3],position.x,position.y,position.z,orient.x,orient.y,orient.z,orient.w,command_state_flag,vel.x,vel.y,vel.z,ang_vel.x,ang_vel.y,ang_vel.z,ang_vel.w,roll,pitch,yaw,ang_vel_x,ang_vel_y,ang_vel_z
    # make the dataframe have only the following columns (in order, without quaternions):
    df = df[['time','commands[0]','commands[1]','commands[2]','commands[3]','position.x','position.y','position.z','roll','pitch','yaw','vel.x','vel.y','vel.z','ang_vel_x','ang_vel_y','ang_vel_z','command_state_flag']]

    # # get rid of any remaining rows containing Nans (breaks quaternion to euler conversion)
    df = df.dropna()

    # Save the interpolated DataFrame back to a CSV file
    df.to_csv(file_to_write, index=False)
    print(f"Wrote {file_to_write}")