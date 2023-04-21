#!/usr/bin/env python
"""
Process a csv file containing April tag detections and states (position/orientation) and perform the following:
1) Find the velocity by taking the difference between state values 
2) Interpolate the states to get a state for each command


# USAGE EXAMPLE:
python3 process_csv.py csv_directory
python3 process_csv.py csv_directory csv_name1 csv_name2
python3 process_csv.py ~/Downloads/bagfile_csvs
python process_csv.py ../bagfiles/bagfile_csvs

IDEAS:
What if we keep only the rows with states?
The high sample rate might be too fast (not enough change between samples)
What is the effect of cubic spline vs interpolation?
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation , RotationSpline, Slerp

LARGE_TIMEGAP_THRESHHOLD = 0.1 # [s] (note that csvs are in nanoseconds by default)

# TODO: Make more modular and replace vvv with data_processing_utils.py 

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

    ### interpolate the commands columns
    df = df.set_index('time')     # set the index to time if doing interp by index
    df['commands[0]'] = df['commands[0]'].interpolate(method='index')
    df['commands[1]'] = df['commands[1]'].interpolate(method='index')
    df['commands[2]'] = df['commands[2]'].interpolate(method='index')
    df['commands[3]'] = df['commands[3]'].interpolate(method='index')
    # convert the commands columns to integers:
    df['commands[0]'] = df['commands[0]'].astype(int)
    df['commands[1]'] = df['commands[1]'].astype(int)
    df['commands[2]'] = df['commands[2]'].astype(int)
    df['commands[3]'] = df['commands[3]'].astype(int)

    ### interpolate positions
    df['position.x'] = df['position.x'].interpolate(method='cubic') #method='index'
    df['position.y'] = df['position.y'].interpolate(method='cubic')
    df['position.z'] = df['position.z'].interpolate(method='cubic')
    df = df.reset_index() # reset the index to be the row number

    ### Get orientations
    # We want to represent the orientation as a 6D vector, 
    # constructed by converting the quaternion to a rotation matrix,
    # taking the first two columns, and then flattening and concatenating them.
    # We need to do two things:
    # 1) interpolate the quaternions to all the commands (but use scipy Slerp to keep the quaternion norm = 1)
    # 2) convert the quaternions to rotation matrix and then to 6D vector representation

    # create a dataframe containing only the rows with states (ie. where command_state_flag == 1)
    df_states = df[df['command_state_flag'] == 1]
    # create a Rotation object from the quaternions
    orient = Rotation.from_quat(df_states[['orient.x', 'orient.y','orient.z','orient.w']])#.values)
    state_times = df_states['time'].values #times that these orientations correspond to
    # create a Slerp object from the state_times and the orientations
    slerp = Slerp(state_times, orient)
    # get the full column of times (which are all the values to which we want to interpolate)
    # these need to be between the first and last state times
    times = df['time'].values
    # remove the times before the first state time and after the last state time
    times = times[(times >= state_times[0]) & (times <= state_times[-1])]
    # get the interpolated orientations
    rotation_interp = slerp(times)

    # convert the rotation object to a matrix
    try:
      rotation_matrix = rotation_interp.as_matrix() #(n, 3, 3)
    except AttributeError:
      rotation_matrix = rotation_interp.as_dcm()
    # flatten the matrix
    col1 = rotation_matrix[:,:,0]
    col2 = rotation_matrix[:,:,1]
    stack = np.concatenate((col1, col2), axis=1) # (n, 6)

    # # get the first two columns of the rotation matrix
    # rotation_matrix_2 = rotation_matrix[:,:,0:2] # (n, 3, 2)
    # stack = rotation_matrix_2.reshape(-1,6) # (n, 6)

    # add the flattened matrix to the dataframe
    # (put them in at the timestamps of state_times[0], and then pad the rest with NaNs)
    rot_df = pd.DataFrame(stack, columns=['R11','R21','R31','R12','R22','R32'])
    rot_df['time'] = times
    df = pd.merge_asof(df, rot_df, on='time', direction='nearest')

    ### Also add euler angles, just to play with
    # convert the rotation object to euler angles
    euler = rotation_interp.as_euler('xyz', degrees=False)
    # add the euler angles to the dataframe
    euler_df = pd.DataFrame(euler, columns=['roll', 'pitch', 'yaw'])
    # merge the euler angles into the dataframe at the correct times
    euler_df['time'] = times
    df = pd.merge_asof(df, euler_df, on='time', direction='nearest')

    ### Add velocity columns (time in seconds)
    df['vel.x'] = df['position.x'].diff() / (df['time'].diff()) 
    df['vel.y'] = df['position.y'].diff() / (df['time'].diff()) 
    df['vel.z'] = df['position.z'].diff() / (df['time'].diff()) 
    # can't do the same for angular velocity
    
    ### Get angular velocity
    # use scipy RotationSpline to get angular velocity from orientation (scipy objects we created above)
    spline = RotationSpline(state_times, orient)
    ang_vel = spline(times, 1)
    df[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']] = pd.DataFrame(ang_vel, columns=['ang_vel_x', 'ang_vel_y', 'ang_vel_z'])
    # angular velocity is a 3D vector, whereas position is not a vector, so it is actually ok to not convert to 6D rotation matrix representation

    # time,commands[0],commands[1],commands[2],commands[3],position.x,position.y,position.z,orient.x,orient.y,orient.z,orient.w,command_state_flag,R11,R21,R31,R12,R22,R32,vel.x,vel.y,vel.z,ang_vel_x,ang_vel_y,ang_vel_z
    # make the dataframe have only the following columns (in order, without quaternions):
    df = df[['time','commands[0]','commands[1]','commands[2]','commands[3]',
             'position.x','position.y','position.z',
             "R11","R21","R31","R12","R22","R32",
             'vel.x','vel.y','vel.z',
             'ang_vel_x','ang_vel_y','ang_vel_z',
             'command_state_flag', 'roll', 'pitch', 'yaw']]

    # get rid of any remaining rows containing Nans (vel doesn't have values at start and end)
    df = df.dropna()

    # Save the interpolated DataFrame back to a CSV file
    df.to_csv(file_to_write, index=False)
    print(f"Wrote {file_to_write}")