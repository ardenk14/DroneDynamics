"""
One problem with the data we have is that the commands come in at a very high frequency, 
so there are large gaps between the states, and furthermore it is hard for the multistep model 
to look far enough into the future to get meaningful changes in the states.

This file downsamples a processed csv file by removing intermediate commands until a desired sample rate is achieved.

USAGE: python downsample_csv.py ../bagfiles/processed_csvs
"""

import pandas as pd
from data_processing_utils import get_directory_and_filenames

def downsample_csv(directory, filename, sample_rate=10):
    """
    Downsamples a processed csv file by removing intermediate commands until a desired sample rate is achieved.
        :param directory: The directory of the csv file to downsample
        :param filename: The name of the csv file to downsample
        :param sample_rate: The desired sample rate in Hz
        Writes the downsampled csv to the same directory as the original csv, with the prefix "{Hz}hz_".
    """
    df = pd.read_csv(directory + filename)
    df2 = df.copy()
    file_to_write = directory + f"/{sample_rate}hz_" + filename
    print(f"Downsampling from {filename} to {file_to_write}")

    # remove every nth command (but leave the rows where command_state_flag == 1 (original states))
    sec_per_sample = 1/sample_rate
    prev_time = df['time'][0]
    # loop through the rows and remove any that are less than sec_per_sample away from prev_time, unless it is a state
    for index, row in df.iterrows():
        if row['time'] - prev_time < sec_per_sample and row['command_state_flag'] != 1:
            # print(f"Removing row {index} with time {row['time']}")
            df2 = df2.drop(index)
        else:
            prev_time = row['time']

  # save the downsampled csv
    df2.to_csv(file_to_write, index=False)


if __name__ == "__main__":
    directory, filenames = get_directory_and_filenames(suffix=".csv")
    sample_rate = 50 #Hz
    for filename in filenames:
        # if the first character is a number, it is a downsampled csv
        if filename[0].isdigit():
            continue
        downsample_csv(directory, filename, sample_rate=sample_rate)