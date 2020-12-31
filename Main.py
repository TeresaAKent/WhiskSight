# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:42:12 2020

@author: tkent
"""

import sys
import os
from random import randint
import numpy as np
import cv2
from numpy import genfromtxt
from VideoOutputs import *
from TrackerwIdentification import RunTracker
## Section of the code you might want to modify regularly
# Arrow multiplier magnifies the movement of the dots arrows on the output video.
# It is used for better visulization
ArrowMultiplier=10



fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Set video to load
# It is important you include the type of file in the name of the video eg. .mp4

def RunMain(FillInName):
    videoPath = "../GitHubUpload/InputVideos/{}.mp4".format(FillInName)
    
    
    #Output folder name
    OutputFolder="../GitHubUpload/Output/"
    if not os.path.exists(OutputFolder):
        os.mkdir(OutputFolder)
        print("Directory " , OutputFolder ,  " Created ")
    else:    
        print("Directory " , OutputFolder,  " already exists")
    
    #The number in this file name is if you want to try out different thresholds
    # Just change this number or replace it with a text so you know what thresholds you tested
    OutputfileName=FillInName+'1'
    
    
    # CSV Files for the csv which tracks the centers of all tracked dots
     
    OutputCSVFileName="{}{}Centers.csv".format(OutputFolder,OutputfileName)
    
    # Helper Image outputs an image of the inital frame
    # Each of the tracked dots has a number which will be on the image and correlate to the label on the csv
    HelperImageFileName="{}{}HelperIMG.png".format(OutputFolder,OutputfileName)
    
    
    
    # OutputVideo_ is a true false value where 1 will output a video and 0 will not output a video
    # Output 1 is the video with the arrows on it
    # I highly caution against having this code always outputing videos and never deleting them
    # It will take up space on your hardrive fast
    OutputVideo1=1

    Video1OutputName="{}{}ArrowVideo.mp4".format(OutputFolder,OutputfileName)

    
    params={}
    # Section of the code that you probably do not want to modify on a whim
    FinalData,FinalDataSizes=RunTracker(videoPath,HelperImageFileName,params)
    
    
    # Calculate the Movement of Each of the Dots
    MovementXandY=FinalData[:,1:-2]-FinalData[0,1:-2]
    params['MovementXandY']=MovementXandY
        
    # Output the csv file with point position
    np.savetxt(OutputCSVFileName, FinalData,delimiter=',', header=params['TitleString'])
    
    framesPerSecond=params['framesPerSecond']
    
    if OutputVideo1==1:
        ArrowVisualization(Video1OutputName,fourcc,framesPerSecond,ArrowMultiplier,params)
        
# FillInName="MultiStimuli"
# RunMain(FillInName)

FillInName="FanOn2WithContact"
RunMain(FillInName)

# FillInName="RollingContact"
# RunMain(FillInName)

# FillInName="FanOn1WithContact"
# RunMain(FillInName)