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
from util import *
from VideoOutputs import *
from TSandHTrackerwIdentification import RunTracker
from DisplacmentGelSight import *
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
    OutputFolder="../MRLWhiskerResearch-master/ClassificationWork/"
    if not os.path.exists(OutputFolder):
        os.mkdir(OutputFolder)
        print("Directory " , OutputFolder ,  " Created ")
    else:    
        print("Directory " , OutputFolder,  " already exists")
    
    #Using the updated system this is all you should have to change the way the files are set up below they will name themself to include the Output file name
    OutputfileName=FillInName+'5'
    
    
    
    # Four CSV Files one gives the centers second gives the Sizes, third outputs force and moment information, 4th detects the movements
    # Third is called forces so that we don't have tons of confusion issues with Movement and Moment
    OutputCSVFileName="{}{}Centers.csv".format(OutputFolder,OutputfileName)
    
    # Helper Image outputs an image of the inital frame
    # with the boxes the same color as the tracking
    # and a number next to each of the boxes showing the order you selected them in
    HelperImageFileName="{}{}HelperIMG.png".format(OutputFolder,OutputfileName)
    
    
    
    # OutputVideo_ is a true false value where 1 will output a video and 0 will not output a video
    # Output 1 is the video with the arrows on it
    # Output 2 is a tracker video you can rewatch to ensure the tracker did not mess up 
    # Output 3 is the video with the forces and moments added
    # I highly caution against having this code always outputing videos and never deleting them
    # It will take up space on your hardrive fast
    # you have to ouput the force and moment video in order to get the force and moment csv if you don't like this lmk and I can change it
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
        # Right now this is only built for one whisker
        # I am happy to update it for multiple whiskers at a later date
        ArrowVisualization(Video1OutputName,fourcc,framesPerSecond,ArrowMultiplier,params)
        
FillInName="MultiStimuli"
RunMain(FillInName)
FillInName="FanOn2WithContact"
RunMain(FillInName)
# FillInName="RollingContact"
# RunMain(FillInName)
FillInName="FanOn1WithContact"
RunMain(FillInName)