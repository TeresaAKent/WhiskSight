# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:07:09 2020

@author: tkent
"""
import sys
from random import randint
import numpy as np
import cv2
import time
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt


def SelectBoxes(frame,bboxes,colors,params):
    while True:
        frame1=frame.copy()
        frame2=frame.copy()
        frame3=frame.copy()
        LongText='Starting from the top left corner of the desired box, \n drag a box centered at the whisker center, \n when happy with the drawn box press the space button then press x \n'
        ShortText='See Console: Pick Whisker Center, Press Space, Press X'
        frame1=cv2.putText(frame1,ShortText,(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
        print(LongText)
        while True:
          # draw bounding boxes over objects
          # selectROI's default behaviour is to draw box starting from the center
          # when from Center is set to false, you can draw box starting from top left corner
          bbox = cv2.selectROI('MultiTracker', frame1)
          bboxes.append(bbox)
          colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
          S=cv2.waitKey(0) & 0xFF
          if (S == 120): #x is pressed
              break
        NumWhiskers=np.size(bboxes,0)
    
    
        LongText='\n Starting from the top left corner of the desired box, \n drag a box centered at one of the points in the outer ring of dots, \n when happy with the drawn box press the space button \n to select another box press the space bar a second time \n on the last box selection only press the space bar once then press O \n'
        ShortText='See Console: Pick Outer Ring Dots Centers, Pick One, Press Space'
        ShortText2='To Pick Another Press Space again, Done Outer Ring? Press O'
        frame2=cv2.putText(frame2,ShortText,(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
        frame2=cv2.putText(frame2,ShortText2,(50,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
        print(LongText)
        while True:
          # draw bounding boxes over objects
          # selectROI's default behaviour is to draw box starting from the center
          # when from Center is set to false, you can draw box starting from top left corner
          bbox = cv2.selectROI('MultiTracker', frame2)
          bboxes.append(bbox)
          colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
          S=cv2.waitKey(0) & 0xFF
          if (S == 111): #(x) is pressed
              break
        NumOuterPoints=np.size(bboxes,0)-NumWhiskers
    
    
        LongText='\n Starting from the top left corner of the desired box, drag a box centered at one of the points in the inner ring of dots, \n when happy with the drawn box press the space button \n to select another box press the space bar a second time \n on the last box selection only press the space bar once then press I \n Then Press q \n'
        ShortText='See Console: Pick Inner Ring Dots Centers, Pick One, Press Space'
        ShortText2='To Pick Another Press Space again, Done Inner Ring? Press I then Q'
        frame3=cv2.putText(frame3,ShortText,(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
        frame3=cv2.putText(frame3,ShortText2,(50,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
        print(LongText)
        while True:
          # draw bounding boxes over objects
          # selectROI's default behaviour is to draw box starting from the center
          # when from Center is set to false, you can draw box starting from top left corner
          bbox = cv2.selectROI('MultiTracker', frame3)
          bboxes.append(bbox)
          colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
          S=cv2.waitKey(0) & 0xFF
          if (S == 105): #I is pressed
              break
    
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            break
    NumInnerPoints=np.size(bboxes,0)-NumWhiskers-NumOuterPoints
    print('Selected bounding boxes {}'.format(bboxes))
    print("NumWhiskers={}, NumOuterRing={}, NumInnerRing={}".format(NumWhiskers,NumOuterPoints,NumInnerPoints))
    params['colors']=colors
    params['bboxes']=bboxes
    params['NumWhiskers']=NumWhiskers
    params['NumOuterPoints']=NumOuterPoints
    params['NumInnerPoints']=NumInnerPoints   
    
    return params

def HelperImagePlusFileHeaders(numberOfBoxes,frame,params,TotalFrames, HelperImageFileName):
    bboxes=params['bboxes']
    colors=params['colors']
    # TitleString is the header of all of the files
    TitleString='Time,'
    TitleString2='Time'
    
    # Initalize the Array which will be printed to the excel file
    # 3 extra columns are added for Time, Movement of x, Movement of y
    FinalData=np.zeros([TotalFrames,numberOfBoxes*2+3])
    #FinalSizeData is an array of widthx,heighty for each of the boxes
    FinalDataSizes=np.zeros([TotalFrames,numberOfBoxes*2+1])
    
    Initalxs=np.zeros([2,numberOfBoxes],int)
    Initalys=np.zeros([2,numberOfBoxes],int)
    
    InitalxSizes=np.zeros([2,numberOfBoxes],int)
    InitalySizes=np.zeros([2,numberOfBoxes],int)
    
    OutputIMG=frame.copy()
    for i in range(numberOfBoxes):
        TitleString=TitleString+'x{},'.format(i)
        TitleString=TitleString+'y{},'.format(i)
        TitleString2=TitleString2+'xsize {},'.format(i)
        TitleString2=TitleString2+'ysize {},'.format(i)
    
        arrayFile=bboxes[i]
        colorsArray=colors[i]
    
        Initalxs[:,i]=[arrayFile[0],arrayFile[0]+arrayFile[2]]
        Initalys[:,i]=[arrayFile[1],arrayFile[1]+arrayFile[3]]
        cv2.rectangle(OutputIMG,(Initalxs[0,i],Initalys[0,i]),(Initalxs[1,i],Initalys[1,i]),(colorsArray),2)
    
        InitalxSizes[:,i]=arrayFile[2]
        InitalySizes[:,i]=arrayFile[3]
    
        FinalData[0,i*2+1]=arrayFile[0]+arrayFile[2]/2
        FinalData[0,i*2+2]=arrayFile[1]+arrayFile[3]/2
    
        FinalDataSizes[0,i*2+1]=arrayFile[0]+arrayFile[2]/2
        FinalDataSizes[0,i*2+2]=arrayFile[1]+arrayFile[3]/2
    
        TextString="{}".format(i)
        cv2.putText(OutputIMG,TextString,(Initalxs[1,i],Initalys[0,i]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
    
    
    cv2.imwrite(HelperImageFileName,OutputIMG)
    
    FramesToVideo=np.zeros((np.size(OutputIMG,0),np.size(OutputIMG,1),np.size(OutputIMG,2),TotalFrames),int)
    FramesToVideo[:,:,:,0]=OutputIMG

    params['InitialXCenters']=(np.round(np.average(Initalxs,0))).astype(int)
    params['InitialYCenters']=(np.round(np.average(Initalys,0))).astype(int)
    
    
    # record the inital loacation of the center of all the points x and y
    params['Initialx']=np.average(Initalxs)
    params['Initialy']=np.average(Initalys)
    
    params['numberOfBoxes']=numberOfBoxes
    params['OutputIMG']=OutputIMG
    
    params['FramesToVideo']=FramesToVideo
    
    params['FinalData']=FinalData
    params['FinalDataSizes']=FinalDataSizes
    
    params['TitleString']=TitleString+'xMovement,'+'yMovement'
    params['TitleString2']=TitleString2
    
    pass    
    
def createTrackerByName(trackerType):
    trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
      tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
      tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
      tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
      tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
      tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
      tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
      tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
      tracker = cv2.TrackerCSRT_create()
    else:
      tracker = None
      print('Incorrect tracker name')
      print('Available trackers are:')
      for t in trackerTypes:
        print(t)
    return tracker
def RemoveaBoxes(Image,numberOfBoxes,params,TotalFrames, HelperImageFileName):
    #This code will only remove an extra box 
    #it will not find a box if one wasn't there because it is unlikely that it will be able to track it
    HelperImagePlusFileHeaders(numberOfBoxes,Image,params,TotalFrames, HelperImageFileName)
    frame=params['OutputIMG']
    # Image with the boxes on it should appear, 
    LongText='Starting from the top left corner of the dot to remove, \n drag a box centered at the whisker center, \n when happy with the drawn box press the space button then press x \n'
    ShortText='See Console: Pick Dot to remove, Press Space'
    ShortText2='To Pick Another Press Space again, Done Press Q'
    frame1=cv2.putText(frame,ShortText,(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
    frame1=cv2.putText(frame,ShortText2,(50,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
    print(LongText)
    bboxes=params['bboxes']
    colors=params['colors']
    print(bboxes)
    bboxes2=np.asarray(bboxes)
    bboxCenters=np.vstack(([bboxes2[:,0]+1/2*bboxes2[:,2],bboxes2[:,1]+1/2*bboxes2[:,3]])).T
    while True:
        maxOut=np.size(bboxCenters,0)
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when from Center is set to false, you can draw box starting from top left corner
        bbox = cv2.selectROI('MultiTracker', frame1)
        bbox=np.reshape(np.asarray(bbox),(1,4))
        
        #if bbox consumes the centers of any dots remove them
        deleteNum=0
        iii=0
        while (iii < maxOut):
            #print(bboxCenters[i,0])
            #print(bbox[0,0],bbox[0,2])
            #print('space')
            if int(bboxCenters[iii,0]) in range(int(bbox[0,0]),int(bbox[0,0]+bbox[0,2])):
                #print('yes')
                if int(bboxCenters[iii,1]) in range(int(bbox[0,1]),int(bbox[0,1]+bbox[0,3])):
                    #print('yes2')
                    bboxes=np.delete(bboxes,iii,0)
                    colors=np.delete(colors,iii,0)
                    bboxCenters=np.delete(bboxCenters,iii,0)
                    colors=colors.tolist()
                    bboxes=bboxes.tolist()
                    maxOut=maxOut-1
                else:
                    iii+=1
            else:
                iii+=1
        params['bboxes']=bboxes
        params['colors']=colors
        #print(bboxes)
        numberOfBoxes=np.size(bboxes,0)
        #print(numberOfBoxes)
        HelperImagePlusFileHeaders(numberOfBoxes,Image,params,TotalFrames, HelperImageFileName)
        frame1=params['OutputIMG']
        S=cv2.waitKey(0) & 0xFF
        if (S == 113): #Q is pressed
            break
    # whiskers should be marked as one color
    # dots should be marked as another
    
    # User should be able to click on anythink picked up that shouldnt be and remove it
    return bboxes, colors
def findLetters2(frame,params,margin):
    # First figure out the groups
    selem=skimage.morphology.disk(6)
    im1,im2,im3=np.shape(frame)
    Step3=np.zeros([im1,im2])
    WStep3=np.zeros([im1,im2])
    # original = skimage.img_as_float(image)
    # sigma = 0.155
    # noisy = skimage.util.random_noise(original, var=sigma**2)
    # # sigma_est = skimage.restoration.estimate_sigma(noisy, multichannel=True, average_sigmas=True)
    # Step1=skimage.restoration.denoise_bilateral(noisy,multichannel=True, sigma_spatial=15)
    Stepppp=skimage.color.rgb2hsv(frame)
    Step2=skimage.color.rgb2gray(frame)
    H=Stepppp[:,:,0]
    S=Stepppp[:,:,1]
    V=Stepppp[:,:,2]
    
    # threshold1=(H*255)
    # threshold2=(S*255)
    # threshold3=(V*255)
    # plt.figure("Threshold1")
    # plt.imshow(threshold1)
    # plt.figure("Threshold2")
    # plt.imshow(threshold2)
    # plt.figure("Threshold3")
    # plt.imshow(threshold3)
    # #plt.figure()
    
    Threshold=1/2*skimage.filters.thresholding.threshold_isodata(V)
    #print(Threshold)
    Threshold2=1.5*skimage.filters.thresholding.threshold_isodata(S)
    #print(Threshold2)
    Threshold= 100/255
    Threshold2=75/255
    #print(Threshold)
    Step3[V>Threshold]=1
    Step3[V<Threshold]=0
    WStep3[S>Threshold2]=0
    WStep3[S<Threshold2]=1
    
    Step4=skimage.morphology.binary_opening(Step3,selem)
    WStep4=skimage.morphology.binary_opening(WStep3,selem)
    Step5,numLet=skimage.measure.label(Step4,return_num=True,background=1)
    WStep5,WnumLet=skimage.measure.label(WStep4,return_num=True,background=1)
    
    bufRange=2*margin
 
    bboxes = []
    colors=[]
    for region in skimage.measure.regionprops(Step5):
        if region.area>=15:
            # create a box to add to bboxes
            minr,minc,maxr,maxc=region.bbox
            bbox=(minc-margin,minr-margin,maxc-minc+bufRange,maxr-minr+bufRange)
            bboxes.append(bbox)
            # pick a color to label it whisker or dot
            inputColor=np.average(Stepppp[minr+2:maxr-2,minc+2:maxc-2],axis=(0,1))
            #print(inputColor)
            colors.append((inputColor))
    Wbboxes = []        
    for region in skimage.measure.regionprops(WStep5):
        if region.area>=100:
            # create a box to add to bboxes
            minr,minc,maxr,maxc=region.bbox
            bbox=(minc-margin,minr-margin,maxc-minc+bufRange,maxr-minr+bufRange)
            Wbboxes.append(bbox)
            # pick a color to label it whisker or dot
            inputColor=np.average(Stepppp[minr+2:maxr-2,minc+2:maxc-2],axis=(0,1))
    # Get the index values for the whiskers
    bboxes=np.asarray(bboxes)
    Wbboxes=np.asarray(Wbboxes)
    Centers=bboxes[:,0:2]+bboxes[:,2:-1]/2
    WCenters=Wbboxes[:,0:2]+Wbboxes[:,2:-1]/2
    WhiskerIndexes=np.zeros((np.size(WCenters,0)))
    ClosestDots=np.zeros((np.size(WCenters,0),6))
    for i in range(np.size(WCenters,0)):
        SubtractionArray=np.absolute(Centers-WCenters[i,:])
        SubtractionVal=np.linalg.norm(SubtractionArray,axis=1,ord=2)
        index=np.argmin(SubtractionVal)
        WhiskerIndexes[i]=index
        SubtractionVal[index]=10000
        for j in range(6):
            index=np.argmin(SubtractionVal)
            ClosestDots[i,j]=index
            SubtractionVal[index]=10000
    params['WhiskerLocations']=Wbboxes
    params['ClosestDots']=ClosestDots
    params['WhiskerIndexes']=WhiskerIndexes
    print(WhiskerIndexes)
    return Step4, bboxes, colors

def MatchBoxes(bboxes0,bboxesNew, bboxesOrigional,colors,params):
    ColorOrigional=params['ColorsOrigional']
    ColorsOrigional=np.asarray(ColorOrigional)
    ColorsNew=colors
    n1=np.size(bboxesOrigional,0)
    while np.size(bboxesNew,0)<np.size(bboxesOrigional,0):
        array=[1000,1000,1000,1000]
        bboxesNew=np.vstack((bboxesNew,array))
    while np.size(ColorsNew,0)<np.size(ColorsOrigional,0):
        array2=[1000,1000,1000]
        ColorsNew=np.vstack((ColorsNew,array2))
        
    bboxesNewF=np.zeros((n1,4))
    # Turn the bboxes values into centers
    bboxes0Centers=np.vstack(([bboxes0[:,0]+1/2*bboxes0[:,2],bboxes0[:,1]+1/2*bboxes0[:,3]])).T
    bboxesOCenters=np.vstack(([bboxesOrigional[:,0]+1/2*bboxesOrigional[:,2],bboxesOrigional[:,1]+1/2*bboxesOrigional[:,3]])).T
    bboxesNCenters=np.vstack(([bboxesNew[:,0]+1/2*bboxesNew[:,2],bboxesNew[:,1]+1/2*bboxesNew[:,3]])).T
    
    for i in range (n1):
        # Subtract the centers of one element from all the centers of the other
        SubtractionArray1=np.absolute(bboxesNCenters-bboxes0Centers[i,:])
        SubtractionArray2=np.absolute(bboxesNCenters-bboxesOCenters[i,:])
        SubtractionArray3=np.absolute(ColorsNew-ColorsOrigional[i,:])
        SubtractionArray=np.linalg.norm(SubtractionArray1+SubtractionArray2,axis=1,ord=2)
        SubtractionArray=SubtractionArray/min(np.max(SubtractionArray),500)
        SubtractionArray4=np.linalg.norm(SubtractionArray3,axis=1,ord=2)+SubtractionArray3[:,0]
        SubtractionArray4=SubtractionArray4/min(np.max(SubtractionArray4),500)
        #print('4',SubtractionArray4,'1',SubtractionArray)
        Distance=SubtractionArray+SubtractionArray4
        #print(Distance)
        val=np.argmin(Distance)
        # print('val',val)
        # norm the x and y values
        # closest gets assigned to that position
        bboxesNewF[i,:]=bboxesNew[val,:] 
    #print(np.shape(bboxesNewF))
    return bboxesNewF
def InitalizationForAll(params, videoPath, HelperImageFileName, buffer):
    cap = cv2.VideoCapture(videoPath)
    TotalFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    framesPerSecond=cap.get(cv2.CAP_PROP_FPS)
    #print(framesPerSecond)
    framerate=1/framesPerSecond
    params['framerate']=int(framerate*1000)/1000
    params['framesPerSecond']=1/framerate
    
    # Read first frame    
    success, frame = cap.read()
    params['frame']=frame
    # quit if unable to read the video file
    if not success:
      print('Failed to read video')
      sys.exit(1)

    # Run the Find Letters Code    
    Step4, bboxes, colors=findLetters2(frame,params,buffer)
    numberOfBoxes=np.size(colors,0)
    params['bboxes']=bboxes
    params['colors']=colors
    
    # In this line of code you can modify the 
    #bboxes,colors=RemoveaBoxes(frame,numberOfBoxes,params,TotalFrames, HelperImageFileName)
    

    # Run the remove the boxes code
    
    # Specify the tracker type, Medianflow,Mosse, csrt
    trackerType = "MOSSE"
    
    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()
    n2=np.size(bboxes,0)
    # print('n2',n2)
    for i in range(n2):
      bbox=tuple(bboxes[i])
      multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    
    # Initialize the data
    # determine the number of points created
   
    
    HelperImagePlusFileHeaders(n2,frame,params,TotalFrames, HelperImageFileName)
    
    return cap, multiTracker,framerate, TotalFrames

def InitalizeOrigionalTracker(params, videoPath, HelperImageFileName):
     # Capture the video and get camera information
    cap = cv2.VideoCapture(videoPath)
    TotalFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    framesPerSecond=cap.get(cv2.CAP_PROP_FPS)
    #print(framesPerSecond)
    framerate=1/framesPerSecond
    params['framerate']=int(framerate*1000)/1000
    params['framesPerSecond']=1/framerate
    
    # Read first frame    
    success, frame = cap.read()

    # quit if unable to read the video file
    if not success:
      print('Failed to read video')
      sys.exit(1)
      
    bboxes = []
    colors = []
    
    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    SelectBoxes(frame,bboxes,colors,params)
    
    # Specify the tracker type, Medianflow,Mosse, csrt
    trackerType = "MEDIANFLOW"
    
    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()
    
    # Initialize MultiTracker
    bboxes=params['bboxes']
    for bbox in bboxes:
      print(bbox)
      multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    
    # Initialize the data
    # determine the number of points created
    colors=params['colors']
    numberOfBoxes=np.size(colors,0)
    
    HelperImagePlusFileHeaders(numberOfBoxes,frame,params,TotalFrames, HelperImageFileName)
    
    return cap, multiTracker,framerate, TotalFrames
    

def RunOrigionalTracker(params,cap,multiTracker,framerate,TotalFrames):
      
    pasttime=0
    hold=1
    
    FramesToVideo=params['FramesToVideo']
    FinalData=params['FinalData']
    FinalDataSizes=params['FinalDataSizes']
    
    #Start the timer
    #print('clock started')
    start_time = time.time()
    
    while cap.isOpened():
      success, frame = cap.read()
      if not success:
        break
    
      # get updated location of objects in subsequent frames
      success, boxes = multiTracker.update(frame)

    
      # update the time value
      pasttime=pasttime+framerate
      FinalData[hold,0]=pasttime
      FinalDataSizes[hold,0]=pasttime
    
      # zero the x and y values
      xsum=0
      ysum=0
    
      for i, newbox in enumerate(boxes):
        # print(newbox)
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cent = np.mean([p1,p2],axis=0)
    
        # Add the center x and y value of each box to the array
        FinalData[hold,i*2+1]=cent[0]
        FinalData[hold,i*2+2]=cent[1]
    
        FinalDataSizes[hold,i*2+1]=newbox[2]
        FinalDataSizes[hold,i*2+2]=newbox[3]
    
        # Recrods the x and y values to get the average
        xsum=cent[0]+xsum
        ysum=cent[1]+ysum
    
      # Add the movement of the points in the x direction and y direction
      numberOfBoxes=np.size(boxes,0)
      FinalData[hold,-2]=xsum/numberOfBoxes-params['Initialx']
      FinalData[hold,-1]=ysum/numberOfBoxes-params['Initialy']
    
      #add the frame of the multitracker to the video
      FramesToVideo[:,:,:,hold]=frame
    
      hold=hold+1
      if hold%50==0:
          print('Tracking PercentBar',hold/TotalFrames)
    
      # quit on ESC button
      if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
    print("--- %s seconds ---" % (time.time() - start_time))
    params['Time']=(time.time() - start_time)
    params['FinalData']=FinalData
    params['FinalDataSizes']=FinalDataSizes
    params['FramesToVideo']=FramesToVideo

def InitalizeSegmentationTracker(params, videoPath, HelperImageFileName):
     # Capture the video and get camera information
    cap = cv2.VideoCapture(videoPath)
    TotalFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    framesPerSecond=cap.get(cv2.CAP_PROP_FPS)
    #print(framesPerSecond)
    framerate=1/framesPerSecond
    params['framerate']=int(framerate*1000)/1000
    params['framesPerSecond']=1/framerate
    
    # Read first frame    
    success, frame = cap.read()

    # quit if unable to read the video file
    if not success:
      print('Failed to read video')
      sys.exit(1)
      
    bboxes = []
    colors = []
    
    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    SelectBoxes(frame,bboxes,colors,params)
    
    # Initialize MultiTracker
    bboxes=params['bboxes']
    
    # Initialize the data
    # determine the number of points created
    colors=params['colors']
    numberOfBoxes=np.size(colors,0)
    
    HelperImagePlusFileHeaders(numberOfBoxes,frame,params,TotalFrames, HelperImageFileName)
    
    return cap,framerate, TotalFrames

def RunSegmentationTracker(params,cap,framerate,TotalFrames):
      
    pasttime=0
    hold=1
    
    FramesToVideo=params['FramesToVideo']
    FinalData=params['FinalData']
    FinalDataSizes=params['FinalDataSizes']
    
    #Start the timer
    #print('clock started')
    start_time = time.time()
    
    bboxOld=np.asarray(params['bboxes'])
    bboxesOrigional=bboxOld
    
    while cap.isOpened():
      success, frame = cap.read()
      if not success:
        break
    
      #buffer the image segmentation tracker does not need the edges to keep track of it
      buffer=0
        # get updated location of objects in subsequent frames
      bw, bboxes, colors=findLetters2(frame,params,buffer)
      bboxes=np.asarray(bboxes)
      
      bboxes=MatchBoxes(bboxOld,bboxes,bboxesOrigional,params)
    
      #segment the image into dots and whisker 
      params['frame']=bw
      params['boxesHold']=bboxes
      bboxOld=params['boxesHold']
      
      # relate them to the origional place

    
      # update the time value
      pasttime=pasttime+framerate
      FinalData[hold,0]=pasttime
      FinalDataSizes[hold,0]=pasttime
    
      # zero the x and y values
      xsum=0
      ysum=0
      
      for i, newbox in enumerate(bboxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cent = np.mean([p1,p2],axis=0)
    
        # Add the center x and y value of each box to the array
        FinalData[hold,i*2+1]=cent[0]
        FinalData[hold,i*2+2]=cent[1]
    
        FinalDataSizes[hold,i*2+1]=newbox[2]
        FinalDataSizes[hold,i*2+2]=newbox[3]
    
        # Recrods the x and y values to get the average
        xsum=cent[0]+xsum
        ysum=cent[1]+ysum
    
      # Add the movement of the points in the x direction and y direction
      numberOfBoxes=np.size(bboxes,0)
      FinalData[hold,-2]=xsum/numberOfBoxes-params['Initialx']
      FinalData[hold,-1]=ysum/numberOfBoxes-params['Initialy']
    
      #add the frame of the multitracker to the video
      FramesToVideo[:,:,:,hold]=frame
    
      hold=hold+1
      if hold%50==0:
          print('Tracking PercentBar',hold/TotalFrames)
    
      # quit on ESC button
      if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
    print("--- %s seconds ---" % (time.time() - start_time))
    params['Time']=(time.time() - start_time)
    params['FinalData']=FinalData
    params['FinalDataSizes']=FinalDataSizes
    params['FramesToVideo']=FramesToVideo

def RunOpticalFlowTracker(params,cap,framerate,TotalFrames):
    pasttime=0
    hold=1
     #Start the timer
    print('clock started')
    start_time = time.time()
    
    lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    FramesToVideo=params['FramesToVideo']
    FinalData=params['FinalData']
    FinalDataSizes=params['FinalDataSizes']
    
    BoxesOrigional=np.asarray(params['bboxes'])
    p0=np.asarray([BoxesOrigional[:,0]+1/2*BoxesOrigional[:,2],BoxesOrigional[:,1]+1/2*BoxesOrigional[:,3]]).T
    p0=np.reshape(p0,(np.size(p0,0),1,2))
    p0=p0.astype(np.float32)
    #print(p0)
    #print(np.shape(p0),'p0')
    
    color = np.random.randint(0,255,(100,3))
    
    old_frame = params['frame']
    # need to change these
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(old_frame)
    
    FinalDataSizes[:,1:np.size(BoxesOrigional,0)+2]
    
    while(1):
        pasttime=pasttime+framerate
        FinalData[hold,0]=pasttime
        FinalDataSizes[hold,0]=pasttime
        
        # read the video
        ret,frame = cap.read()
        # turn the image to gray
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # run an optical flow algorithum
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        
        # for i,(new,old) in enumerate(zip(good_new,good_old)):
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        # img = cv2.add(frame,mask)
        
        xsum=0
        ysum=0
          
        for i, newbox in enumerate(p1):
            cent = p1[i,:,:]
            cent=np.reshape(cent,([1,2]))
        
            # Add the center x and y value of each box to the array
            FinalData[hold,i*2+1]=cent[0,0]
            FinalData[hold,i*2+2]=cent[0,1]
        
            # Recrods the x and y values to get the average
            xsum=cent[0,0]+xsum
            ysum=cent[0,1]+ysum
        
        # Add the movement of the points in the x direction and y direction
        numberOfBoxes=np.size(p1,0)
        FinalData[hold,-2]=xsum/numberOfBoxes-params['Initialx']
        FinalData[hold,-1]=ysum/numberOfBoxes-params['Initialy']
      
        #add the frame of the multitracker to the video
        FramesToVideo[:,:,:,hold]=frame
        
        hold=hold+1
        if hold%50==0:
          print('Tracking PercentBar',hold/TotalFrames)
        if hold==TotalFrames:
            break
    
        #cv2.imshow('frame',img)
        k = cv2.waitKey(300) & 0xff
        if k == 27:
            break
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    
    #print("--- %s seconds ---" % (time.time() - start_time))
    params['Time']=(time.time() - start_time)
    params['FinalData']=FinalData
    params['FinalDataSizes']=FinalDataSizes
    params['FramesToVideo']=FramesToVideo
    cv2.destroyAllWindows()
    cap.release()
    
def RunTracker(videoPath,HelperImageFileName,params):
    #buffer should have a value if you are using a motion tracking but does not need one if you are using the segmentation tracker
    # This is because the motion trackers use edge detection
    #buffer=0
    buffer=25
    
    cap, multiTracker,framerate, TotalFrames=InitalizationForAll(params, videoPath, HelperImageFileName,buffer)
    #cap, multiTracker,framerate, TotalFrames=InitalizeOrigionalTracker(params, videoPath, HelperImageFileName)
    RunOrigionalTracker(params,cap,multiTracker,framerate,TotalFrames)
    #RunSegmentationTracker(params,cap,framerate,TotalFrames)
    #RunOpticalFlowTracker(params,cap,framerate,TotalFrames)
    
    # cap,framerate, TotalFrames=InitalizeSegmentationTracker(params, videoPath, HelperImageFileName)
    # RunSegmentationTracker(params,cap,framerate,TotalFrames)
    

    FinalData= params['FinalData']
    FinalDataSizes=params['FinalDataSizes']
    
    
    return FinalData,FinalDataSizes