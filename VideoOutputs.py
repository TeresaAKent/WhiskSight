# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:00:43 2020

@author: tkent
"""

import sys
from random import randint
import numpy as np
import cv2

def ArrowVisualization(Video1OutputName,fourcc,framesPerSecond,ArrowMultiplier,params):
    numberOfBoxes=params['numberOfBoxes']
    FinalData=params['FinalData']
    FramesToVideo=params['FramesToVideo']
    InitalXCenters=params['InitialXCenters']
    InitalYCenters=params['InitialYCenters']
    Wbboxes=params['WhiskerLocations']
    ClosestDots=params['ClosestDots'].astype(int)
    WhiskerIndexes=params['WhiskerIndexes'].astype(int)

    
    # video=cv2.VideoWriter(Video1OutputName,fourcc,framesPerSecond,(np.size(FramesToVideo,1),np.size(FramesToVideo,0)))
    # InitalPositions=(FinalData[10,:]).flatten()
    # for j in range(0,np.size(FramesToVideo,3)-1):
    #     ArrowFrame=FramesToVideo[:,:,:,j].copy()
    #     ArrowFrame=np.ascontiguousarray(ArrowFrame, dtype=np.uint8)
    #     if j==0:
    #         video.write(np.uint8(ArrowFrame))
    #     else:
    #         for k in range(numberOfBoxes):
    #             XCenter=int(FinalData[j,k*2+1])
    #             YCenter=int(FinalData[j,k*2+2])
    #             ArrowX=XCenter+ArrowMultiplier*int(XCenter-InitalPositions[k*2+1])
    #             ArrowY=YCenter+ArrowMultiplier*int(YCenter-InitalPositions[k*2+2])
    #             cv2.arrowedLine(ArrowFrame,(XCenter,YCenter),(ArrowX,ArrowY),(255,255,255),5,tipLength=.4)
    #         # cv2.imshow( "Display window", img);
    #         # cv2.waitKey(0)
    #         video.write(np.uint8(ArrowFrame))
            
    #     if j%50==0:
    #         print(j/np.size(FramesToVideo,3))
            
    
    # video.release()
    # cv2.destroyAllWindows()
    video=cv2.VideoWriter(Video1OutputName+"identification.mp4",fourcc,framesPerSecond,(np.size(FramesToVideo,1),np.size(FramesToVideo,0)))
    InitalPositions=FinalData[10,:]
    # Movement is a jxnumDots Array
    Movement=FinalData-InitalPositions
    SignChangeCount=0
    for j in range(0,np.size(FramesToVideo,3)-1):
        if j%50==0:
            print('Video Output Percentage', j/np.size(FramesToVideo,3))
            #print(SignChangeCount)
            
        airflow=0
        inertia=0
        ArrowFrame2=FramesToVideo[:,:,:,j].copy()
        ArrowFrame2=np.ascontiguousarray(ArrowFrame2, dtype=np.uint8)
        if j<20:
            video.write(np.uint8(ArrowFrame2))
        else:
            # Go through each of the whiskers
            Last10x=Movement[j-20:j+1,WhiskerIndexes*2+1]-Movement[j,WhiskerIndexes*2+1]
            Last10y=Movement[j-20:j+1,WhiskerIndexes*2+2]-Movement[j,WhiskerIndexes*2+2]
            Last10=np.square(Last10x)+np.square(Last10y)
            
            Last10DirectionChangesX=Movement[j-20:j,WhiskerIndexes*2+1]-Movement[j-19:j+1,WhiskerIndexes*2+1]
            Last10DirectionChangesY=Movement[j-20:j,WhiskerIndexes*2+2]-Movement[j-19:j+1,WhiskerIndexes*2+2]
            
            
            Airflow=np.zeros((np.size(WhiskerIndexes)))
            for l in range (np.size(WhiskerIndexes)):
                asignX=np.sign(Last10DirectionChangesX[:,l])
                asignY=np.sign(Last10DirectionChangesY[:,l])
                signchangeX = ((np.roll(asignX, 1) - asignX) != 0).astype(int)
                signchangeY = ((np.roll(asignY, 1) - asignY) != 0).astype(int)
                SignChangeCount=signchangeX.sum()+signchangeY.sum()
                WhiskerMovement2=Movement[j,WhiskerIndexes[l]*2+1:WhiskerIndexes[l]*2+3]
                xMovement=0
                yMovement=0
                for k in range(np.size(ClosestDots,1)):
                    index2=int(ClosestDots[l,k])
                    #print(index2)
                    xMovement+=Movement[j,index2*2+1]
                    yMovement+=Movement[j,index2*2+2]
                # Average the Movement of the Closest Points to that whisker
                XaverageMovement=xMovement/np.size(ClosestDots,1)
                YaverageMovement=yMovement/np.size(ClosestDots,1)
                # First determine if it looks like airflow
                # Look at the past 10 values within 2 pixels
                numClose=(abs(Last10[:,l])<4).sum()                
                # for the values within 2 pixels how many significant changes are there
                Stdv=np.std(Last10[:,l])
                # pick a threshold and call that air flow
                # put a blue box around it if its airflow
                #print('WhiskerNum',l,numClose,Stdv)
                
                if SignChangeCount>10 and numClose>10 and np.linalg.norm(WhiskerMovement2)>3 and np.linalg.norm(WhiskerMovement2)>=3*np.linalg.norm([XaverageMovement,YaverageMovement]):
                    xCenter=FinalData[j,WhiskerIndexes[l]*2+1]
                    yCenter=FinalData[j,WhiskerIndexes[l]*2+2]
                    cv2.rectangle(ArrowFrame2,(int(xCenter-50),int(yCenter-50)), (int(xCenter+50),int(yCenter+50)),(128,50,0),3)
                    if WhiskerIndexes[l]!=WhiskerIndexes[l-1]:
                        airflow+=1
                    Airflow[l]=1
        
                # Second determine if it looks like inertia or contact
                # Look at the movement of the whisker
                
                # Subtract that from the movement of the whisker
                WhiskerMovement2=Movement[j,WhiskerIndexes[l]*2+1:WhiskerIndexes[l]*2+3]
                if np.linalg.norm(WhiskerMovement2)>3*np.linalg.norm([XaverageMovement,YaverageMovement]) and np.linalg.norm(WhiskerMovement2)>3 and Airflow[l]==0:
                    xCenter=FinalData[j,WhiskerIndexes[l]*2+1]
                    yCenter=FinalData[j,WhiskerIndexes[l]*2+2]
                    cv2.rectangle(ArrowFrame2, (int(xCenter-50),int(yCenter-50)), (int(xCenter+50),int(yCenter+50)),(100,255,200),3)
                elif np.linalg.norm(WhiskerMovement2)<=3*np.linalg.norm([XaverageMovement,YaverageMovement]) and np.linalg.norm(WhiskerMovement2)>=2.5:
                    xCenter=FinalData[j,WhiskerIndexes[l]*2+1]
                    yCenter=FinalData[j,WhiskerIndexes[l]*2+2]
                    cv2.rectangle(ArrowFrame2, (int(xCenter-50),int(yCenter-50)), (int(xCenter+50),int(yCenter+50)),(120,0,128),3)
                    if WhiskerIndexes[l]!=WhiskerIndexes[l-1]:
                        inertia+=1
                # Look at the movement of the k closest dots compared to location at frmae 10
                # Are they the same with in a threshold?
                # If yes its inertia put a purple box on it
                # If no is the whisker moving above a certain threshold
                #if yes contact put a green box around it
            if airflow>=3:
                cv2.putText(ArrowFrame2, 'Air Flow', (50,250), cv2.FONT_HERSHEY_SIMPLEX , 10, (128,50,0),thickness=10)
                Averagex=0
                Averagey=0
                for l in range (np.size(WhiskerIndexes)):
                    Averagex+=Movement[j,WhiskerIndexes[l]*2+1]
                    Averagey+=Movement[j,WhiskerIndexes[l]*2+2]
                Averagex=Averagex/np.size(WhiskerIndexes)
                Averagey=Averagey/np.size(WhiskerIndexes)
                for l in range(np.size(WhiskerIndexes)):
                    if Averagex-Movement[j,WhiskerIndexes[l]*2+1]>5:
                        cv2.rectangle(ArrowFrame2, (int(xCenter-50),int(yCenter-50)), (int(xCenter+50),int(yCenter+50)),(100,255,200),3)
                    elif Averagey-Movement[j,WhiskerIndexes[l]*2+2]>5:
                        cv2.rectangle(ArrowFrame2, (int(xCenter-50),int(yCenter-50)), (int(xCenter+50),int(yCenter+50)),(100,255,200),3)
                        
            elif inertia>=3:
                cv2.putText(ArrowFrame2, 'Inertia', (200,250), cv2.FONT_HERSHEY_SIMPLEX , 10, (120,0,128),thickness=10)
            video.write(np.uint8(ArrowFrame2))
    video.release()
    cv2.destroyAllWindows()
                

    # For each frame

   
def TrackerVerification(Video2OutputName,fourcc,framesPerSecond,params):
    colors=params['colors']
    numberOfBoxes=params['numberOfBoxes']
    FinalData=params['FinalData']
    FinalDataSizes=params['FinalDataSizes']
    FramesToVideo=params['FramesToVideo']
        
    video=cv2.VideoWriter(Video2OutputName,fourcc,framesPerSecond,(np.size(FramesToVideo,1),np.size(FramesToVideo,0)))
    
    jUpperBound=int(np.size(FramesToVideo,3))
    for j in range(0,jUpperBound):
        img=FramesToVideo[:,:,:,j]
        img=np.ascontiguousarray(img, dtype=np.uint8)
        if j==0:
            video.write(np.uint8(img))
        else:
            for k in range(numberOfBoxes):
                XCenter=int(FinalData[j,k*2+1])
                YCenter=int(FinalData[j,k*2+2])
    
                SizeX=FinalDataSizes[j,k*2]
                SizeY=FinalDataSizes[j,k*2+1]
    
                p1 = (int(XCenter-SizeX/2), int(YCenter-SizeY/2))
                p2 = (int(XCenter+SizeX/2), int(YCenter+SizeY/2))
    
                cv2.rectangle(img,p1,p2,colors[k],2,1)
                #centroid_step.append(cent)
                img=cv2.circle(img,(XCenter,YCenter),10,colors[k])
                # print(int(i),int(cent[0]),int(cent[1]))
                TextString="{}".format(k)
                img=cv2.putText(img,TextString,(p2[0],p1[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
            video.write(np.uint8(img))
        if j%50==0:
            print('Video Output Percentage',j/np.size(FramesToVideo,3))
    video.release()
    cv2.destroyAllWindows()


    