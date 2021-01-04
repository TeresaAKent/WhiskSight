Note from the author:

This code was uploaded as part of a ICRA 2021 submission. The code takes in a video from the WhiskSight sensor and outputs the following:
-An excel file with all the tracked points of interest for the duration of the video
-A picture which visualizes the location of each of the numbered tracked points
-A video of an algorithm which predicts the stimuli (contact, inertia, airflow) applied to the WhiskSight.

Any questions about this code can be directed to tkent@andrew.cmu.edu.  
If anyone would like a copy of a video from the WhiskSight sensor to play with the code or the parameters, the authors are happy to oblige a request by email.
-------------------------------------------------------------------------
To run this code you will need to download the following codes: 

Main.py
TrackerwIdentification.py
VideoOutputs.py

In addition the following python libraries need to be installed on your machine:
numpy
OpenCV
matplotlib
skimage
random
----------------------------------------------------------------------------
The purpose of each of the author written codes is as follows:

Main: A shorter code which is most user friendly.  This code is the one which should be run.  
Simply change the videoPath line to the location of the input video.
Then change the OutputFolder to your desired folder, the program is also capable of creating a new folder of specified name.

TrackerwIdentification.py: Contained in this file are all of the functions used to take in a video file and output the location of all dots and whiskers in each frame.  This code also generates the location of all of the whiskers and their closest dots for later.

VideoOutputs.py: This code creates output videos.  Three possible videos can be made although the main code only calls one of them.
ArrowVisualization: Adds Arrows to the tracked dots which demonstrate their movement
AlgorithmVisualization: Demonstrates the algorithms ability to predict the cause of deflections to the membrane and the whisker
TrackerVerification: Places numbered boxes around each of the tracked points for verification the tracker is working well.  The numbers are the same as those output in the excel file.
----------------------------------------------------------------------------------------------------------------------------------------
Modifications Located in the TrackerwIdentification code:

If you create a system which might need different thresholds for identification, at line 271-280 uncomment the plt commands.
The Hue, Saturation, and Value channel of each image will plot in order to help determine the best thresholds.

At line 361: "for j in range(6)" you can change the k value
The k value determines the number of nearest neighbors associated with each whisker on the bottom of the magnet.

Line 432 in the code: If you uncomment this line the first frame of the video will appear with the boxes around each tracked dot.
On this frame you can draw boxes around the dots you do not want tracked and then press space twice to have them removed.
This can speed up the code or remove whiskers if you are only interested in a few.


Line 412: Change which of the OpenCV trackers is being used