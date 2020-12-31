

Modifications Located in the TrackerwIdentification code:

If you create a system which might need different thresholds for identification, at line 271-280 uncomment the plt commands.
The Hue, Saturation, and Value channel of each image will plot in order to help determine the best thresholds.

Line 432 in the code: If you uncomment this line the first frame of the video will appear with the boxes around each tracked dot.
On this frame you can draw boxes around the dots you do not want tracked and then press space twice to have them removed.
This can speed up the code or remove whiskers if you are only interested in a few.

At line 361: "for j in range(6)" you can change the k value
The k value determines the number of nearest neighbors associated with each whisker on the bottom of the magnet.