CS 510 PA1

Group Members:

Ben Fraj
Joshua Gillham
Kaden Strand
Isaac Wang

python version : 2.7 
opencv version : 2.4
requires numpy python module

How to run: python transform.py source points output flag

Argument descriptions:
source = input video filename
points = point correspondence text filename
output = output video filename
flag = optional character argument to specify additional video processing.
Valid characters are 'f', 'g', and 'p'

Regardless of whether arg4 is given, the program will apply the 
corresponding transformation according to the number of points in the 
point correspondence file, and write output video to filename given in arg3.

Example: "python transform.py in.avi points.txt out.avi"
This command will apply the appropriate point-correspondence transformation 
given in "points.txt" to the video "in.avi" and write the transformed video to "out.avi"
Note: We tested using integer pixel point correspondences
----------
If arg4 = 'f': Outputs additional Fourier transformation plot + edge detection + feature detection.
Example: "python transform.py in.avi points.txt out.avi f"
In addition to the point transform output, the magnitude plot of the input video will be written to 
"outmagnitude.avi", the edge detect result of high pass filtering will be written to "outedges.avi",
and the result of a corner find function will be written to "outcorners.avi"
---------- 
If arg4 = 'g': Apply Gaussian blurring to the image for scaling to half size.
Example: "python transform.py in.avi points.txt out.avi g"
In addition to the point transform output, the results of the Gaussian blur will be 
written to "outGaussianFiltered.avi" 
----------
If arg4 = 'p': Scale the video by 1, 2 and 3 octaves.
Example: "python transform.py in.avi points.txt out.avi p"
In addition to the point transform, the results of the image pyramid will be 
written to "out-pyramid-octave1.avi", "out-pyramid-octave2.avi", and "out-pyramid-octave3.avi"
Note: Depending on the codecs used, small octave videos may fail to save (videos too small)
----------
For each of the optional arguments, additional output files will be written in the video format given by arg3.
Example: "python transform.py in.flv points.txt out.flv f"
The files "out.flv", "outmagnitude.flv", "outedges.flv", and "outcorners.flv" will be written.
