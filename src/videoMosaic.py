import time
import argparse
import os
import sys
import glob
import imageio

import numpy as np
import cv2 as cv
from PIL import Image
#from split import replace_image_with_avgs

# Work on video frame by frame to create the video mosaic
# Idea: Split video into frames, process on each frame,
# put back together.

# Accessor methods-----------------------------------------
def getFrameCount(videoFile):
    """ Helper method to calculate number of frames
        of an image """
    cap = cv.VideoCapture(videoFile)
    frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    return frameCount

def getFrameWidth(videoFile):
    cap = cv.VideoCapture(videoFile)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    return width

def getFrameHeight(videoFile):
    cap = cv.VideoCapture(videoFile)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    return height
#-------------------------------------------------------
def playVideo(videoFile):
    cap = cv.VideoCapture(videoFile)
    while(cap.isOpened()):
         ret, frame = cap.read()
         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
         cv.imshow('frame',frame)
         if cv.waitKey(25) & 0xFF == ord('q'):
             break
    cap.release()
    cv.destroyAllWindows()

def splitVideoIntoFrames(videoFile):
    """
    Split video into frames and save in a directory
    """
    cap = cv.VideoCapture(videoFile)
    count = 0
    while cap.isOpened():
        ret,frame = cap.read()
        cv.imshow('frame',frame)
        cv.imwrite("frames%d.jpg" % count, frame)
        count = count + 1
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()  # destroy all the opened windows

def replace_image_with_avgs(Image,scaleFactor):
    """
    Feature extraction -
    accesses the colour of the pixels in each block,
    replaces tiles with average RGB
    """
    sf = scaleFactor
    all_pixels = np.empty((0,3),int)
    img = np.array(Image)
    width, height, col_pixels = Image.shape[:3] # W,h, channels
    no_pixels = width * height
    #x_block = [x*no_pixels for x in range(sf)]
    #while count < no_blocks:
    print(width)
    print(height)
    #for w in range(0, width, sf):
    for w in range(0, width, sf):
        for h in range(0, height, sf):
            all_pixels = np.empty((0,3),int)
            #temp_tuple = np.empty((49,3),int)
            for x in range(sf):
                #print(z+x)
                for y in range(sf):

                    col_pixel= [img[x + w, h + y]] # Read pixel colour
                    all_pixels = np.concatenate((all_pixels,col_pixel),axis=0)

                    col_avg = np.mean(all_pixels, axis=0) # Averaging is O(no_blocks) #2mins # find average colour of each block
            #print()
        #    print(col_avg)

            # Place average colour of block in the image

            for x in range(sf):
                for y in range(sf):
                    img[x + w,h + y] = col_avg

            #z = z + sf
            if h+2*sf > height:
                break;
        if w + 2*sf > width:
            break;


    # Save image of avgs
    #cv.imwrite(r'C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\VideoFrames\AVGS.jpg',img)
    cv.imshow("img",img)



def processOnVideoFrames(fileDir):
    """
    Process on the frames individually
    """
    for (i, image) in enumerate(glob.iglob(fileDir)):
        replace_image_with_avgs(image, 8)

def create_gif(input_imgDir, output_Dir):
    """
    Join individual frames to form a video
    """
    images = []
    filenames = os.listdir(input_imgDir)
    for filename in filenames:
        if filename.endswith('.jpg'):
            path = os.path.join(input_imgDir,filename)
            images.append(imageio.imread(path))
    imageio.mimsave(output_Dir, images)

def main():

    #playVideo("concert2.gif")
    #getFrameCount("concert2.gif")
    #splitVideoIntoFrames("concert2.gif")
    #processOnVideoFrames(r"C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\VideoFrames\*.jpg")
    #create_gif(r"C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\VideoFrames","C:\\Users\\NFO\\Desktop\\Uni\\3rd Year\\3rd year project rescources\\Code\\VideoFrames\\movie.gif")
main()
