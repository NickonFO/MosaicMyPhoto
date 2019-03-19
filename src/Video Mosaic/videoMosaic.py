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
    frameCount = 0
    cap = cv.VideoCapture(videoFile)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    while(True):
        ret,frame = cap.read()
        if not ret:
            break
        frameCount += 1
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
    print("Splitting video into individual frames...")
    #print
    cap = cv.VideoCapture(videoFile)
    count = 0
    while cap.isOpened():
        ret,frame = cap.read()
        cv.imshow('frame',frame)
        cv.imwrite("frames%d.jpg" % count, frame)
        count += 1
        if cv.waitKey(5) & 0xFF == ord('q'):
            break;
        if count >= getFrameCount(videoFile):
            break;
        #    pass
    cap.release()
    cv.destroyAllWindows()  # destroy all the opened windows


def replaceImageWithAvgs(Image,scaleFactor):
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

    for w in range(0, width, sf):
        for h in range(0, height, sf):
            all_pixels = np.empty((0,3),int)

            for x in range(sf):

                for y in range(sf):

                    col_pixel= [img[x + w, h + y]] # Read pixel colour
                    all_pixels = np.concatenate((all_pixels,col_pixel),axis=0)
                    #col_avg = (50,50,32)
                    col_avg = np.mean(all_pixels, axis=0) # Averaging is O(no_blocks) #2mins # find average colour of each block

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
    return img
    cv.imshow("img",img)

def processOnVideoFrames(imgDir, scl):
    print("Processing on video frames....")
    count = 0
    for img in glob.iglob(imgDir + r'\*.jpg'):
        image = cv.imread(img)
        new_image = replaceImageWithAvgs(image,scl)
        cv.imwrite("Avg_frames%d.jpg" % count, new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        count = count + 1

def deleteFrames(imgDir):
    """
    remove the redundant frames
    """
    print("Deleting redundant frames...")
    files = os.listdir(imgDir)
    for file in files:
        if file.endswith(".jpg"):
            os.remove(file)


def createGif(input_imgDir, output_Dir):
    """
    Join individual frames to form a video
    """
    print("Creating gif....")
    images = []
    filenames = os.listdir(input_imgDir)
    for filename in filenames:
        if filename.startswith('Avg_frames'):
            if filename.endswith('.jpg'):
                path = os.path.join(input_imgDir,filename)
                images.append(imageio.imread(path))
    imageio.mimsave(output_Dir, images)

def main():
    #parser = argparse.ArgumentParser()
    # Arguments
    #parser.add_argument('--target-video', dest = 'target_video', required = True)
    #parser.add_argument('--input-directory', dest = 'input_directory', required = True)
    #parser.add_argument('--scale-factor', dest = 'scale_factor', required = False)
    #parser.add_argument('--output-file', dest='outfile', required=True)
    #args = parser.parse_args()

    #splitVideoIntoFrames(target_video)
    #processOnVideoFrames(input_directory,8)
    #createGif(input_directory, 'movie.gif')
    #deleteFrames(input_directory)
    #splitVideoIntoFrames("psychcat.gif")
    #processOnVideoFrames(r"C:\Users\NFO\Desktop\Project\Video frames", 8)
    #createGif(r"C:\Users\NFO\Desktop\Project\Video frames","movie.gif")
    #deleteFrames(r"C:\Users\NFO\Desktop\Project\Video frames")

    #### INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-video', dest= 'target_video',required = True)
    parser.add_argument('--input-dir',dest = 'input_directory',required = True)
    parser.add_argument('--scale-factor',dest ='scl',type=int,required=True)
    parser.add_argument('--output-file', dest= 'output',required=True)

    args = parser.parse_args()
    splitVideoIntoFrames(args.target_video)
    processOnVideoFrames(args.input_directory,args.scl)
    createGif(args.input_directory,args.output)
    deleteFrames(args.input_directory)

main()
