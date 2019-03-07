import numpy as np
import scipy as sc
import cv2 as cv
import matplotlib as plt
from matplotlib import pyplot
import pylab as pyl
import PIL
import glob
import os
from PIL import Image
from PIL import ImagePalette
from statistics import mean
#https://stackoverflow.com/questions/20368413/draw-grid-lines-over-an-image-in-matplotlib
#https://stackoverflow.com/questions/19062875/how-to-get-the-number-of-channels-from-an-image-in-opencv-2
#https://stackoverflow.com/questions/13167269/changing-pixel-color-python

def splitImage(Image,scaleFactor):
    """
    Method to split image into blocks and display pixel values of the blocks
    using np array and strides
    """
    img = np.array(Image)


    sf = scaleFactor
    width, height, col_pixels = Image.shape[:3] # W,h, channels
    no_pixels = width * height
    print(no_pixels)
    no_blocks = no_pixels / (sf * sf)
    shape = (height - sf+1, width - sf + 1, sf, sf)
    size = img.itemsize
    strides = (width* size, size, width*size,size)
    blocks = np.lib.stride_tricks.as_strided(img, shape = shape, strides=strides)
    #pyplot.imshow(img, interpolation='nearest')
    #pyplot.show()
    #no_blocks = 3
    print (blocks[1,1])
    print()
    #print(width / 8)
    #print(height/ 8)
    #count = 0
    return blocks

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
                    #print(all_pixels)
            #        print(temp_tuple)
            #        temp_tuple = temp_tuple + all_pixels
            #col_avg = temp_tuple / cf

                    #count = count + 1
            #print(all_pixels.shape)


            #print(col_pixels)
            #print()
            #print (all_pixels)


                    #sumRGB = [(x[0]*x[1][0], x[0]*x[1][1], x[0]*x[1][2]) for x in all_pixels]
                    #col_avg = tuple([sum(x)/no_pixels for x in zip(*sumRGB)])
                    #col_avg = (50,50,50) # 10 seconds
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
    cv.imwrite(r'C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\AVGS.jpg',img)
    cv.imshow("img",img)
    return img

def readTileImages(imageDir):
    """
    Gather Tiles:
    Read list of images from a directory and put in list
    """
    images = [] # List to store all tile image files
    files = os.listdir(imageDir)

    for file in files:
        path = os.path.abspath(os.path.join(imageDir,file))
        image = Image.open(path)
        images.append(image)
        print(images)
    return images

def averageRGB(Image):
    """
    Given a single Image, return avg colour (r,g,b)
    """
    img = np.array(Image)
    width, height, col_pixels = img.shape[:3]
    img.shape = (width * height,col_pixels)

    return tuple(np.mean(img,axis=0))

def getAVGsInDir(imgDir):
    """
    Gets the average RGB of tiles in the tile directory and appends them to a
    list
    """
    avgs = []
    for img in glob.iglob(imgDir + r'\*.jpg'):
        image = cv.imread(img)
        avg = averageRGB(image)
        avgs.append(avg)
    return avgs


#def tileMatchAlgorithm(in_avg, avg):
#    in_avg = 1


def main():
   img = cv.imread('Original Dog image.png')
   #averageRGB(img)
  # getAVGsInDir(r"C:\Users\NFO\Desktop\Project\Test flowers")
   splitImage(img,50)
   #replaceImageWithAvgs(img,8)
   #readTileImages(r'C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\Test flowers')
   cv.waitKey(0)
main()
# I want something that loops through the image rows and cols, finds the average
# RGB for that pixel and replaces the pixel with the average value
