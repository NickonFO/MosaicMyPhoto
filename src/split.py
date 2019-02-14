import numpy as np
import cv2 as cv
import matplotlib as plt
from matplotlib import pyplot
import pylab as pyl
from PIL import Image
from PIL import ImagePalette
from statistics import mean
#https://stackoverflow.com/questions/20368413/draw-grid-lines-over-an-image-in-matplotlib
#https://stackoverflow.com/questions/19062875/how-to-get-the-number-of-channels-from-an-image-in-opencv-2

# Downsample the image and replace tiles with average rgb tiles
# Get the average rgb
def getAverageRGB(image):
  """
  Given PIL Image, return average value of color as (r, g, b)
  """
  # get image as numpy array
  im = np.array(image)
  # get shape
  w,h,d = im.shape
  # get average
  return tuple(np.average(im.reshape(w*h, d), axis=0))

# split the image into tiles
"""def splitImage(image, width, height):
    width = image.size[0]
    height = image.size[1]

    for i in range (0,width):
        for j in (0,height):
            current_col = image.getpixel((i,j))
            image.putpixel((x,y),getAverageRGB(image))
    return image

"""
def splitImage(Image,scaleFactor):
    """
    Method to split image into blocks and display pixel values of the blocks
    using np array and strides
    """
    img = np.array(Image)
    sf = scaleFactor
    width, height, col_pixels = Image.shape[:3] # W,h, channels
    shape = (height - sf+1, width - sf + 1, sf, sf)
    size = img.itemsize
    strides = (width* size, size, width*size,size)
    blocks = np.lib.stride_tricks.as_strided(img, shape = shape, strides=strides)
    #pyplot.imshow(img, interpolation='nearest')
    #pyplot.show()
    print (blocks[1,1])
    print()

    """
    This part of method accesses the colour of the pixels in each block

    """

    all_pixels = np.empty((0,3),int)
    for x in range(sf):
        for y in range(sf):
            col_pixel= [img[x,y]] # Read pixel colour

            all_pixels = np.concatenate( (all_pixels,col_pixel),axis=0)
    print(all_pixels.shape)
    print (all_pixels)
    col_avg = np.mean(all_pixels, axis=0)
    for x in range(sf):
        for y in range(sf):
            img[x,y] = col_avg
    pyplot.imshow(img, interpolation='nearest')
    pyplot.show()

def main():
   img = cv.imread(r'C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\Original Dog image.png')
   #img_Array = np.array(r'C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\Original Dog image.png')
  # avg = getAverageRGB(img)
   # w =900, h = 1440
#   width,height,channels = img.shape[:3]
 #  strides = (width* size, size, width*size,size)
   splitImage(img,100)
  # new_col = [0,0,0]
   #no_pixels = width * height
   #scale_Factor = 8

   # Make a grid
   #dx,dy = scale_Factor
   # Go through each pixel,

  # for i  in range (0,width,scale_Factor):
    #   for j in range (0,height,scale_Factor):

    #       channels_xy = img[i,j] # Access colour of pixel in image
    #       img[i,j] = (0,0,0)

           #channels_xy = tuple(img.mean(axis=0))  # change colour to average color


          # pixels[i,j] = getAverageRGB(img)
          # r,g,b = img.getpixel((i,j))
           #img.putpixel((i,j),55)



  # print(no_pixels)
   cv.imshow("Original Dog image.png",img)

   cv.waitKey(0)



main()

# I want something that loops through the image rows and cols, finds the average
# RGB for that pixel and replaces the pixel with the average value
