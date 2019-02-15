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

    """
    This part of method accesses the colour of the pixels in each block

    """
################################################################################

    all_pixels = np.empty((0,3),int)

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
                    print(all_pixels)
            #        print(temp_tuple)
            #        temp_tuple = temp_tuple + all_pixels
            #col_avg = temp_tuple / cf

                    #count = count + 1
            #print(all_pixels.shape)


            #print(col_pixels)
            #print()
            #print (all_pixels)


                    col_avg = np.mean(all_pixels, axis=0) # find average colour of each block
            #print()
            #print(col_avg)
                    #col_avg = (50,50,50)
            # Place average colour of block in the image

            for x in range(sf):
                for y in range(sf):
                    img[x + w,h + y] = col_avg

            #z = z + sf
            if h+2*sf > height:
                break;
        if w + 2*sf > width:
            break;



######################################################################




        #if w + sf >= width - sf:
        #    break;
################################################################################


    cv.imshow("img",img)


def main():
   img = cv.imread(r'C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\Original Dog image.png')
   #img_Array = np.array(r'C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\Original Dog image.png')
  # avg = getAverageRGB(img)
   # w =900, h = 1440
#   width,height,channels = img.shape[:3]
 #  strides = (width* size, size, width*size,size)
   splitImage(img,50)
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
  # cv.imshow("Original Dog image.png",img)

   cv.waitKey(0)



main()

# I want something that loops through the image rows and cols, finds the average
# RGB for that pixel and replaces the pixel with the average value
