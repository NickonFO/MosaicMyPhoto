import numpy as np
import scipy as sc
import cv2 as cv
import matplotlib as plt
from matplotlib import pyplot
import pylab as pyl
import PIL
import glob
import os
import imageio
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
#    blocks = Image.fromarray(blocks)
    #pyplot.imshow(img, interpolation='nearest')
    #pyplot.show()
    #no_blocks = 3
    #print (blocks[1,1])
    #print()
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
    col_avgs = []
    all_pixels = np.empty((0,3),int)
    img = np.array(Image)
    width, height, col_pixels = Image.shape[:3] # W,h, channels
    no_pixels = width * height
    #x_block = [x*no_pixels for x in range(sf)]
    #while count < no_blocks:
    #print(width)
    #print(height)
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
                    col_avgs = np.concatenate((col_avgs,col_avg),axis=0)
            #print()



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
    #cv.imwrite(r'C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\AVGS.jpg',img)
    #cv.imshow("img",img)
    #return img
    return col_avgs

def readTileImages(imageDir):
    """
    Gather Tiles:
    Read list of images from a directory and put in list
    """
    images = [] # List to store all tile image files
    files = os.listdir(imageDir)

    for file in files:
        #path = os.path.abspath(os.path.join(imageDir,file))
        path = os.path.abspath(os.path.join(imageDir,file))
        image = Image.open(path)
        #images = np.concatenate((images,image), axis=0)
        images.append(image)
        #print(images)
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

def resizeImage2(Image,sf):
    """
    Function to resize an image into a square block,
    by scalefactor "sf".
    """
    resized_image = cv.resize(Image, (sf, sf))
    return resized_image

def resizeImage(img,sf):
    """
    Function to resize an image into a square block,
    by scalefactor "sf".
    """
    img = img.resize((sf,sf), Image.ANTIALIAS)
    return img



def createImageGrid(images,width,height,col,row):
    """
    Creates a grid of resized tile images.
    """
    p_width = width // col
    p_height = height // row
    size = p_width, p_height

    # Create blank canvas RGB image
    new_im = np.zeros((width,height, 3), np.uint8)

    new_im2 = np.zeros((width,height, 3), np.uint8)
    new_im3 = np.zeros((width,height, 3), np.uint8)
    # rows = []
    # for i in range(row)
    #     rows[i] =

    #new_im = np.zeros(width,height)
    imgs = []
    for p in images:
        tmp = resizeImage(p, p_width)
        imgs.append(tmp)

    #i = 0
    x = 0
    y = 0

    # for c in range(col):
    #     for r in range(row):
    #         print(i, x, y)
    #         new_im.paste(imgs[i], (x, y))
    #         i += 1
    #         y += p_height
    #     x += p_width
    #     y = 0

    #for i in range(1, len(imgs)-1):

    rows = []
    k = 0
    # for all photos (replace 9 with dynamic variable)
    for i in range(9):
        if i % col == 0: # if you are done with the row
            if k > 0:
                rows.append(cur_row)

            cur_row = imgs[i]
            k += 1
        else:
            cur_img = imgs[i]
            cur_row = np.hstack(cur_row, cur_img)

        #collage = rows[0]

        for i in range(1, len(rows)):
            collage = np.vstack([collage, rows[i]])

    #return collage




    cv.imshow("im",collage)
        #imgs.append()
    #grid_img = Image.new('RGB', (n*width, m*height))


def createGrid(images, w, dims, imageSize):
  """
  Create a grid of images given a list of tiles.
  """
  m = dims
  n = w
  width = imageSize
  height = imageSize
  # sanity check
  #assert m*n == len(images)

  resizedImages = []
  for img in images:
      resizedImages.append(resizeImage(img,imageSize))

  # create empty image to overlay tiles on
  grid_img = Image.new('RGB', (n*width, m*height))

  # paste images
  for index in range(len(resizedImages)):
    row = int(index/n)
    col = index - n*row
    grid_img.paste(resizedImages[index], (col*width, row*height))

  return grid_img


# copied
def tileMatchAlgorithm(in_avg, avgs):
    """
    Algorithm to find the index of the best matching tile using Euclidian distance
    """
    avg = in_avg
    i = 0
    min_i = 0
    dist = 0
    min_dist = float("inf")
    for val in avgs:
        # euclidian distance function
        dist = ((val[0] - avg[0])*(val[0] - avg[0]) +
            (val[1] - avg[1])*(val[1] - avg[1]) +
            (val[2] - avg[2])*(val[2] - avg[2]))

        #np.linalg.norm(val - avg)
    if dist < min_dist:
      min_dist = dist
      min_i = i
    i += 1
    return min_i

def mosaic(target_image, tile_images):


#def mosaicCreator(target_image, tile_images):
#    """
#    Creates the photomosaic.
#    """
    # output images
#    output_images = []
    # Empty canvas to overlay mosaic
    #mosaic = np.zeros((target_image.shape))

    # divide source image
    #target_image = splitImage(target_image,8)

    # Get a list of average RGB's from directory
    #tile_avgs = getAVGsInDir(tile_images)

    # get list of averages for target image
    #avg = replaceImageWithAvgs(target_image,8)
    #for img in tile_images:
        #img = cv.imread(img)
        #tile_avgs = averageRGB(img)
                # target sub-image average



        # find match index
        #match_index = tileMatchAlgorithm(avg, tile_avgs)
        #output_images.append(tile_images[match_index])

    #mosaic = createGrid(output_images,8)


    return mosaic
    # for each block
    #for i in
    # get tile averages
    #getAVGsInDir(tile_images)





def main():
    img = cv.imread('GermanShep.jpeg')
    images = readTileImages(r'C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\apmw_flowers')

    #cv.imshow(splitImage(img,8))
    grid = createGrid(images,50, 36,30)
    grid.show()
    #mos = mosaicCreator(img,images)
    #mos.show()


    # imagesSameSize = []
    #
    # for i in range(len(images)):
    #     imagesSameSize[i] = resizeImage(images[i],100)
  # createImageGrid(images,900,900,3,3)
    #createImageGrid2(imagesSameSize,300)
  # resizeImage(img,300)

  # mosaicCreator(img,r'C:\Users\NFO\Desktop\Project\Test flowers')
   #imshow(img)
   #averageRGB(img)
   #getAVGsInDir(r"C:\Users\NFO\Desktop\Project\Test flowers")
   #mosaicCreator(img)

   #splitImage(img,50)
   #replaceImageWithAvgs(img,8)
   #readTileImages(r'C:\Users\NFO\Desktop\Uni\3rd Year\3rd year project rescources\Code\Test flowers')
    cv.waitKey(0)
main()
# I want something that loops through the image rows and cols, finds the average
# RGB for that pixel and replaces the pixel with the average value
