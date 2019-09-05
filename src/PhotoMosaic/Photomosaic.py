import numpy as np
import argparse
import scipy as sc
import cv2 as cv
import matplotlib as plt
from matplotlib import pyplot
import pylab as pyl
import PIL
import glob
import os
import imageio
from gooey import *
from PIL import Image
from PIL import ImagePalette
from statistics import mean
#https://stackoverflow.com/questions/20368413/draw-grid-lines-over-an-image-in-matplotlib
#https://stackoverflow.com/questions/19062875/how-to-get-the-number-of-channels-from-an-image-in-opencv-2
#https://stackoverflow.com/questions/13167269/changing-pixel-color-python

def splitImage2(Image,scaleFactor):
    """
    Method to split image into blocks and display pixel values of the blocks
    using np array and strides.
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
    return blocks


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
    return tuple(np.mean(img,axis=(0,1)))


def getAVGsInDir(imgDir):
    """
    Gets the average RGB of jpeg tiles in the tile directory and appends them to a
    list.
    """
    avgs = []
    for img in glob.iglob(imgDir + '/*.jpg'):
        image = Image.open(img)
        avg = averageRGB(image)
        avgs.append(avg)

    return avgs

def resizeCvImage(Image,sf):
    """
    Function to resize a cv image into a square block,
    by scalefactor "sf".
    """
    resized_image = cv.resize(Image, (sf, sf))
    return resized_image

def resizeImage(img,sf):
    """
    Function to resize a PIL image into a square block,
    by scalefactor "sf".
    """
    img = img.resize((sf,sf), Image.ANTIALIAS)
    return img


def createGrid(images, w, h, imageSize):
  """
  Create a grid of images given a list of tiles.
  """
  m = h
  n = w
  width = imageSize
  height = imageSize
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


def tileMatchAlgorithm(avg, avgs):
  """
  Use Euclidian distance equation to find min distance between RGB values
  and return the index of the best matching tile.
  """
  i = 0
  min_i = 0
  min_dist = float("inf")
  for tup in avgs:
    euclid = (((tup[0]-avg[0])**2) + ((tup[1] - avg[1])**2) + ((tup[2]-avg[2])**2))
    if euclid < min_dist:
      min_dist = euclid
      min_i = i
    i += 1

  return min_i

def splitImage(image, blockdims):
  """
  Splits image into blocks and returns a list of sub images
  """
  # list of sub images
  sub_images = []

  width = image.size[0]
  height = image.size[1]
  m = blockdims
  n = blockdims
  w, h = int(width/n), int(height/m)

  # generate list of dimensions
  for j in range(m):
    for i in range(n):
      # add cropped images to array
      sub_images.append(image.crop((i*w, j*h, (i+1)*w, (j+1)*h)))
  return sub_images

def mosaicMaker(image, tile_dir, tile_size, gridWidth, gridHeight):
    """
    Create the photomosaic.
    """
    # Create a list of images for the output
    output = []
    # Create a list of tile images
    print("Reading from tile database....")
    tiles = readTileImages(tile_dir)

    print("Dividing source image....")
    # divide the source image
    image_blocks = splitImage(image, tile_size)

    # Get a list of averages for the Tiles
    tile_avgs = getAVGsInDir(tile_dir)
    # get averages of the target image blocks
    print("Matching average RGB's ....")
    for block in image_blocks:
        block_avg = averageRGB(block)
        # find the index of the best matching block with the tile
        index = tileMatchAlgorithm(block_avg,tile_avgs)
        #print(index)
        output.append(tiles[index])
    print("Creating Mosaic...")
    mosaic = createGrid(output, gridWidth, gridHeight, tile_size)

    return mosaic
@Gooey
def main():

    #### INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    #parser = GooeyParser()
    parser.add_argument("--tiles", dest ='tiles', type=str, required=True,help="Folder of tile images")
    #parser.add_argument("tiles",help="Folder of tile images", widget ='FileChooser')
    parser.add_argument("--target-image",dest='target',type=str,required=True,help="Target image of mosaic")
    parser.add_argument("--tile-size",dest='tile_size',type=int,required=True,help="size of your tiles")
    #parser.add_argument("--grid-size",dest='grid_size',type=int,required=True,help="grid size")
    parser.add_argument("--output-file",dest='output',type=str,required=True,help="output file")
    args = parser.parse_args()
    img = Image.open(args.target)

    #img, args,tilesize,grid,grid
    mosaic = mosaicMaker(img,args.tiles,args.tile_size ,args.tile_size,args.tile_size)
    mosaic.save(args.output)
    mosaic.show()

    cv.waitKey(0)
main()
