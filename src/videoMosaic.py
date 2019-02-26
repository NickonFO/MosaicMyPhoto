import cv2 as cv
from PIL import Image

# Work on video frame by frame to create the video mosaic

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


def replace_image_with_avgs(videoFile,scaleFactor):
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



def main():

    playVideo("concert2.gif")
main()
