# Using neural networks
# SSD Mobilenet v3 model

import cv2
from djitellopy import tello
import cvzone

confidence = 0.6
nmsThreshold = 0.2

# here 0 is the camera index(if multiple cameras use 1,2 ..)
capture = cv2.VideoCapture(0)
# here param '3' is open cv constants for width
capture.set(3, 1920)
# here param '4' is open cv constants for height
capture.set(4, 1080)
# here param '5' is open cv constants for fps
capture.set(5, 60)


classNames = []
classFile = 'coco.names'

# open the coco.names file and add it to the list
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

# path
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'


# loading the pretrained weights and config files into the network
# dnn = deep neural network
network = cv2.dnn_DetectionModel(weightsPath, configPath)
# setting predefined standard parameters as per documentation

network.setInputSize(320, 320) # the input image will be resized to 320 by 320

# { each image has a range of [0,255] meaning every pixel(either Red Green or Blue) in an image
# can contain 0 to 255 values (higher the value more brighter or intense the color)
# for neural networks to function better we need to normalize(make the image less bright and less saturated or colorful)
# so achieve this we setInputScale(1.0 /127.5) meaning, every pixel in an image fed to the network is divided
# by 127.5 this will scale down the pixel value and make the image less intense and saturated
# (but not black and white)  }
# eg: lets say we have an image with values (100, 150, 200) after inputscale we get (0.78, 1.18, 1.57)
network.setInputScale(1.0 / 127.5)

# after the inputscale compresses the image R G and B values we subtract the mean values (127.5) from each of the channel
# so the image RGB values will be (-126.72, -126.32, -125.93)
# this transforms the image into a neutral gray
network.setInputMean(127.5)
# swapping the red and blue channels
network.setInputSwapRB(True)
# loop
while True:
    # read returns a tuple with two elements
    # success is boolean if true (frame was successfully captured)
    # if false(frame was not captured)
    # img contains the actual image captured
    success, img =capture.read()
    # .detect method of the dnn_detectionModel will return and array of 3 elements and accepts 3 arguments
    # img: is the image taken by the camera
    #confThreshold : is the confidence value (0 to 1 )
    # nmsThreshold(non-maximum suppression) : used to remove duplicate bounding boxes with lower confidence values
    # for the same object
    # here nmsThreshold is set to 20%, this means if two boxes for the same object overlaps by more than 20% then the one
    # with the lower confidence value is removed
    # -----------------------------------------------------------------------------------------
    # classIds ( refer coco.names) : if a person detected returns classIds = 1
    # confidenceScores:  the 2nd element contains confidence scores
    # boundingBox: The third element of the array is the box coordinates that bounds the identified object.
    classIds, conf, bbox  = network.detect(img, confThreshold = confidence, nmsThreshold = nmsThreshold)

    # this for loop iterates over three array simultaneously using the zip method
    # the zip method creates a tuple that stores the elements from each of the array in any given instance
    # and stores their values in these three forloop variables
    # .flatten() is used to convert and 2d array into 2d array for easier access
    try:
        for classId, conf, box in zip(classIds.flatten(), conf.flatten(), bbox):
            # print(f' object code {classId}\n confidence value : {conf} box dimesions: {box}')
            cvzone.cornerRect(img, box)
            cv2.putText(img, f' {classNames[classId-1].upper()} {round(conf*100,2)}',
                        (box[0] +10, box[1]+40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255))
    except Exception as e:
        print(f'Error: {e}')
    cv2.imshow("Image", img)
    cv2.waitKey(1) # adding 1ms delay




