import time
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from datetime import datetime
import arducam_mipicamera as arducam
import pandas as pd
from statistics import mean
import keyboard

import serial




import argparse
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils



# Depth map default preset
SWS = 5
PFS = 5
PFC = 29
MDS = -30
NOD = 160
TTH = 100
UR = 10
SR = 14
SPWS = 100


log = False 
detected = False 
score_dict = {}
log_count = 0 
angle = None
score_ls = []
value = 0 


#!/usr/bin/env python3



try:
  camera_params = json.load(open("camera_params.txt", "r"))
except Exception as e:
  print(e)
  print("Please run 1_test.py first.")
  exit(-1)

def align_down(size, align):
    return (size & ~((align)-1))

def align_up(size, align):
    return align_down(size + align - 1, align)

def get_frame(camera):
    frame = camera.capture(encoding = 'i420')
    fmt = camera.get_format()
    height = int(align_up(fmt['height'], 16))
    width = int(align_up(fmt['width'], 32))
    image = frame.as_array.reshape(int(height * 1.5), width)
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
    image = image[:fmt['height'], :fmt['width']]
    return image

# Initialize the camera
camera = arducam.mipi_camera()
print("Open camera...")
camera.init_camera()
mode = camera_params['mode']
camera.set_mode(mode)
fmt = camera.get_format()
print("Current mode: {},resolution: {}x{}".format(fmt['mode'], fmt['width'], fmt['height']))

# Camera settimgs
cam_width = fmt['width']
cam_height = fmt['height']

img_width= camera_params['width']
img_height = camera_params['height']
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# camera.set_control(0x00980911, 1000)
# Implementing calibration data
print('Read calibration data and rectifying stereo pair...')
calibration = StereoCalibration(input_folder='calib_result')

# Initialize interface windows
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50,100)
# cv2.namedWindow("left")
# cv2.moveWindow("left", 450,100)
# cv2.namedWindow("right")
# cv2.moveWindow("right", 850,100)


disparity = np.zeros((img_width, img_height), np.uint8)
sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)

def stereo_depth_map(rectified_pair, detection_results):
    global log
    global log_count
    global angle
    global score_ls
    global value
    global detected 
   
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    local_max = disparity.max()
    local_min = disparity.min()
    # print('local min', local_min)
    # print('local max', local_max)
    local_max = 1200
    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    disparity_vis = utils.visualize(disparity_color, detection_results)
    # print(detection_results.detections[0].bounding_box)
    # print('detection results', detection_results)



   
    try:
        detection_score = detection_results.detections[0].classes[0].score 
        score_dict[str(angle)] = []
    except IndexError:
        detection_score = 0 

    if detection_score > 0.8:
        detected = True
    else:
        detected = False


    
    

    try:
        x1 = (detection_results.detections[0].bounding_box.origin_x) - 40
        x2 = (x1 + detection_results.detections[0].bounding_box.width) + 80
        y1 = detection_results.detections[0].bounding_box.origin_y - 40
        y2 = y1 + detection_results.detections[0].bounding_box.height  + 60

        # x1 = 0
        # x2 =  10
        # y1 = 0
        # y2 = 10
        # x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        # print(detection_results)

        # print('x1, x2, y1, y2', x1, x2, y1, y2)
       
        box = [x1, y1, x2-x1, y2-y1]
        cv2.rectangle(disparity_color,box,color=(0,255,0),thickness=2)
        rect = disparity_color[y1:y1+y2, x1:x1+x2]
        # depth_value = rect.mean()
        boxcentre = (x1 +x2)/2
        

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0

        
       
        # if boxcentre > img_width/4 :
            # print('frame in right')
         
        # elif boxcentre < img_width/4 :
            # print('frame in left')
            # test
            
        
        # print('box centre', boxcentre)
        print('target value is', (img_width/4) )
        value = boxcentre 
        print('actual val', (value) )
        # ser.write(str(value).encode()+ "\n")
       
     
        



        rect = disparity_color[y1:y1+y2, x1:x1+x2]
        ls = []
        for i in rect:
            # print(int(i.mean()))
            if int(i.mean()) > 65:
                ls.append(int(i.mean()))
        if len(ls) != 0:
            rect_filt = mean(ls)
            depth_value = rect_filt
        else:
            depth_value = 0
        print('depth value', depth_value)
       
    except IndexError:
        pass





    cv2.imshow("Image", disparity_color)
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        quit();
    return disparity_color

def load_map_settings( fName ):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
    print('Loading parameters from file...')
    f=open(fName, 'r')
    data = json.load(f)
    SWS=data['SADWindowSize']
    PFS=data['preFilterSize']
    PFC=data['preFilterCap']
    MDS=data['minDisparity']
    NOD=data['numberOfDisparities']
    TTH=data['textureThreshold']
    UR=data['uniquenessRatio']
    SR=data['speckleRange']
    SPWS=data['speckleWindowSize']    
    #sbm.setSADWindowSize(SWS)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    f.close()
    print ('Parameters loaded from file '+fName)


load_map_settings ("3dmap_set.txt")

#####################
#####################


def run(img_left, model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:


  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

#   # Start capturing video input from the camera
#   cap = cv2.VideoCapture(camera_id)
#   cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=1, score_threshold=0.8)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)


#   counter += 1
#   image = cv2.flip(image, 1)

#   # Convert the image from BGR to RGB as required by the TFLite model.
  
  rgb_image = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

  # Create a TensorImage object from the RGB image.
  
  input_tensor = vision.TensorImage.create_from_array(rgb_image)

  # Run object detection estimation using the model.
  detection_result = detector.detect(input_tensor)

  # Draw keypoints and edges on input image
  img_left = utils.visualize(img_left, detection_result)

  # Calculate the FPS
  if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
  fps_text = 'FPS = {:.1f}'.format(fps)
  text_location = (left_margin, row_size)
  cv2.putText(img_left, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,font_size, text_color, font_thickness)
  cv2.imshow('object_detector', img_left)
  

  return detection_result
  #####################
  ######################




# capture frames from the camera
# for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
detection_result = None
# ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
# ser.reset_input_buffer()
print('started')
while True:
    
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    ser.reset_input_buffer()
    if detected == True:
        ser.write(bytes(str(value), 'utf-8'))
        ser.write(b"\n")
    else:
        value = 'lost'
        ser.write(bytes(str(value), 'utf-8'))
        ser.write(b"\n")
    
    line = ser.readline().decode('utf-8').rstrip()
    print(line)
    

    

    


    frame = get_frame(camera)
    frame = cv2.resize(frame, (img_width, img_height))
     


    imgLeft_col = frame [0:img_height,0:int(img_width/2)] #Y+H and X+W
    t1 = datetime.now()
    pair_img = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    
    imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
    imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
    rectified_pair = calibration.rectify((imgLeft, imgRight))







    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='efficientdet_lite0.tflite')
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=1280)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    args = parser.parse_args()

    detection_result = run(imgLeft_col, args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
        int(args.numThreads), bool(args.enableEdgeTPU))



    ######

    disparity = stereo_depth_map(rectified_pair, detection_result)
    # show the frame
    # cv2.imshow("left", imgLeft)
    # cv2.imshow("right", imgRight)    
    t2 = datetime.now()
    # print ("DM build time: " + str(t2-t1))


    ######



