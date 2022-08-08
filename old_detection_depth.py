import time
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from datetime import datetime
import arducam_mipicamera as arducam
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
object_box = None



classNames = []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)


# net = cv2.dnn.readNetFromDarknet("custom-yolov4-tiny-detector.cfg","custom-yolov4-tiny-detector_best.weights")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



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




# cv2.imshow("crop",crop)
#cv2.waitKey(1)



#### removed to no show image  ###
# # Initialize interface windows 
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50,100)
cv2.namedWindow("left")
cv2.moveWindow("left", 450,100)
cv2.namedWindow("right")
cv2.moveWindow("right", 850,100)
##################################


disparity = np.zeros((img_width, img_height), np.uint8)
sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    global object_box
    objectInfo =[]
    if len(classIds) != 0:
        
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    object_box = box
                   
                    

    return img,objectInfo,object_box

def stereo_depth_map(rectified_pair, box, box_bool):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    local_max = disparity.max()
    local_min = disparity.min()
    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    cv2.rectangle(disparity_color,box,color=(0,255,0),thickness=2)
    
    # print(box)
    if box_bool == True:
        # print(type(box))
        try:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # print(x1)
            # print(y1)
            box_centre = (x1 + x2)/2, (y1 + y2)/2
            print(box_centre)
            if box_centre < 80:
                print('move left')
            elif box_centre > 220:
                print('move_right')
            else:
                print('move forward')

            rect = disparity_color[y1:y1+y2, x1:x1+x2]
            #make change to not include colour blue
            # rect = disparity_color[0:3, 0:5]
            depth_value = rect.mean()
            
            print('box mean:', depth_value)
            if depth_value > 80:
                print('move forward')
            else:
                print('move backward')
        except TypeError:
            print("No object")
            box_bool == False
            pass


    # box_list = box.values.tolist()
    # print(type(box_list))
    

    #### removed to no show image  ####
    cv2.imshow("Image", disparity_color)
    ###################################


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

# capture frames from the camera
# for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
box_bool = False
while True:
    frame = get_frame(camera)
    frame = cv2.resize(frame, (img_width, img_height))

    t1 = datetime.now()
    pair_img = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    pair_img2 = cv2.cvtColor (frame, cv2.COLOR_BGR2RGB)
    imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
    imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
    rectified_pair = calibration.rectify((imgLeft, imgRight))
    
   
    imgLeft2 = pair_img2 [0:img_height,0:int(img_width/2)] #Y+H and X+W
    # print(img_width/2, img_height)
    # crop = imgLeft2[0:320, 0:320]
    crop = imgLeft2
    # print('this is crop')
    # print(crop.shape)
    # cv2.imshow("crop",crop)
 
   
    result, objectInfo, box = getObjects(crop,0.2,0.2, objects=['cup'])
    #print(objectInfo)


    #### removed to no show image  ###
    cv2.imshow("crop",crop)
    cv2.waitKey(1)
    ##################################

    disparity = stereo_depth_map(rectified_pair, box, box_bool)
    box_bool = True

    # show the frame
    
    #### removed to no show image  ###
    cv2.imshow("left", imgLeft)
    cv2.imshow("right", imgRight)
    # ##################################    

    t2 = datetime.now()
    # print ("DM build time: " + str(t2-t1))

