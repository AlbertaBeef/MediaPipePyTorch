import numpy as np
import torch
import cv2
import sys
import os

from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
#from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] device = ",device)
torch.set_grad_enabled(False)

back_detector = True

palm_detector = BlazePalm().to(device)
palm_detector.load_weights("blazepalm.pth")
palm_detector.load_anchors("anchors_palm.npy")
palm_detector.min_score_thresh = .75

#hand_regressor = BlazeHandLandmark().to(device)
#hand_regressor.load_weights("blazehand_landmark.pth")

def capture_filenames(directory_name):
    images = []
    files = ['{}/{}'.format(directory_name,i) for i in sorted(os.listdir(directory_name)) ]
    #or i,file in enumerate(files):
    #   files += file
    return files

image_location = "dataset"
image_filenames = capture_filenames(image_location)
print(image_filenames)
 
image_cnt = len(image_filenames)
print("[INFO] ",image_cnt," images found in ",image_location)

for image_id in range(image_cnt):

    frame = cv2.imread(image_filenames[image_id])

    img1, img2, scale, pad = resize_pad(frame)

    normalized_palm_detections = palm_detector.predict_on_image(img1)

    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)


    #xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())
    #img, affine2, box2 = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
    #flags2, handed2, normalized_landmarks2 = hand_regressor(img.to(device))
    #landmarks2 = hand_regressor.denormalize_landmarks(normalized_landmarks2.cpu(), affine2)
    

    #for i in range(len(flags2)):
    #    landmark, flag = landmarks2[i], flags2[i]
    #    if flag>.5:
    #        draw_landmarks(frame, landmark[:,:2], HAND_CONNECTIONS, size=2)

    #draw_roi(frame, box2)
    draw_detections(frame, palm_detections)

    #cv2.imshow(WINDOW, frame[:,:,::-1])
    filename = "output"+str(image_id)+".png"
    cv2.imwrite(filename, frame)

cv2.destroyAllWindows()
