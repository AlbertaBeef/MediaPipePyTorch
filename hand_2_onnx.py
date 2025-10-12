import numpy as np
import torch
import cv2
import sys
import os

from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] device = ",device)
torch.set_grad_enabled(False)

back_detector = True

palm_detector = BlazePalm().to(device)
palm_detector.load_weights("blazepalm.pth")
palm_detector.load_anchors("anchors_palm.npy")
palm_detector.min_score_thresh = .75

hand_regressor = BlazeHandLandmark().to(device)
hand_regressor.load_weights("blazehand_landmark.pth")

def capture_filenames(directory_name):
    images = []
    files = ['{}/{}'.format(directory_name,i) for i in sorted(os.listdir(directory_name)) ]
    #or i,file in enumerate(files):
    #   files += file
    return files

WINDOW='test'
cv2.namedWindow(WINDOW)

image_location = "dataset"
image_filenames = capture_filenames(image_location)
print(image_filenames)
 
image_cnt = len(image_filenames)
print("[INFO] ",image_cnt," images found in ",image_location)

for image_id in range(image_cnt):

    frame = cv2.imread(image_filenames[image_id])
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    img1, img2, scale, pad = resize_pad(frame)

    normalized_palm_detections = palm_detector.predict_on_image(img1)

    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

    xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())
    img, affine2, box2 = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags2, handed2, normalized_landmarks2 = hand_regressor(img.to(device))
    landmarks2 = hand_regressor.denormalize_landmarks(normalized_landmarks2.cpu(), affine2)

    for i in range(len(flags2)):
        landmark, flag = landmarks2[i], flags2[i]
        #if flag>.5:
        #    draw_landmarks(frame, landmark[:,:2], HAND_CONNECTIONS, size=2)
        draw_landmarks(frame, landmark[:,:2], HAND_CONNECTIONS, size=2)

    draw_roi(frame, box2)
    draw_detections(frame, palm_detections)

    cv2.imshow(WINDOW, frame[:,:,::-1])
    cv2.waitKey(0)
    #filename = "output"+str(image_id)+".png"
    #cv2.imwrite(filename, frame)
    
    torch.onnx.export(
        hand_regressor,              # model to export
        img,                         # model input (or a tuple for multiple inputs)
        "BlazeHandLandmark.onnx",    # where to save the model (can be a file or file-like object)
        export_params=True,          # store the trained parameter weights inside the model file
        opset_version=12,            # the ONNX version to export the model to
        do_constant_folding=True,    # whether to execute constant folding for optimization
        # INFO - Model inputs: ['input_1']
        # INFO - Model outputs: ['ld_21_3d', 'output_handflag', 'output_handedness']
        input_names = ['x'],         # the model's input names
        output_names = ['hand_flag', 'handed', 'landmarks'], # the model's output names
        )

    break
    
cv2.destroyAllWindows()
