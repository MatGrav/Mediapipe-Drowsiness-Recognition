#**************************************************************************************
#
#   Driver Monitoring Systems using AI (code sample)
#
#   File: eyes_position.m
#   Author: Jacopo Sini
#   Company: Politecnico di Torino
#   Date: 19 Mar 2024
#
#**************************************************************************************

# 1 - Import the needed libraries 
import cv2
import mediapipe as mp
import numpy as np 
import time
import statistics as st
import os

from collections import deque

# 2 - Set the desired setting
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Enables  detailed eyes points
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Get the list of available capture devices (comment out)
#index = 0
#arr = []
#while True:
#    dev = cv2.VideoCapture(index)
#    try:
#        arr.append(dev.getBackendName)
#    except:
#        break
#    dev.release()
#    index += 1
#print(arr)

# 3 - Open the video source
cap = cv2.VideoCapture(0) # Local webcam (index start from 0)

# 3.1 - Declaration of some variables 
normalized_EAR = deque()
elapsed_time = deque()
calib_index = 0
CALIBRATION_BUFFER_DIM = 30 # TO DO : Change if needed
pitch_calibration = np.zeros(CALIBRATION_BUFFER_DIM,dtype=float)
yaw_calibration = np.zeros(CALIBRATION_BUFFER_DIM,dtype=float)


# 4 - Iterate (within an infinite loop)
while cap.isOpened(): 
    
    # 4.1 - Get the new frame
    success, image = cap.read()

    start = time.time()

    # Also convert the color space from BGR to RGB
    if image is None:
        break
        #continue
    #else: #needed with some cameras/video input format
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # To improve performace
    image.flags.writeable = False
    
    # 4.2 - Run MediaPipe on the frame
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    img_h, img_w, img_c = image.shape


    FONT_SCALE = 1.5 * 1e-3  # Adjust for larger font size in all images
    font_scale = min(img_w, img_h) * FONT_SCALE
    line_scale = min(img_w, img_h) * FONT_SCALE


    point_RER = [] # Right Eye Right
    point_REB = [] # Right Eye Bottom
    point_REL = [] # Right Eye Left
    point_RET = [] # Right Eye Top

    point_LER = [] # Left Eye Right
    point_LEB = [] # Left Eye Bottom
    point_LEL = [] # Left Eye Left
    point_LET = [] # Left Eye Top

    point_REIC = [] # Right Eye Iris Center
    point_LEIC = [] # Left Eye Iris Center

    face_2d = []
    face_3d = []
    left_eye_2d = []
    left_eye_3d = []
    right_eye_2d = []
    right_eye_3d = []

    # 4.3 - Get the landmark coordinates
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):


                # Eye Gaze (Iris Tracking)
                # Left eye indices list
                #LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
                # Right eye indices list
                #RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
                #LEFT_IRIS = [473, 474, 475, 476, 477]
                #RIGHT_IRIS = [468, 469, 470, 471, 472]
                if idx == 33:
                    point_RER = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 145:
                    point_REB = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 133:
                    point_REL = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 159:
                    point_RET = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 362:
                    point_LER = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 374:
                    point_LEB = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 263:
                    point_LEL = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 386:
                    point_LET = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx == 468:
                    point_REIC = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 255, 0), thickness=-1)                    
                if idx == 469:
                    point_469 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)
                if idx == 470:
                    point_470 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)
                if idx == 471:
                    point_471 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)
                if idx == 472:
                    point_472 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)
                if idx == 473:
                    point_LEIC = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 255), thickness=-1)
                if idx == 474:
                    point_474 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)
                if idx == 475:
                    point_475 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)
                if idx == 476:
                    point_476 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)
                if idx == 477:
                    point_477 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1) 
                


                # face orientation
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    
                    # Get the 2D Coordinates
                    face_2d.append([x, y])
                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                #LEFT_IRIS = [473, 474, 475, 476, 477]
                if idx == 473 or idx == 362 or idx == 374 or idx == 263 or idx == 386: # iris points
                #if idx == 473 or idx == 474 or idx == 475 or idx == 476 or idx == 477: # eye border
                    if idx == 473:
                        left_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        left_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    left_eye_2d.append([x, y])
                    left_eye_3d.append([x, y, lm.z])

                #RIGHT_IRIS = [468, 469, 470, 471, 472]
                if idx == 468 or idx == 33 or idx == 145 or idx == 133 or idx == 159: # iris points
                # if idx == 468 or idx == 469 or idx == 470 or idx == 471 or idx == 472: # eye border
                    if idx == 468:
                        right_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        right_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    right_eye_2d.append([x, y])
                    right_eye_3d.append([x, y, lm.z]) 

                # EAR
                # Left eye
                if idx==362:
                    P1_left = (int(lm.x * img_w), int(lm.y * img_h))
                if idx==385:
                    P2_left = (int(lm.x * img_w), int(lm.y * img_h))
                if idx==387:
                    P3_left = (int(lm.x * img_w), int(lm.y * img_h))
                if idx==263:
                    P4_left = (int(lm.x * img_w), int(lm.y * img_h))
                if idx==373:
                    P5_left = (int(lm.x * img_w), int(lm.y * img_h))
                if idx==380:
                    P6_left = (int(lm.x * img_w), int(lm.y * img_h))

                # Right eye
                if idx==33:
                    P1_right = (int(lm.x * img_w), int(lm.y * img_h))
                if idx==160:
                    P2_right = (int(lm.x * img_w), int(lm.y * img_h))
                if idx==158:
                    P3_right = (int(lm.x * img_w), int(lm.y * img_h))
                if idx==133:
                    P4_right = (int(lm.x * img_w), int(lm.y * img_h))
                if idx==153:
                    P5_right = (int(lm.x * img_w), int(lm.y * img_h))
                if idx==144:
                    P6_right = (int(lm.x * img_w), int(lm.y * img_h))


            # 4.4. - Draw the positions on the frame
            l_eye_width = point_LEL[0] - point_LER[0]
            l_eye_height = point_LEB[1] - point_LET[1]
            l_eye_center = [(point_LEL[0] + point_LER[0])/2 ,(point_LEB[1] + point_LET[1])/2]
            #cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=int(horizontal_threshold * l_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius 
            cv2.circle(image, (int(point_LEIC[0]), int(point_LEIC[1])), radius=3, color=(0, 255, 0), thickness=-1) # Center of iris
            cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye
            #print("Left eye: x = " + str(np.round(point_LEIC[0],0)) + " , y = " + str(np.round(point_LEIC[1],0)))
            cv2.putText(image, "Left eye:  x = " + str(np.round(point_LEIC[0],0)) + " , y = " + str(np.round(point_LEIC[1],0)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2) 

            r_eye_width = point_REL[0] - point_RER[0]
            r_eye_height = point_REB[1] - point_RET[1]
            r_eye_center = [(point_REL[0] + point_RER[0])/2 ,(point_REB[1] + point_RET[1])/2]
            #cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=int(horizontal_threshold * r_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius 
            cv2.circle(image, (int(point_REIC[0]), int(point_REIC[1])), radius=2, color=(0, 255, 0), thickness=-1) # Center of iris
            cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=2, color=(0, 0, 255), thickness=-1) # Center of eye
            #print("right eye: x = " + str(np.round(point_REIC[0],0)) + " , y = " + str(np.round(point_REIC[1],0)))
            cv2.putText(image, "Right eye: x = " + str(np.round(point_REIC[0],0)) + " , y = " + str(np.round(point_REIC[1],0)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2) 

            # speed reduction (comment out for full speed)
            time.sleep(1/30) # [s]

        X = 0
        Y = 1
        
        EAR_left  = (abs(P2_left[Y]-P6_left[Y]) + abs(P3_left[Y]-P5_left[Y]))/(2*abs(P1_left[X]-P4_left[X])) 
        EAR_right = (abs(P2_right[Y]-P6_right[Y]) + abs(P3_right[Y]-P5_right[Y]))/(2*abs(P1_right[X]-P4_right[X]))
        
        OPEN_val = 0.32
        CLOSED_val = 0.02

        Left_open = (EAR_left-CLOSED_val)/(OPEN_val-CLOSED_val)
        Right_open = (EAR_right-CLOSED_val)/(OPEN_val-CLOSED_val)

        cv2.putText(image, "EAR Left eye: " + str(np.round(Left_open*100,2)), (25, int(img_h/2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2) 
        cv2.putText(image, "EAR Right eye: " + str(np.round(Right_open*100,2)), (25, int(img_h/2)+40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2) 

        
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        left_eye_2d = np.array(left_eye_2d, dtype=np.float64)
        left_eye_3d = np.array(left_eye_3d, dtype=np.float64)
        right_eye_2d = np.array(right_eye_2d, dtype=np.float64)
        right_eye_3d = np.array(right_eye_3d, dtype=np.float64)

        # The camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
        [0, focal_length, img_w / 2],
        [0, 0, 1]])

        # The distorsion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        
        # Computation of rotation matrices
        # This still contains the variables used in 3D approach with eyes, not used for the actual gazing


        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        success_left_eye, rot_vec_left_eye, trans_vec_left_eye = cv2.solvePnP(left_eye_3d, left_eye_2d, cam_matrix, dist_matrix)
        success_right_eye, rot_vec_right_eye, trans_vec_right_eye = cv2.solvePnP(right_eye_3d, right_eye_2d, cam_matrix, dist_matrix)


        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)
        rmat_left_eye, jac_left_eye = cv2.Rodrigues(rot_vec_left_eye)
        rmat_right_eye, jac_right_eye = cv2.Rodrigues(rot_vec_right_eye)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        angles_left_eye, mtxR_left_eye, mtxQ_left_eye, Qx_left_eye, Qy_left_eye, Qz_left_eye = cv2.RQDecomp3x3(rmat_left_eye)
        angles_right_eye, mtxR_right_eye, mtxQ_right_eye, Qx_right_eye, Qy_right_eye, Qz_right_eye = cv2.RQDecomp3x3(rmat_right_eye)

        pitch = angles[0] * 1800
        yaw = -angles[1] * 1800
        roll = 180 + (np.arctan2(point_RER[1] - point_LEL[1], point_RER[0] - point_LEL[0]) * 180 / np.pi)
        if roll > 180:
            roll = roll - 360
        
        
        pitch_left_eye = angles_left_eye[0] * 1800
        yaw_left_eye = angles_left_eye[1] * 1800
        pitch_right_eye = angles_right_eye[0] * 1800
        yaw_right_eye = angles_right_eye[1] * 1800
        
        
        
        # Compute the 2D eyes gaze
        # Components are in the [-1;1] range, looking  RIGHT->left  or  DOWN (in theory) -> UP
        eye_gaze_2d_right = ((point_REIC[X] - r_eye_center[X])/(r_eye_width/2), (point_REIC[Y] - r_eye_center[Y])/(r_eye_height/2))
        eye_gaze_2d_left = ((point_LEIC[X] - l_eye_center[X])/(l_eye_width/2), (point_LEIC[Y] - l_eye_center[Y])/(l_eye_height/2))

        # DEBUG
        #cv2.putText(image, "REIC_Y: " + str(np.round(point_REIC[Y], 3)), (315, 140), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        #cv2.putText(image, "RIGHT EYE CENTER y: " + str(np.round(r_eye_center[Y], 3)), (315, 160), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        #cv2.putText(image, "R HEIGHT: " + str(np.round(r_eye_height, 3)), (315, 180), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        
        
        # Angles approach
        '''
        # Angles source: https://biology.stackexchange.com/questions/72511/what-are-the-max-angles-of-human-eyeball-rotation
        # Written in degrees, these are meant per side
        HORIZONTAL_MAX_ROTATION_DEGREES = 35 
        VERTICAL_MAX_ROTATION_DEGREES = 25

        angles_left_eye_2d = (eye_gaze_2d_left[X] * HORIZONTAL_MAX_ROTATION_DEGREES, eye_gaze_2d_left[Y] * VERTICAL_MAX_ROTATION_DEGREES)
        #print(angles_left_eye_2d)

        angles_right_eye_2d = (eye_gaze_2d_right[X] * HORIZONTAL_MAX_ROTATION_DEGREES, eye_gaze_2d_right[Y] * VERTICAL_MAX_ROTATION_DEGREES)
        #print(angles_right_eye_2d)

        #cv2.putText(image, "Angles left eye: " + str(np.round(angles_left_eye_2d, 3)), (15, 140), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        #cv2.putText(image, "Angles right eye: " + str(np.round(angles_right_eye_2d, 3)), (15, 160), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        '''

        # Qual è la soglia migliore?
        # Va stampato a intermittenza o secondo un intervallo continuo?
        X_THRESHOLD = 0.25
        Y_THRESHOLD = 0.25
        
        if max(abs(eye_gaze_2d_right[X]),abs(eye_gaze_2d_left[X]))>X_THRESHOLD or max(abs(eye_gaze_2d_right[Y]),abs(eye_gaze_2d_left[Y]))>Y_THRESHOLD :
            cv2.putText(image, "DOVE GUARDI?!", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        
        cv2.putText(image, "right_X: " + str(np.round(eye_gaze_2d_right[X], 3)), (15, 320), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        cv2.putText(image, "right_y: " + str(np.round(eye_gaze_2d_right[Y], 3)), (15, 340), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        cv2.putText(image, "left_X: " + str(np.round(eye_gaze_2d_left[X], 3)), (305, 320), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        cv2.putText(image, "left_y: " + str(np.round(eye_gaze_2d_left[Y], 3)), (305, 340), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)


        # Calibration array for pitch computation, as our webcam may not be at the same level of our head
        # => Our head's pitch is detected even when we are actually trying to look "straight ahead" 
        # It is calibrated based on an average of the pitch in the first 30 captured frames
        # In a real world application, calibration is static as we assume the camera stays fixed in place in the car
        
        key = cv2.waitKey(1)
        while calib_index < len(pitch_calibration) or key == 114 or key == 82:
            
            if key==114 or key==82: # Pressing r or R
                pitch_calibration = np.zeros(CALIBRATION_BUFFER_DIM,dtype=float)
                yaw_calibration = np.zeros(CALIBRATION_BUFFER_DIM,dtype=float)
                calib_index = 0
                pitch_constant = 0
                yaw_constant = 0
                key = 0
            
            pitch_calibration[calib_index] = pitch
            yaw_calibration[calib_index] = yaw
            calib_index += 1
            pitch_constant = np.mean(pitch_calibration)
            yaw_constant = np.mean(yaw_calibration)
            

        pitch = pitch - pitch_constant
        yaw = yaw - yaw_constant

        yaw = -yaw # ?
        alfa = 0
        # Distraction detection
        # alternative conditions:
        if abs(roll)>30 or abs(pitch)>30 or abs(yaw)>30:
        # if abs(yaw + alfa)>30:
            cv2.putText(image, "ALARM: The driver is distracted", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

        
        # DEBUG
        cv2.putText(image, "yaw: " + str(np.round(yaw, 3)), (15, 140), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        cv2.putText(image, "alfa: " + str(np.round(alfa, 3)), (15, 160), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        
        # DEBUG
        cv2.putText(image, "roll: " + str(np.round(roll, 4)), (15, 220), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        cv2.putText(image, "pitch: " + str(np.round(pitch, 4)), (15, 240), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        cv2.putText(image, "yaw: " + str(np.round(yaw, 4)), (15, 260), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        #cv2.putText(image, "pitch_left_eye: " + str(pitch_left_eye), (15, 280), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        #cv2.putText(image, "pitch_right_eye: " + str(pitch_right_eye), (15, 300), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        #cv2.putText(image, "yaw_left_eye: " + str(yaw_left_eye), (15, 320), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        #cv2.putText(image, "yaw_right_eye: " + str(yaw_right_eye), (15, 340), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        

        # Display directions
        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        
        p2 = (int(nose_2d[0] - yaw * line_scale), int(nose_2d[1] - pitch * line_scale))
        cv2.line(image, p1, p2, (255, 0, 0), 3)
        ''' ####### CHECK
        # right eye direction
        # eye_right_3d_projection, jacobian_right_eye = cv2.projectPoints(right_eye_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        p3 = (int(right_pupil_2d[0]), int(right_pupil_2d[1]))
        p4 = (int(right_pupil_2d[0] + yaw_right_eye * line_scale), int(right_pupil_2d[1] - pitch_right_eye * line_scale))
        cv2.line(image, p3, p4, (255, 0, 0), 3)

        # left eye direction
        # eye_left_3d_projection, jacobian_left_eye = cv2.projectPoints(left_eye_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        p5 = (int(left_pupil_2d[0]), int(left_pupil_2d[1]))
        p6 = (int(left_pupil_2d[0] + yaw_left_eye * line_scale), int(left_pupil_2d[1] - pitch_left_eye * line_scale))
        cv2.line(image, p5, p6, (255, 0, 0), 3)
        '''       

        end = time.time()
        totalTime = end-start

        if totalTime>0:
            fps = 1 / totalTime
        else:
            fps=0
        
        #Drowsiness detection
        NORM_EAR_THRESHOLD = 0.65

        while sum(elapsed_time) > 10:
            normalized_EAR.popleft()
            elapsed_time.popleft()
        
        normalized_EAR.append(min(Left_open,Right_open))
        elapsed_time.append(totalTime)

        indices = [index for index, value in enumerate(normalized_EAR) if value < NORM_EAR_THRESHOLD]
        selected_elements = [elapsed_time[index] for index in indices]
        
        MAX_INTERVAL = 0.8 * 10

        closed_time = sum(selected_elements)
        if closed_time >= MAX_INTERVAL:
            cv2.putText(image, "DROWSY", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2) 
        
        #print("FPS:", fps)

        cv2.putText(image, f'FPS : {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # 4.5 - Show the frame to the user
        cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI code', image)       
                    
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 5 - Close properly soruce and eventual log file
cap.release()
#log_file.close()
    
# [EOF]
