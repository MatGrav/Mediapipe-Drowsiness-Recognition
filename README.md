# Mediapipe-Drowsiness-Recognition
Real-time drowsiness recognition based on mediapipe

Authors: Matteo Gravagnone, Danilo Guglielmi

## Introduction

This project shall provide a solution to the Assignment 1 of the course 'Technologies for Autonomous Vehicles' year 2023/24 at Politecnico di Torino.

The functionalities requested are the following:
- Drowsiness recognition through EAR and PERCLOS computation
- Distraction recognition through computation of eyes and head gaze position
  
The code is based on the one provided by professor Jacopo Sini and requires opencv-python and mediapipe.

## Drowsiness recognition

In order to detect whether the person is drowsy or not, we first compute, for each captured frame, EAR values for both eyes.
For further informations the referenced paper is the [following](https://ieeexplore.ieee.org/document/10039811).

For each frame, the algorithm takes the position of 6 relevant points for each eye
```python
if idx==362:
  P1_left = (int(lm.x * img_w), int(lm.y * img_h))
if idx==385:
  ...
```

The EAR values for both eyes are then computed according to the formula.
Two threshold values have been chosen, with trial and error, in order to normalize the EAR values in the range [0;1], therefore obtaining a percentage of eye opening.
```python
OPEN_VAL = 0.32
CLOSED_VAL = 0.02
# ...

# Reporting only left eye 
EAR_left  = (abs(P2_left[Y]-P6_left[Y]) + abs(P3_left[Y]-P5_left[Y]))/
            (2*abs(P1_left[X]-P4_left[X])) 

Left_open = (EAR_left-CLOSED_VAL)/(OPEN_VAL-CLOSED_VAL)
```

It is then possible to compute whether the driver is drowsy or not accordingly to the assignment, which was interpreted as follows:  
_If the EAR (normalized) is below a threshold for more than 80% of the time in the last 10 seconds, a warning message shall be printed as it indicates drowsiness._

Two deques have been used in order to consider the last 10 seconds as a circular buffer: `normalized_EAR` contains the minimum normalized EAR _(this implies that the driver receives a warning even if the other eye may be completely open, it is a somewhat conservative approach)_, `elapsed_time` the time interval between the last frame and the currently processed one.

```python
NORM_EAR_THRESHOLD = 0.65

while sum(elapsed_time) > 10:
    normalized_EAR.popleft()
    elapsed_time.popleft()

normalized_EAR.append(min(Left_open,Right_open))
elapsed_time.append(totalTime)

indices = [index for index, value in enumerate(normalized_EAR)
          if value < NORM_EAR_THRESHOLD]
selected_elements = [elapsed_time[index] for index in indices]
```

The time with EAR deemed "too low" is computed and a message is shown if conditions are met. The message shall not be printed before reaching 10 seconds of statistics or after the driver has intervened. In this case the message disappears after enough time (for 80% and 10 seconds, this means that 2 seconds of open eyes shall be enough).
```python
MAX_INTERVAL = 0.8 * 10

closed_time = sum(selected_elements)
if closed_time >= MAX_INTERVAL:
    cv2.putText(image, "DROWSY",...
```

Note that this interpretation of the assignment substitutes PERCLOS computation, **and the code could be extended in order to check drowsiness with true PERCLOS and/or change modes**


## Distraction recognition

To detect whether the driver is distracted or not, we check both head and eye gaze. Firstly, pitch, yaw, and roll related to the head are computed, and a warning message is shown when one of them differs more than 30° from the rest position. Additionally, the same message is shown whenever the center of the iris is considered too far from the center of the eye.


### 3D Head Gazing

Pitch, roll and yaw have been computed using 3D representations, with the definition of a camera matrix (accordingly to the pinhole model) and computation of first rotational vectors, then rotational matrix and angles.

A calibration "step" has been introduced: since our webcam may not be at the same level as our eyes when running the application, 
an high degree of pitch can be detected even when we are actually trying to look straight ahead, as we would when driving, leading to erroneous distraction detection.
A similar reasoning can be done regarding the yaw, as our head may not easily be exactly in front of the camera.
```python
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
```
The adopted solution subtracts from pitch and yaw values calculated by averaging them either in the first 30 processed frames or after pressing the R key: this proves to be a simple yet effective approach which requires the user to look correctly at startup or calibration time.

We would expect calibration to be implemented also in a real world scenario, with the difference that the position of the camera with respect to the driver shall be known for each car.


### 2D Eye gazing

Even though the 3D approach is theoretically valid also for eye angles, the results are unsatisfactory when using low resolution cameras, due to the low number of pixels for the eyes. 

In order to obtain a code with a distraction recognitizion algorithm compatible with most laptop cameras, a 2D approach has been proposed.

The algorithm works with the 2D image by computing, for each eye, the relative positions between the iris center and eye center both in horizontal and vertical components, which are then normalized by dividing by half of the height or width.
During our tests, differences between these distances have been noticed between left and right eye, so we have chosen to compare the max value in each direction to threshold values — `X_THRESHOLD` or `Y_THRESHOLD` — which have been empirically chosen.


```python
#Reporting for simplicity code regarding right eye only 
eye_gaze_2d_right = ((point_REIC[X] - r_eye_center[X])/(r_eye_width/2),
(point_REIC[Y] - r_eye_center[Y])/(r_eye_height/2))
eye_gaze_2d_left = ((point_LEIC[X] - l_eye_center[X])/(l_eye_width/2),
(point_LEIC[Y] - l_eye_center[Y])/(l_eye_height/2))


if max(abs(eye_gaze_2d_right[X]),abs(eye_gaze_2d_left[X]))>X_THRESHOLD
or max(abs(eye_gaze_2d_right[Y]),abs(eye_gaze_2d_left[Y]))>Y_THRESHOLD :
            eye_distraction = True

if ... or abs(yaw)>30 or eye_distraction is True:
            cv2.putText(image, "ALARM: The driver is distracted", ... )
```
