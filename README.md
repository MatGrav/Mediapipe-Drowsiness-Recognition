# Mediapipe-Drowsiness-Recognition
Real-time drowsiness recognition based on mediapipe

## Introduction

This project shall provide a solution to the Assignment 1 of the course 'Technologies for Autonomous Vehicles' year 2023/24 at Politecnico di Torino.

The functionalities requested are the following:
- Drowsiness recognition through EAR and PERCLOS computation
- Distraction recognition through computation of eyes and head gaze position
  
The code is based on the one provided by professor Jacopo Sini and requires opencv-python and mediapipe.

## Drowsiness recognition

In order to detect whether the person is drowsy or not, we first compute, for each captured frame, EAR values for both eyes.
For further informations the referenced paper is the [following](https://ieeexplore.ieee.org/document/10039811)

For each frame, the algorithm takes the position of 6 relevant points for each eye
```python
if idx==362:
  P1_left = (int(lm.x * img_w), int(lm.y * img_h))
if idx==385:
  ...
```

The EAR values for both eyes are then computed accorting to the formula.
Two threshold values have been chosen, with trial and error, in order to normalize the EAR values in the range [0;1], therefore obtaining a percentage of eye opening.
```python
# Reporting only left eye 
EAR_left  = (abs(P2_left[Y]-P6_left[Y]) + abs(P3_left[Y]-P5_left[Y]))/(2*abs(P1_left[X]-P4_left[X])) 

OPEN_val = 0.32
CLOSED_val = 0.02

Left_open = (EAR_left-CLOSED_val)/(OPEN_val-CLOSED_val)
```

It is then possible to compute whether the driver is drowsy or not accordingly to the assignment, which was interpreted as follows:  
_If the EAR(normalized) is below a threshold for more than 80% of the time in the last 10 seconds, a warning message shall be printed as it indicates drowsiness._
```python
THRESHOLD = 0.65
while sum(elapsed_time) > 10:
  normalized_EAR.popleft()
  elapsed_time.popleft()

normalized_EAR.append(min(Left_open,Right_open))
elapsed_time.append(totalTime)

indices = [index for index, value in enumerate(normalized_EAR) if value < THRESHOLD]
selected_elements = [elapsed_time[index] for index in indices]
        
MAX_INTERVAL = 0.8 * 10
closed_time = sum(selected_elements)
if closed_time >= MAX_INTERVAL:
  cv2.putText(image, "DROWSY", ...
```
Two deques have been used in order to consider the last 10 seconds as a circular buffer: one of them contains the minimum normalized EAR (this implies that the driver receives a warning even if the other eye may be completely open, it is a somewhat conservative approach), the other one the time interval between the last frame and the currently processed one.

The time with EAR deemed "too low" is computed and a message is shown when necessary. The message shall not be printed before reaching 10 seconds of statistics or after the driver has intervened, in this case the message disappears after enough time (for 80% and 10 seconds, this means that 2 seconds of open eyes shall be enough)

Note that this interpretation of the assignment substitutes PERCLOS computation, **and the code could be extended in order to check drowsiness with true PERCLOS and/or change modes**


## Distraction recognition

In order to detect whether the driver is distracted or not, we first compute pitch and yaw, considering the combination of Eyes and Head gaze positions.
If the difference between the axes is at least of 30°, we have to print an alarm.

At the end of the code, there is the code to print the eye gazing.
Pitch, roll and yaw have been computed yet using 3D representations. *For pitch and yaw of both eyes, we can just add a flag (look at the code) to use the Enhanced algorithm, instead of default one, for 2d vectors. Anyway, using 3d computation (so without the flag), the algorithm works better.
However this seems to hold true in the case of HD camera or better, as the increased number of pixels makes the algorithm work properly. Tests with 640x480p have not been successful and therefore we expect to implement a 2D version, possibly leaving the 3D version to be used when HD_MODE is True*

A calibration "step" has been introduced: since our webcam may not be at the same level as our eyes when running the application, 
an high degree of pitch can be detected even when we are actually trying to look straight ahead, as we would when driving, leading to erroneous distraction detection.
```python
pitch_calibration = np.zeros(30,dtype=float)
while calib_index < len(pitch_calibration):
    pitch_calibration[calib_index] = pitch
    calib_index += 1
    pitch_constant = np.mean(pitch_calibration)

pitch = pitch - pitch_constant #for each frame
```
The adopted solution subtracts from the pitch a value calculated by averaging the pitch of the first 30 processed frames: this proves to be a simple yet effective approach which requires the user to look correctly at startup.

We would expect calibration to be implemented also in a real world scenario, with the difference that the position of the camera with respect to the driver shall be known for each car.
         

To verify the condition and to print the alarm, we have to verify that abs(roll+pitch+yaw)>30 or abs(pitch_left_eye + yaw_left_eye + pitch_right_eye + yaw_right_eye)>30.
