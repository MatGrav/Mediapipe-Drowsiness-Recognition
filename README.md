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

## Distraction recognition

In order to detect whether the driver is distracted or not, we first compute pitch and yaw, considering the combination of Eyes and Head gaze positions.
If the difference between the axes is at least of 30°, we have to print an alarm.

At the end of the code, there is the code to print the eye gazing.
Pitch, roll and yaw have been computed yet using 3D representations. For pitch and yaw of both eyes, we can just add a flag (look at the code) to use the Enhanced algorithm, instead of default one, for 2d vectors. Anyway, using 3d computation (so without the flag), the algorithm works better.

To verify the condition and to print the alarm, we have to verify that abs(roll+pitch+yaw)>30 or abs(pitch_left_eye + yaw_left_eye + pitch_right_eye + yaw_right_eye)>30.
