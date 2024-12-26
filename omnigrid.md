# Omnigrid: Multiview Fisheye Occupancy Grid Prediction

This blog-form illustrates what is extensively covered in the [corresponding report](https://drive.google.com/file/d/1s42BNxuo1BWgJO1owSTYkNOkcjr6JB_F/view?usp=drive_link).

> Abstract: This project aims to leverage large field-of-view (FOV) cameras to explore point cloud prediction approaches, and essentially develop a 3D perception stack for a multi-camera large FOV setup.

### Table of Contents
1. [Introduction](#introduction)

<a name="introduction"></a>

## Introduction

This project aims to leverage large field-of-view (FOV) cameras to explore point cloud prediction approaches, and
essentially develop a 3D perception stack for a multi-camera large FOV setup. The larger FOV along with multiple
cameras enables the robot to detect blind spots, which is extremely crucial for safety-critical applications. By hav-
ing multiple viewpoints, each with FOV ≈ 195◦, the same object/feature will be visible across multiple cameras.
These redundancies can help overcome problems like occlusions and variable lighting conditions, thus enhancing the
robustness of the perception stack.

Through this work, we attempt to break down the point cloud prediction task into 2 subtasks - fisheye depth prediction,
and multi-fisheye depth-to-point cloud prediction. Additionally, we explore this problem in outdoor settings. The code
can be found [here](https://github.com/WarrG3X/omnigrid).

![drone setup](images/drone_setup.png)