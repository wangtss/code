# Spatial Temporal Net

This is a no-reference video quality assessment project that recently I've been working on. It's part of my graduation project.

**Be Aware** that this model is designed and implemented all by myself, it could be useful but it may contain many flaws.

This file contains a basic description of the project, its not very neat but I think it does the job.

## Description

This project is based on a **Deep 3D-CNN** named STNet. 

At **training stage**, to avoid overfitting, I split the videos in video quality databases into 64x64x64 video blocks with no overlapping. You can find the generation detail of the block and the corresponding soft label in `utils.buildVideoBlock`.

The STNet takes a video block as input and outputs its quality score, its architecture is as below.

![cnn_arc.PNG](https://i.loli.net/2018/12/27/5c24dbc52d2ae.png)

At **testing stage**, the video should also be split into blocks, and the video blocks will be entered into the network to generate block scores. And the overall video score is the mean of these values.



## Usage

God this is annoying, I'll pick another day to finish this document.