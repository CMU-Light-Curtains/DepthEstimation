# Exploiting & Refining Depth Distributions with Triangulation Light Curtains

## About

Active sensing through the use of Adaptive Depth Sensors is a nascent field, with potential in areas such as Advanced driver-assistance systems (ADAS). They do however require dynamically driving a laser / light-source to a specific location to capture information, with one such class of sensor being the Triangulation Light Curtains (LC). In this work, we introduce a novel approach that exploits prior depth distributions from RGB cameras to drive a Light Curtain's laser line to regions of uncertainty to get new measurements. These measurements are utilized such that depth uncertainty is reduced and errors get corrected recursively. We show real-world experiments that validate our approach in outdoor and driving settings, and demonstrate qualitative and quantitative improvements in depth RMSE when RGB cameras are used in tandem with a Light Curtain. This code has the official pytorch implementation for this paper:

Monocular RGB Depth Estimation
![Monocular](https://github.com/soulslicer/adap-fusion/blob/master/pics/before.gif?raw=true)

Monocular RGB Depth Estimation + Light Curtion Fusion
![Monocular](https://github.com/soulslicer/adap-fusion/blob/master/pics/after.gif?raw=true)

## Links

Youtube Video

[![Youtube Video](https://img.youtube.com/vi/kIjn3U8luV0/0.jpg)](https://www.youtube.com/watch?v=kIjn3U8luV0)

Paper

https://github.com/soulslicer/adap-fusion/blob/master/pics/paper.pdf

## Explanation

![Intro](https://github.com/soulslicer/adap-fusion/blob/master/pics/lc.png?raw=true)

Triangulation Light Curtains were introduced in 2019 and developed by the Illumination and Imaging (ILIM) Lab at CMU. This sensor (comprising of a rolling shutter NIR camera and a galvomirror with a laser) uses a unique imaging strategy that relies on the user providing the depth to be sampled, with the sensor returning the return intensity at said location. The user provides a 2D top-down profile, which then generates a curtain in 3D. Any surface that is within the thickness of this curtain, returns a signal in the NIR image which we can capture. 

![Intro](https://raw.githubusercontent.com/soulslicer/adap-fusion/master/pics/intro.png)

Neural RGBD introduced the world to Depth Probability Fields (DPV), where instead of predicting depth per pixel, we predict a distribution per pixel. To help us visualize the uncertainty, we collapsed the distribution along the surface of the road so that you can visualize the Uncertainty Field (UF). You can read more details in [here](https://github.com/soulslicer/probabilistic-depth/blob/main/pics/explanation.pdf). One can see that Monocular RGB depth estimation has significant uncertainty. 

![Intro](https://github.com/soulslicer/adap-fusion/blob/master/pics/together.png?raw=true)

We show how errors in Monocular Depth Estimation are corrected when used in tandem with an Adaptive Sensor such as a Triangulating Light Curtain (Yellow Points and Red lines are Ground Truth). (b) We predict a per-pixel Depth Probability Volume from Monocular RGB and we observe large per-pixel uncertainties (σ=3m) as seen in the Bird's Eye View /  Top-Down Uncertainty Field slice. (c) We actively drive the Light Curtain sensor's Laser to exploit and sense multiple regions along a curve that maximize information gained. (d) We feed these measurements back recursively to get a refined depth estimate, along with a reduction in uncertainty.

## Installation

```
- Download Kitti
    # Download from http://www.cvlibs.net/datasets/kitti/raw_data.php
    # Use the raw_kitti_downloader script
    # Link kitti to a folder in this project folder
    ln -s {absolute_kitti_path} kitti
- Install Python Deps
    torch, matplotlib, numpy, cv2, pykitti, tensorboardX
- Compile external deps
    cd external
    # Select the correct sh file depending on your python version
    # Modify it so that you select the correct libEGL version
    # Eg. /usr/lib/x86_64-linux-gnu/libEGL.so.1.1.0
    sudo sh compile_3.X.sh
    # Ensure everything compiled correctly with no errors
- Download pre-trained models
    # https://drive.google.com/file/d/1XN5lrFobInkcJ4F6cHgAd385eX9Galfw/view?usp=sharing
    unzip output.zip
```

## Running

```
# Eval
- Moving Monocular Depth Estimation
    python3 train.py --config configs/default_mono.json --eval --viz
- Moving Monocular Depth Estimation with Feedback
    python3 train.py --config configs/default_mono_feedback.json --eval --viz
- Stereo Depth Estimation
    python3 train.py --config configs/default_stereo.json --eval --viz
- Stereo Depth Estimation with Feedback
    python3 train.py --config configs/default_stereo_feedback.json --eval --viz
- Moving Monocular Lidar Upsampling
    python3 train.py --config configs/default_mono_upsample.json --eval --viz

# Training
    Training only works with Python 3 as it uses distributed training
    To train, simply remove the eval and viz flags. Use the `batch_size` flag to change batch size. It automatically splits it among the GPUs available
    `pkill -f -multi` to clear memory if crashes
```

## Math?

Coming soon. Disclaimer: This is a template code that I have written to extend some research I am working on. Please be sure to cite this if you use this code

## References

```
Uses code from https://github.com/lliuz/ARFlow (ARFlow) and https://github.com/NVlabs/neuralrgbd (NeuralRGBD)
```