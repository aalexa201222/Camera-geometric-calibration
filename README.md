# Camera Geometric Calibration
![alt text](https://github.com/aalexa201222/Camera-geometric-calibration/blob/0e4242d545020719d763b59ef6e785b60f815b09/calibration.jpg?raw=true)
## Overview
This project is focused on implementing geometric camera calibration using Python and OpenCV. The calibration process involves both an offline and online phase. In the offline phase, the camera is calibrated using images of a chessboard taken from various angles and distances. This calibration is critical for achieving precise 3D reconstructions in computer vision applications. The online phase demonstrates the application of the calibrated parameters by projecting 3D objects onto new camera images.

## Key Features
- **Automated and Manual Corner Detection**: While the system primarily relies on OpenCV's automated corner detection to identify chessboard corners in images, it also features a robust manual interface. This interface allows users to manually specify corners when automatic detection is inadequate, ensuring the system remains functional across a range of image qualities.
- **Multiple Calibration Runs**: The system supports three distinct calibration runs with different sets of images to evaluate the calibration's robustness and precision under various conditions. This feature allows users to understand how variations in input data quality affect the calibration outcome.
- **3D Projection in Real-Time**: Leveraging the calibrated camera parameters, the system can visualize 3D objects overlaid on real-time video feeds. This feature is crucial for augmented reality applications and provides a vivid demonstration of the calibration's accuracy.

## Requirements
- **Python**: Version 3.8 or higher.
- **OpenCV**: Needed for all image processing and camera calibration functionalities.


## Installation
Clone the repository to your local machine:
git clone https://github.com/aalexa201222/Camera-geometric-calibration.git

## Contributors
[Andreas Alexandrou](https://www.linkedin.com/in/andreas-alexandrou-056528242) <br />
Sotiris Zenios
