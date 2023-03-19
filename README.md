# PROJECT 1
## ENPM673 - Perception for Autonomous Robots

## Dependencies
python 3.11 (any version above 3 should work)
Python running IDE (I used VS Code)

## Libraries
1. OpenCV
2. NumPy
3. Matplotlib

## Contents
1. ball.mov
2. pc1.csv
3. pc2.csv
4. problem1.py
5. problem2.py
6. Report.pdf
7. README.md
8. problem1_output.png
9. problem2_output.png

## Installation Instructions
1. Download the zip file and extract it
2. Install python and the required dependencies: pip install opencv-python numpy matplotlib

## Problem 1 - Ball Trajectory Prediction
This code analyzes the trajectory of a ball in a given video and predicts its landing spot using computer vision and curve fitting techniques.

### Features
1. Extracts the coordinates of the center of the ball from a given video file
2. Fits a parabolic curve to the extracted coordinates using the standard least squares method
3. Predicts the x-coordinate of the ball's landing spot based on the fitted curve
4. Displays the trajectory of the ball with the predicted landing spot in a scatter plot

### Usage
1. Place the video file in the same directory as the code file
2. Set the filename of the video in the cv2.VideoCapture() function call in line 5 of the code
3. Run the code: problem1.py

### Example Output

Equation of the parabolic trajectory of the ball: y = 0.000589700381382697x^2 + -0.5977806132188792x + 455.14988106710655
Possible landing spots: x = 1372.9428397397155, -359.2405731458969
The x-coordinate of the ball's landing spot is (As the other point is out of frame): 1372.9428397397155
The ball lands at (1372.9428397397155,746)

## Output
![Ball Trajectory](https://github.com/vishnumandala/Ball-Trajectory-Prediction-and-Point-Cloud-Processing/blob/main/problem1_output.png)

## Problem 2 - Point Clouds Processing
This code for processing point clouds using the NumPy and Matplotlib libraries. 

### Features
1. Computes the covariance matrix for a given point cloud
2. Computes the surface normal and magnitude of a flat, ground plane in a point cloud
3. Fits a plane to a given set of points using Standard Least Squares, Total Least Squares, and RANSAC methods

### Usage
1. Place the data files in the same directory as the code file
2. Set the filenames in the np.loadtxt function call in lines 5 and 6 of the code
3. Run the code: problem2.py

### Example Output

Covariance Matrix for pc1.csv:
 [[ 33.7500586   -0.82513692 -11.39434956]
 [ -0.82513692  35.19218154 -23.23572298]
 [-11.39434956 -23.23572298  20.62765365]]
Surface Normal to the flat, ground plane ----- Vector: [0.28616428 0.53971234 0.79172003] and Magnitude: 1.0
 
Coefficients for Standard Least Squares :  [-0.2997447  -0.67026719  3.44546813]
Coefficients for Total Least Squares :  [ 0.25336279  0.56455808  0.78554533 -2.70518836]
Coefficients for RANSAC :  [ 0.25336279  0.56455808  0.78554533 -2.70518836 22.53942768]

## Output
![Point Clouds Processing](https://github.com/vishnumandala/Ball-Trajectory-Prediction-and-Point-Cloud-Processing/blob/main/problem2_output.png)
