import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('ball.mov')                          # Read the video

'''-----------------Question 1. Finding the coordinates of the Center of the Ball------------------'''
#Create lists to store the x-coordinates and y-coordinates of the center of the ball
x = []
y = []

# Loop over the frames of the video
while cap.isOpened():
    ret, frame = cap.read()                                 # Read a frame
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)            # Convert frame to HSV color space
    mask = cv2.inRange(hsv, (0,150,130), (6,255,255))       # Filter the Red Channel
    ycoords, xcoords = np.where(mask > 0)                   #Unpacks the non-zero indices obtained through the mask into xcoords and ycoords
    
    # Find center point of the ball
    if len(ycoords) > 0 and len(xcoords) > 0:
        center  = (int(np.mean(xcoords)), int(np.mean(ycoords)))
        cv2.circle(frame, center, 3, (255, 0, 0), -1)       #Draws a blue, solid circle at the center of the ball
        x.append(center[0])
        y.append(center[1])
    # Display the frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(1) 
    
# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

'''-----------------Question 2. Using Standard Least Squares Method to fit a curve to the extracted coordinates------------------'''
#Convert lists to numpy arrays
x = np.array(x)
y = np.array(y)

#Fit parabola to data using Standard Least Squares
M = np.array([x**2, x, np.ones(len(x))]).T                  #Matrix with 3 columns containing 'x square' values, 'x' values and '1's is created and transposed
coeffs = np.linalg.inv(M.T.dot(M)).dot(M.T).dot(y)          #Co-efficients of x^2, x, constant for the equation y=ax^2+bx+c are obtained
a, b, c = coeffs
print(f"\nEquation of the parabolic trajectory of the ball: y = {a}x^2 + {b}x + {c}")

'''-----------------Question 3. Finding the x-coordinate of the ball's landing spot------------------'''
# Compute the x-coordinate of the landing spot
y_first = y[0]
y_land = y_first + 300
d = b**2 - 4*a*(c - y_land)
if d < 0:
    print("Error: No real solutions")
else:
    x1 = (-b + np.sqrt(d)) / (2*a)
    x2 = (-b - np.sqrt(d)) / (2*a)
    print(f"Possible landing spots: x = {x1}, {x2}")
    # Check if roots are negative
    if x1 > 0 and x2 < 0:
        print(f"The x-coordinate of the ball's landing spot is (As the other point is out of frame): {x1}\nThe ball lands at ({x1},{y_land})\n")
        l = x1
    elif x1 < 0 and x2 > 0:
        print(f"The x-coordinate of the ball's landing spot is (As the other point is out of frame): {x2}\nThe ball lands at ({x2},{y_land})\n")
        l = x2
    else: 
        print(f"The ball does not land in frame\n")

# Plot the data with the best fit curve
plt.scatter(x, y)
plt.scatter(l, y_land, marker='o', color='green')
plt.text(l+10, y_land+10, "Landing spot", color='green')
plt.plot(x, a*x**2 + b*x + c, 'r-')
plt.gca().invert_yaxis()
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Trajectory of the Ball')
plt.show()