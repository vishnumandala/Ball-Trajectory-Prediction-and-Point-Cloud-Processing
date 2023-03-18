import numpy as np
import matplotlib.pyplot as plt

# Load the data from the CSV files
pc1 = np.loadtxt('pc1.csv', delimiter=',')
pc2 = np.loadtxt('pc2.csv', delimiter=',')

data = np.concatenate([pc1, pc2])                                   # Combine the two dataframes into one

#Define a function to plot a graph
def plotting(f,i):
    print("Coefficients for" , f, ": ", n)                          # Print the coefficients of the fitted plane
    ax = fig.add_subplot(1, 3, i, projection='3d')    
    ax.scatter(data[:,0], data[:,1], data[:,2], c='r', marker='o')
    x_vals, y_vals = np.meshgrid(np.linspace(min(data[:,0]), max(data[:,0]), 10), np.linspace(min(data[:,1]), max(data[:,1]), 10)) 
    # Plot the data and the fitted plane
    if f == 'Standard Least Squares':
        z_vals = n[0]*x_vals + n[1]*y_vals + n[2]
    elif f == 'Total Least Squares':
        z_vals = (-n[0]*x_vals - n[1]*y_vals - n[3]) / n[2]
    elif f == 'RANSAC':       
        z_vals = (-n[0]*x_vals - n[1]*y_vals - n[3]) / n[2]
    ax.plot_surface(x_vals, y_vals, z_vals, alpha = 0.5)
    ax.set_xlabel('X-Coordinates')
    ax.set_ylabel('Y-Coordinates')
    ax.set_zlabel('Z-Coordinates')
    plt.title(f)

'''--------------------------------------------------Question 1: For pc1.csv------------------------------------------------------------------------------------'''

'''----------------------a: Computing Covariance Matrix---------------------'''
#Define a function to calculate the Covariance of a Matrix
def cov_mat(M):
    mean = np.mean(M,axis=0)                                        # Compute the mean of all dimensions
    center = M - mean                                               # Center the point cloud data around the mean

    # Compute and print the covariance matrix
    cov_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cov_matrix[i][j] = np.sum(center[:,i] * center[:,j]) / (len(center) - 1)
    return cov_matrix, mean

mat, _ = cov_mat(pc1)
print('\nCovariance Matrix for pc1.csv:\n', mat)

'''--------------b:  Computing Magnitude and Direction of Surface Normal using Covariance Matrix------------------'''
eig_vals, eig_vecs = np.linalg.eig(mat)                             # Compute the Eigenvectors and Eigenvalues of the Covariance Matrix
surface_normal = eig_vecs[:, np.argmin(eig_vals)]                   # Compute the Surface Normal as the Eigenvector corresponding to the smallest Eigenvalue
surface_normal_magnitude = np.linalg.norm(surface_normal)           # Compute the magnitude of the Surface Normal

print(f"Surface Normal to the flat, ground plane ----- Vector: {surface_normal} and Magnitude: {surface_normal_magnitude}\n")   # Print the Surface Normal Direction and its Magnitude

'''-------------------------------------------------Question 2. For pc1.csv and pc2.csv -------------------------------------------------------------------------'''

'''----------------------Part (a)---------------------'''

'''-----------------Standard Least Squares----------------------'''
A = np.column_stack((data[:,0], data[:,1], np.ones(len(data))))
b = data[:,2]

# Compute the Normal Equations Matrix
ATA = np.matmul(A.T, A)
ATb = np.matmul(A.T, b)

n = np.dot(np.linalg.inv(ATA), ATb)
fig = plt.figure()
plotting('Standard Least Squares',1)                                


'''-----------------Total Least Squares----------------------'''
c, m = cov_mat(data)

#Creating a Manual Partial Singular Value Decomposition Function
ATA = np.dot(c.T, c)                            
val, vec = np.linalg.eig(ATA)                                       # Compute the eigenvalues and eigenvectors of A^T A
S = np.sqrt(val)                                                    # Compute the singular values
V = vec[:,np.argsort(-S)].T                                         # Sort the singular vectors in descending order
n = V[-1, :]                                                        # Extract the last row of V to get the normal vector
n = n / np.linalg.norm(n)                                           # Normalize the vector
# Translate the plane back to its original position
o = -np.dot(n, m)                                                   # Calculate offset of the plane
n = np.append(n, o)
plotting('Total Least Squares',2)

'''----------------------Part (b)---------------------'''

'''-----------------RANSAC----------------------'''
threshold = 0.1
max_iters = 1000
best_inliers = []
i=1
while i <= max_iters:
    # Randomly sample 3 points from the data
    sample = data[np.random.choice(data.shape[0], size=3, replace=False)]
    
    # Fit a plane to the sampled points
    p1, p2, p3 = sample
    normal = np.cross(p2-p1, p3-p1)                                 # Using the normal vector obtained by the cross-product of two vectors
    if np.linalg.norm(normal) < 1e-6:                               # Check if magnitude of normal vector is very small
        continue
    o = -np.dot(normal, p1)
    plane = np.concatenate((n, [o]))
    distances = np.abs(np.dot(data, plane[:3]) + plane[3]) / np.linalg.norm(plane[:3])  # Calculate the distance of all points from the plane
    inliers = np.where(distances <= threshold)[0]                   # Count the number of inliers (points within the threshold distance from the plane)
    
    # Update the best model if the current model has more inliers
    if len(inliers) > len(best_inliers):
        n = plane
        best_inliers = inliers
    i += 1
plotting('RANSAC',3)

fig.suptitle('Fitting a surface to the data using various estimation Algorithms')                                                    
plt.show()
print('\n')