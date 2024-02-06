#chessboard camera calibration

import numpy as np
import cv2
import glob

# PART 3:------------------------------------------------------------------------------------------------------------------------------------------------------ 

print("\nPART 3: --------------------------------------------------------------------")
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./calib_data/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners2, ret)
        cv2.imshow(fname, img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

#calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("\ndistortion coeffs:\n", dist)

#refining the camera matrix with scaling parameter alpha=1 (all pixels are retained with some extra black images)
img1 = cv2.imread('./calib_data/IMG_3276.jpg')
img1 = cv2.resize(img1, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
cv2.imshow("IMG_3276", img1)
cv2.waitKey(0)
h,  w = img1.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print("\ncamera_matrix:\n", newcameramtx)

# undistort
dst1 = cv2.undistort(img1, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst1 = dst1[y:y+h, x:x+w]
cv2.imshow('calib_result_IMG_3276', dst1)
cv2.waitKey(0)

#re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
rp_error=np.array((mean_error/len(objpoints)))
print( "\nRe-projection error: {}".format(rp_error))

#saving the camera matrix, distortion coeffs and re-projection error in a npz file for future use
np.savez("calib_results.npz", cameramtx=newcameramtx, dist_coeffs=dist, rp_error=rp_error)

print("----------------------------------------------------------------------------")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------      




'''
# PART 1:------------------------------------------------------------------------------------------------------------------------------------------------------ 

print("\nPART 1: --------------------------------------------------------------------")
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./part1_data/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        cv2.imshow(fname, img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

#calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("\ndistortion coeffs:\n", dist)

#refining the camera matrix with scaling parameter alpha=1 (all pixels are retained with some extra black images)
img = cv2.imread('./part1_data/left12.jpg')
cv2.imshow("part1_left12", img)
cv2.waitKey(0)
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print("\ncamera_matrix:\n", newcameramtx)


# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst1 = dst[y:y+h, x:x+w]
cv2.imshow('part1_calib_result_left12', dst1)
cv2.waitKey(0)

#re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
rp_error=np.array((mean_error/len(objpoints)))
print( "\nRe-projection error: {}".format(rp_error))

#saving the camera matrix, distortion coeffs and re-projection error in a npz file for future use
np.savez("part1_results.npz", cameramtx=newcameramtx, dist_coeffs=dist, rp_error=rp_error)

print("----------------------------------------------------------------------------")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------      


# PART 2:------------------------------------------------------------------------------------------------------------------------------------------------------ 

print("\nPART 2: --------------------------------------------------------------------")
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((11*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:11].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./part2_data/calibration_data/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,11), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (9,9), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,11), corners2, ret)
        cv2.imshow(fname, img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

#calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("\ndistortion coeffs:\n", dist)

#refining the camera matrix with scaling parameter alpha=1 (all pixels are retained with some extra black images)
img = cv2.imread('./part2_data/calibration_data/IMG_6516.jpg')
cv2.imshow("part2_IMG_6516", img)
cv2.waitKey(0)
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print("\ncamera_matrix:\n", newcameramtx)


# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst2 = dst[y:y+h, x:x+w]
cv2.imshow('part2_calib_result_IMG6516', dst2)
cv2.waitKey(0)

#re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
rp_error=np.array((mean_error/len(objpoints)))
print( "\nRe-projection error: {}".format(rp_error))

#saving the camera matrix, distortion coeffs and re-projection error in a npz file for future use
np.savez("part2_results.npz", cameramtx=newcameramtx, dist_coeffs=dist, rp_error=rp_error)

print("----------------------------------------------------------------------------")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


