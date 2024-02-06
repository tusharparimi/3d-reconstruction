# 3D Triangulation using stereo images and point cloud visualization

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

#reading the input images
img1=cv2.imread("./stereo_images/left.jpg")
img2=cv2.imread("./stereo_images/right.jpg")
img1=cv2.resize(img1, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
img2=cv2.resize(img2, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

#reading the calibration params using the same params computed in lab-8 as am using the same camera
calib_data=np.load("calib_data.npz")
mtx=calib_data['cameramtx']
dist=calib_data['dist_coeffs']
roi=calib_data['roi']

# undistort the input images
dst1 = cv2.undistort(img1, mtx, dist, None, mtx)
dst2 = cv2.undistort(img2, mtx, dist, None, mtx)

# crop the image
x, y, w, h = roi
dst1 = dst1[y:y+h, x:x+w]
dst2 = dst2[y:y+h, x:x+w]

#display undistorted images
cv2.imshow('left_calib', dst1)
cv2.waitKey(0)
cv2.imshow('right_calib', dst2)
cv2.waitKey(0)

#grayscale
gray1=cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)

# finding the keypoints and descriptors with SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)
kimg1=np.zeros(shape=dst1.shape)
kimg2=np.zeros(shape=dst2.shape)
kimg1=cv2.drawKeypoints(gray1,kp1,kimg1,color=(0,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kimg2=cv2.drawKeypoints(gray2,kp2,kimg2,color=(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("kimg1", kimg1)
cv2.waitKey(0)
cv2.imshow("kimg2", kimg2)
cv2.waitKey(0)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

# finding matches based on FLANN
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
pts1 = []
pts2 = []
matchesMask=[[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
                matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,255),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
matchimg = cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,matches,None,**draw_params)
cv2.imshow("matchimg", matchimg)
cv2.waitKey(0)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

#calculating the fundamental matrix using 2d-2d correspondences
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

#verifying the ur.T*F*ul==0 condition 
verified=True
for i in range(pts1.shape[0]):
        ul=np.ones((3,1))
        ur=np.ones((3,1))
        ul[0:2,:]=pts1[i].reshape((2,1))
        ur[0:2,:]=pts2[i].reshape((2,1))
        if not int(np.matmul(np.matmul(ur.T, F), ul))==0:
                verified=False
                print("ur.T*F*ul==0 not verified!!!")                
if verified==True:
        print("ur.T*F*ul==0 verified.")

#calculating the essential matrix
E=np.matmul(np.matmul(mtx.T,F),mtx)

#verifying det(E)=0
if int(np.linalg.det(E))==0:
        print("det(E)==0 verified.")

#SVD of the essential matrix for computing the projection matrices
U, D, VT=np.linalg.svd(E)
u3=U[:,2:3]
S=np.array([[0,-1,0],[1,0,0],[0,0,1]])
R1=np.matmul(np.matmul(U,S),VT)
R2=np.matmul(np.matmul(U,S.T),VT)

#4 mathematically possible projection matrices
P0=np.zeros((3,4))
P0[:3,:3]=np.diag([1,1,1])
P0k=np.matmul(mtx,P0)

P11=np.zeros((3,4))
P11[:3,:3]=R1
P11[:,3:4]=u3
P11k=np.matmul(mtx,P11)

P12=np.zeros((3,4))
P12[:3,:3]=R1
P12[:,3:4]=-u3
P12k=np.matmul(mtx,P12)

P13=np.zeros((3,4))
P13[:3,:3]=R2
P13[:,3:4]=u3
P13k=np.matmul(mtx,P13)

P14=np.zeros((3,4))
P14[:3,:3]=R2
P14[:,3:4]=-u3
P14k=np.matmul(mtx,P14)

P1=[P11,P12,P13,P14]
P1k=[P11k,P12k,P13k,P14k]

#Triangulation using Linear least squares for finding the 3d estimates
def LinearLSTriangulation(pts1, P0, pts2, P1k):
        x3d=np.zeros((pts1.shape[0],4,3))
        for i in range(pts1.shape[0]):
                for j in range(len(P1)):
                        A=np.zeros((4,4))
                        A[0:1,:]=pts1[i][1]*P0k[2:3,:]-P0k[1:2,:]
                        A[1:2,:]=pts1[i][0]*P0k[2:3,:]-P0k[0:1,:]
                        A[2:3,:]=pts2[i][1]*(P1k[j])[2:3,:]-P1k[j][1:2,:]
                        A[3:4,:]=pts2[i][0]*(P1k[j])[2:3,:]-P1k[j][0:1,:]
                        b=-A[:,3:4]
                        A=A[:,0:3]
                        U, S, VT=np.linalg.svd(A)
                        Si=1/S
                        for k in range(Si.shape[0]):
                                if Si[k]==np.inf:
                                        Si[k]=0
                        Smat=np.zeros((3,4))
                        Smat[:3,:3]=np.diag(Si)                        
                        x=np.matmul(np.matmul(np.matmul(VT.T,Smat),U.T),b).reshape(-1)
                        x3d[i][j]=x
        return x3d

x3d=LinearLSTriangulation(pts1, P0k, pts2, P1k)


#using Chierality constraint to find the actual projection matrix and its 3d estimates
def chierality(P1, x3d):
        n={0:0,1:0,2:0,3:0}
        for i in range(x3d.shape[0]):
                for j in range(len(P1)):
                        if np.matmul(P1[j][2:3,0:3], x3d[i][j].reshape(3,1)-P1[j][:,3:4])>0:
                                n[j]=n[j]+1
        return n

n=chierality(P1k, x3d)
nmax=max(zip(n.values(),n.keys()))[1]               
x3dfinal=x3d[:,nmax:nmax+1,:].reshape(x3d.shape[0],x3d.shape[2]).astype(np.float32)
P1final=P1k[nmax]

#Finding the projection of the 3d estimates again on the 2 image planes
p02d=np.zeros((x3dfinal.shape[0],2))
p12d=np.zeros((x3dfinal.shape[0],2))
for i in range(x3dfinal.shape[0]):
        x3dh=np.ones((4,))
        x3dh[0:3]=x3dfinal[i]
        x3dh=x3dh.reshape(1,4).T
        p02di=np.matmul(P0k,x3dh)
        p02di=p02di/p02di[2,:]
        p02d[i]=p02di[0:2,:].reshape(2,)
        p12di=np.matmul(P1final,x3dh)
        p12di=p12di/p12di[2,:]
        p12d[i]=p12di[0:2,:].reshape(2,)

pts1=pts1.astype(np.float64)
pts2=pts2.astype(np.float64)
p02d=p02d.astype(np.float64)
p12d=p12d.astype(np.float64)

#Calculating the reprojection errors 
sum_error1=0
sum_error2=0
for i in range(pts1.shape[0]):
        error1 = cv2.norm(pts1[i], p02d[i], cv2.NORM_L2)/p02d[i].shape[0]
        sum_error1 += error1
        error2 = cv2.norm(pts2[i], p12d[i], cv2.NORM_L2)/p12d[i].shape[0]
        sum_error2 += error2
rp_error1=np.array((sum_error1/pts1.shape[0]))
rp_error2=np.array((sum_error2/pts2.shape[0]))
print("Reprojection error cam1: ", rp_error1)
print("Reprojection error cam2: ", rp_error2)


#visualizing the 3d estimates as a 3D point cloud and saving the point cloud as a .pcd file  (using open3d)
def SavePCDToFile(x3dfinal, p12d, dst1):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x3dfinal)
        rgb=np.zeros((p02d.shape[0],3))
        for i in range(p02d.shape[0]):
                rgb[i]=dst1[int(p02d[i][1]), int(p02d[i][0]), :]
        rgb=rgb.astype(np.float64)/255.0
        pcd.colors=o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud("point_cloud.pcd", pcd)
        o3d.visualization.draw_geometries([pcd],
                                        front=[ -0.22712054279188112, -0.66890133401968066, 0.70780453826505874 ],
                                        lookat=[ 0.0055884420871734619, 0.20136691629886627, -1.4273181259632111 ],
                                        up=[ 0.1195391086083254, -0.74045043349079642, -0.66139515953506856 ],
                                        zoom=0.69999999999999996)
         
SavePCDToFile(x3dfinal, p12d, dst2)
               







