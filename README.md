# 3d-reconstruction
This project is to generate a 3d-reconstruction of an 90-degree surface using just a pair of stereo images taken of that surface. This depth estimation is done using Triangulation of SIFT features correspondenses between the two stereo images. The implementation is done using OpenCV and Open3D.

### Results

- Stereo images
<img width="333" alt="image" src="https://github.com/tusharparimi/3d-reconstruction/assets/93556280/108a2b5b-866e-4c8f-926f-edfdf7355638">

- 3d reconstruction results (front & top views)
<img width="220" alt="image" src="https://github.com/tusharparimi/3d-reconstruction/assets/93556280/2c72051d-d951-490e-a374-8c5cbbf425e6">
<img width="179" alt="image" src="https://github.com/tusharparimi/3d-reconstruction/assets/93556280/1ff5e121-8459-4e28-a77e-824b3770366c">

video for 3d view- https://youtu.be/Uw53UdLdsOM?si=gAgVVJTPpVnZ2maY

### Implementation steps
- First we need a calibrated camera this can be done taking bunch of pictures of a chessboard pattern with your camera and feeding them into the chess_calib.py script. This will gives you the camera's intrinsic and extrinsic parameters(camera mtx, distortion coeffs and reproojection error). These help calibrate an uncalibrated camera.
- Once you have the camera parameters you undistort your stereo images using those params and extract the SIFT keypoints and find matches between the pair of sstereo images.
<img width="371" alt="image" src="https://github.com/tusharparimi/3d-reconstruction/assets/93556280/3f99fddf-f80d-44b8-a22c-469503f38d62">
<img width="382" alt="image" src="https://github.com/tusharparimi/3d-reconstruction/assets/93556280/d5748d70-1729-4cd4-b1b2-e679a563ac4b">

- Now use Triangulation(Linear least squares method used here particularly) to find 3d estimates of the SIFT correspondenses.
- Use Chierality constraint to find the actual projection
- Finally, the 3d points are saved as a 3d poiint cloud in a .pcd file and visualized using Open3d.




