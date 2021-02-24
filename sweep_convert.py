# Python
import numpy as np
import time
import cv2

# Custom
try:
    import kittiloader.kitti as kitti
    import kittiloader.batch_loader as batch_loader
except:
    import kitti
    import batch_loader

# Data Loading Module
import torch.multiprocessing
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Value, cpu_count
import utils.img_utils as img_utils
import utils.misc_utils as misc_utils
import external.utils_lib.utils_lib as kittiutils
import torch.nn.functional as F

# Load
sweep_arr = np.load("/media/raaj/Storage/sweep_data/2021_02_23/2021_02_23_drive_0010_sweep/sweep/000001.npy").astype(np.float32)
velodata = np.fromfile("/media/raaj/Storage/sweep_data/2021_02_23/2021_02_23_drive_0010_sweep/lidar/000001.bin", dtype=np.float32).reshape((-1, 4))
#rgbimg = cv2.imread("/media/raaj/Storage/sweep_data/2021_02_23/2021_02_23_drive_0010_sweep/left_img/test.png")
rgbimg = cv2.imread("/media/raaj/Storage/sweep_data/2021_02_23/2021_02_23_drive_0010_sweep/left_img/000001.png")
rgbimg = cv2.resize(rgbimg, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

# print(rgbimg.shape)
# stop

# Params
large_intr = np.array([[189.3673075,   0.,        184.471915,    0.       ],
 [  0.,        189.3673075, 131.7250825,   0.       ],
 [  0.,          0.,          1.,          0.,       ]]).astype(np.float32)
M_velo2cam = np.array([[-0.08417412, -0.99644539, -0.00336614, -0.00763412],
 [ 0.18816988, -0.01257801, -0.98205594, -0.14401643],
 [ 0.97852276, -0.08329711,  0.18855976, -0.19474508],
 [ 0.,          0.,          0.,          1.        ]]).astype(np.float32)
large_size = [320, 256]
K_lc = np.array([
    [893.074542/2, 0.000000, 524.145998/2],
    [0.000000, 893.177518/2, 646.766885/2],
    [0.000000, 0.000000, 1.000000]
]).astype(np.float32)
D_lc = np.array([-0.033918, 0.027494, -0.001691, -0.001078, 0.000000]).astype(np.float32)
M_left2LC = np.array([[0.9985846877098083, 0.0018874829402193427, 0.0531516931951046, 0.07084044814109802], 
[-0.0029097397346049547, 0.9998121857643127, 0.019162006676197052, 0.0055979466997087], 
[-0.053105540573596954, -0.019289543852210045, 0.9984025955200195, 0.10840931534767151], 
[0.0, 0.0, 0.0, 1.0]]).astype(np.float32)

# Undistort
start = time.time()
for i in range(0, sweep_arr.shape[0]):
    sweep_arr[i, :,:, 0] = cv2.undistort(sweep_arr[i, :,:, 0], K_lc, D_lc)
    sweep_arr[i, :,:, 1] = cv2.undistort(sweep_arr[i, :,:, 1], K_lc, D_lc)
print((time.time()-start))

# Convert to Torch
sweep_arr_int = torch.tensor(sweep_arr[:,:,:,1]) # [128,640,512]
sweep_arr_z = torch.tensor(sweep_arr[:,:,:,0]) # [128,640,512]
lc_height = sweep_arr_int.shape[1]
lc_width = sweep_arr_int.shape[2]

# Half the LC image (This might be wrong, check again)
print(sweep_arr_int.shape)
#sweep_arr_int = F.interpolate(sweep_arr_int.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0) # [128,320,256]
#sweep_arr_z = F.interpolate(sweep_arr_z.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0) # [128,320,256]
K_lc /= 2
K_lc[2,2] = 1.
lc_height = sweep_arr_int.shape[1]
lc_width = sweep_arr_int.shape[2]
print(sweep_arr_int.shape)

# Viz
combined_image_temp = np.max(sweep_arr_int.numpy(), axis=0) # 640, 512, 4
combined_image_temp = combined_image_temp[:,:].astype(np.uint8)
combined_image = cv2.cvtColor(combined_image_temp,cv2.COLOR_GRAY2RGB)

# Generate Depth maps
large_params = {"filtering": 2, "upsample": 0}
dmap_large = kittiutils.generate_depth(velodata, large_intr, M_velo2cam, large_size[0], large_size[1], large_params)
dmap_large = torch.tensor(dmap_large)
dmap_height = dmap_large.shape[0]
dmap_width = dmap_large.shape[1]

# Points
pts_img = img_utils.depth_to_pts(dmap_large.unsqueeze(0), large_intr) # [3, 256, 320]
pts_img = torch.cat([
    pts_img, torch.ones(1, pts_img.shape[1], pts_img.shape[2])
    ])
pts_img = pts_img.reshape(4, pts_img.shape[1]*pts_img.shape[2]) # [4, 81920]
pts_img = torch.matmul(torch.tensor(M_left2LC), pts_img) # Now in LC frame

# # Validate projection on rgb image?
# proj_points = torch.matmul(torch.tensor(large_intr), pts_img)
# proj_points[0,:] /= proj_points[2,:]
# proj_points[1,:] /= proj_points[2,:]
# for i in range(0, proj_points.shape[1]):
#     try:
#         cv2.circle(rgbimg,(int(proj_points[0,i]), int(proj_points[1,i])), 1, (0,255,0), -1)
#     except:
#         pass

# Lousy Projection
K_lc = torch.tensor(K_lc)
K_lc = torch.cat([K_lc, torch.zeros(3,1)], dim=1)
proj_points = torch.matmul(K_lc, pts_img)
proj_points[0,:] /= proj_points[2,:]
proj_points[1,:] /= proj_points[2,:]
proj_points[2,:] = pts_img[2, :] # Copy the Z values
proj_points = proj_points[:,:].T
print(proj_points.shape) # 81920, 2

# Copy Feats
start = time.time()
feat_tensor = torch.zeros((128, proj_points.shape[0])) # 81920, 2
mask_tensor = torch.zeros((1, proj_points.shape[0])) # 81920, 2
for i in range(0, proj_points.shape[0]):
    lc_pix_pos = (int(proj_points[i,0]), int(proj_points[i,1])) # ADD 0.5 HERE?
    z_val = proj_points[i,2]
    if lc_pix_pos[0] < 0 or lc_pix_pos[1] < 0 or lc_pix_pos[0] >= lc_width or lc_pix_pos[1] >= lc_height:
        continue
    if z_val > 15:
        continue
    feature_int = sweep_arr_int[:, lc_pix_pos[1], lc_pix_pos[0]]
    feature_z = sweep_arr_z[:, lc_pix_pos[1], lc_pix_pos[0]]
    
    # Handle the nan ops? Have to handle during training? The first one is nan is bad
    # Need to check the Z value of sweep_arr
    if torch.isnan(feature_z[0]):
        continue

    feat_tensor[:, i] = feature_int

    mask_tensor[:, i] = 1
print((time.time()-start))

# Resize
feat_tensor = feat_tensor.reshape(128, dmap_height, dmap_width)
mask_tensor = mask_tensor.reshape(1, dmap_height, dmap_width)

"""
Todo in C++
* Objective is to get a [128, 256, 320] FV in the RGB frame
* Try to get a [256 * 320, 2] tensor that stores the pixel positions in LC frame? - Try in Simple Pytorch first
* Then we need to generate a tensor somehow that reference sweep_arr?

"""

# # Validate Projection?
# K_lc = torch.tensor(K_lc)
# K_lc = torch.cat([K_lc, torch.zeros(3,1)], dim=1)
# proj_points = torch.matmul(K_lc, pts_img)
# proj_points[0,:] /= proj_points[2,:]
# proj_points[1,:] /= proj_points[2,:]
# for i in range(0, proj_points.shape[1]):
#     try:
#         cv2.circle(combined_image,(int(proj_points[0,i]), int(proj_points[1,i])), 1, (0,255,0), -1)
#     except:
#         pass

# print(proj_points)




"""
3. We want the LC sweep data in the RGB frame.
    1. Depthmap -> [H, W, 4] XYZ1
    2. Transform to NIR -> [H, W, 4] XYZ1
    3. Reshape back to [H*W, 4]
    4. Pass into a C++ function that takes in [N,4] cloud, IR intrinsics, the Feature Map
        Project it using techniques to find the corresponding pixel
        Copy over into [N, 128]
4. Handle the various cases of range over extend past 18m etc. and handling nans

WE ARE NOT HANDLING THE OUT OF FOV IN LC!!!
WE ARE NOT HANDLING EXCEEDED DEPTHS!!
"""



print(sweep_arr.shape)
print(velodata.shape)
print(dmap_large.shape)


while 1:
    cv2.imshow("image", (combined_image))
    cv2.imshow("dmap_large", (dmap_large.numpy()/10))
    cv2.imshow("rgbimg", rgbimg/255. + mask_tensor.squeeze(0).unsqueeze(-1).numpy())
    cv2.imshow("mask", mask_tensor.squeeze(0).numpy())
    cv2.waitKey(15)
