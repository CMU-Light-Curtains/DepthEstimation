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
sweep_arr = np.load("/media/raaj/Storage/sweep_data/2021_02_25/2021_02_25_drive_0003_sweep/sweep/000055.npy").astype(np.float32)
velodata = np.fromfile("/media/raaj/Storage/sweep_data/2021_02_25/2021_02_25_drive_0003_sweep/lidar/000055.bin", dtype=np.float32).reshape((-1, 4))
rgbimg = cv2.imread("/media/raaj/Storage/sweep_data/2021_02_25/2021_02_25_drive_0003_sweep/left_img/000055.png")
nirimg = cv2.imread("/media/raaj/Storage/sweep_data/2021_02_25/2021_02_25_drive_0003_sweep/nir_img/000055.png")
rgbimg = cv2.resize(rgbimg, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
nirimg = cv2.resize(nirimg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

"""
Speed up
Make the true mask tensor
Integrate with KITTI
Check if right image works fine by putting viz code etc.
Try to speed up code
"""

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
# Half the LC Int
K_lc /= 2
K_lc[2,2] = 1.
lc_size = [256, 320]

# # Right
# rgbimg = cv2.imread("/media/raaj/Storage/sweep_data/2021_02_25/2021_02_25_drive_0003_sweep/right_img/000014.png")
# rgbimg = cv2.resize(rgbimg, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
# M_velo2cam = np.array([[-0.08417412, -0.99644539, -0.00336614, -0.75311549],
#  [ 0.18816988, -0.01257801, -0.98205594, -0.14401643],
#  [ 0.97852276, -0.08329711,  0.18855976, -0.19474508],
#  [ 0. ,         0.   ,       0.     ,     1.  ,      ]]).astype(np.float32)
# M_left2LC = np.array([[0.9985846877098083, 0.0018874829402193427, 0.0531516931951046, 0.07084044814109802], 
# [-0.0029097397346049547, 0.9998121857643127, 0.019162006676197052, 0.0055979466997087], 
# [-0.053105540573596954, -0.019289543852210045, 0.9984025955200195, 0.10840931534767151], 
# [0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
# M_left2Right = np.array([[ 1.   ,       0.   ,       0.      ,   -0.74548137],
#  [ 0.   ,       1.        ,  0.    ,      0.        ],
#  [ 0.     ,     0.     ,     1.   ,       0.        ],
#  [ 0.    ,      0.       ,   0.      ,    1.        ]]).astype(np.float32)
# M_right2Left = np.linalg.inv(M_left2Right)
# M_left2LC = np.matmul(M_right2Left, M_left2LC)

# Undistort LC
start = time.time()
nirimg = cv2.undistort(nirimg, K_lc, D_lc)
for i in range(0, sweep_arr.shape[0]):
    sweep_arr[i, :,:, 0] = cv2.undistort(sweep_arr[i, :,:, 0], K_lc, D_lc)
    sweep_arr[i, :,:, 1] = cv2.undistort(sweep_arr[i, :,:, 1], K_lc, D_lc)
end = time.time()
print(end-start)

# Generate Depth maps
large_params = {"filtering": 2, "upsample": 0}
dmap_large = kittiutils.generate_depth(velodata, large_intr, M_velo2cam, large_size[0], large_size[1], large_params)
dmap_large = torch.tensor(dmap_large)
dmap_height = dmap_large.shape[0]
dmap_width = dmap_large.shape[1]

# Compute
start = time.time()
feat_int_tensor, feat_z_tensor, mask_tensor, train_mask_tensor, combined_image = img_utils.lcsweep_to_rgbsweep(
    sweep_arr=sweep_arr, dmap_large=dmap_large, rgb_intr=large_intr, rgb_size=large_size, lc_intr=K_lc, lc_size=lc_size, M_left2LC=M_left2LC)
end = time.time()
print(end-start)

# Reduce Tensor Size here itself?
print(mask_tensor.shape)
feat_int_tensor = F.interpolate(feat_int_tensor.unsqueeze(0), size=[64, 80], mode='nearest').squeeze(0)
feat_z_tensor = F.interpolate(feat_z_tensor.unsqueeze(0), size=[64, 80], mode='nearest').squeeze(0)
mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=[64, 80], mode='nearest').squeeze(0)
rgbimg = cv2.resize(rgbimg, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

# During training we need to generate a mask to handle the nans. Basically we mask out those.
# Create a mask volume for those nans?
# Combine it with the image mask?

import random
from matplotlib import pyplot as plt
def mouse_callback(event,x,y,flags,param):
    global mouseX,mouseY, combined_image, sweep_arr
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y
        rgb = (random.random(), random.random(), random.random())

        # Extract vals
        disp_z = feat_z_tensor[:,y,x]
        disp_i = feat_int_tensor[:,y,x]/255.
        print(disp_i)
        print(disp_z)
        first_nan = np.isnan(disp_z).argmax(axis=0)
        if first_nan:
            disp_z = disp_z[0:first_nan]
            disp_i = disp_i[0:first_nan]

        plt.plot(disp_z, disp_i, c=rgb, marker='*')
        plt.pause(0.1)


cv2.namedWindow('rgbimg')
cv2.setMouseCallback('rgbimg',mouse_callback)


while 1:
    #cv2.imshow("image", (combined_image))
    #cv2.imshow("dmap_large", (dmap_large.numpy()/10))
    cv2.imshow("rgbimg", rgbimg/255. + mask_tensor.squeeze(0).unsqueeze(-1).numpy())
    #cv2.imshow("mask", mask_tensor.squeeze(0).numpy())
    cv2.waitKey(15)



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
