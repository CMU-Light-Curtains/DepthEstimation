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
import json
from easydict import EasyDict

def load_datum(path, name, indx):
    datum = dict()
    # Generate Paths
    date = name.split("_drive")[0]
    index_str = "%06d" % (indx,)
    sweep_path = path + "/" + date + "/" + name + "/sweep/" + index_str + ".npy"
    left_img_path = path + "/" + date + "/" + name + "/left_img/" + index_str + ".png"
    right_img_path = path + "/" + date + "/" + name + "/right_img/" + index_str + ".png"
    nir_img_path = path + "/" + date + "/" + name + "/nir_img/" + index_str + ".png"
    velo_path = path + "/" + date + "/" + name + "/lidar/" + index_str + ".bin"
    json_path = path + "/" + date + "/" + name + "/calib.json"
    # Load Data
    datum["sweep_arr"] = np.load(sweep_path).astype(np.float32)
    datum["velodata"] = np.fromfile(velo_path, dtype=np.float32).reshape((-1, 4))
    datum["left_img"] = cv2.imread(left_img_path)
    datum["right_img"] = cv2.imread(right_img_path)
    datum["nir_img"] = cv2.imread(nir_img_path)
    datum["left_img"] = cv2.resize(datum["left_img"], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    datum["nir_img"] = cv2.resize(datum["nir_img"], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    datum["right_img"] = cv2.resize(datum["right_img"], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    # Load Json
    with open(json_path) as json_file:
        calib = json.load(json_file)
    datum["large_intr"] = np.array(calib["left_P"]).astype(np.float32) / 4.
    datum["large_intr"][2,2] = 1.
    datum["M_velo2left"] = np.linalg.inv(np.array(calib["left_2_lidar"]).astype(np.float32))
    datum["large_size"] = [datum["left_img"].shape[1], datum["left_img"].shape[0]]
    datum["M_left2right"] = np.array(calib["left_2_right"]).astype(np.float32)
    datum["M_right2left"] = np.linalg.inv(datum["M_left2right"])
    datum["M_left2LC"] = np.array(calib["left_2_lc"]).astype(np.float32)
    datum["M_right2LC"] = np.matmul(datum["M_right2left"], datum["M_left2LC"])
    datum["D_lc"] = np.array([-0.033918, 0.027494, -0.001691, -0.001078, 0.000000]).astype(np.float32)
    datum["K_lc"] = np.array([
        [893.074542/2, 0.000000, 524.145998/2],
        [0.000000, 893.177518/2, 646.766885/2],
        [0.000000, 0.000000, 1.000000]
    ]).astype(np.float32)
    datum["K_lc"] /= 2
    datum["K_lc"][2,2] = 1.
    datum["lc_size"] = [256, 320]
    datum["M_velo2right"] = np.matmul(datum["M_left2right"], datum["M_velo2left"])
    datum["M_velo2LC"] = np.matmul(datum["M_left2LC"], datum["M_velo2left"])

    # Easydict
    datum = EasyDict(datum)
    return datum

# Load
datum = load_datum("/media/raaj/Storage/sweep_data", "2021_03_05_drive_0004_sweep", 6)

# Undistort LC
datum.nir_img = cv2.undistort(datum.nir_img, datum.K_lc, datum.D_lc)
for i in range(0, datum.sweep_arr.shape[0]):
    datum.sweep_arr[i, :,:, 0] = cv2.undistort(datum.sweep_arr[i, :,:, 0], datum.K_lc, datum.D_lc)
    datum.sweep_arr[i, :,:, 1] = cv2.undistort(datum.sweep_arr[i, :,:, 1], datum.K_lc, datum.D_lc)

# Depths
large_params = {"filtering": 2, "upsample": 0}
datum["left_depth"] = kittiutils.generate_depth(datum.velodata, datum.large_intr, datum.M_velo2left, datum.large_size[0], datum.large_size[1], large_params)
datum["right_depth"] = kittiutils.generate_depth(datum.velodata, datum.large_intr, datum.M_velo2right, datum.large_size[0], datum.large_size[1], large_params)
datum["lc_depth"] = kittiutils.generate_depth(datum.velodata, datum.K_lc, datum.M_velo2LC, datum.lc_size[0], datum.lc_size[1], large_params)

# Visualize Depth Check
left_depth_debug = datum.left_img.copy().astype(np.float32)/255
left_depth_debug[:,:,0] += datum.left_depth
right_depth_debug = datum.right_img.copy().astype(np.float32)/255
right_depth_debug[:,:,0] += datum.right_depth
lc_depth_debug = datum.nir_img.copy().astype(np.float32)/255
lc_depth_debug[:,:,0] += datum.lc_depth
cv2.imshow("left_depth", left_depth_debug)
cv2.imshow("right_depth", right_depth_debug)
cv2.imshow("lc_depth", lc_depth_debug)
cv2.waitKey(1)

# Compute
start = time.time()
datum.left_feat_int_tensor, datum.left_feat_z_tensor, datum.left_mask_tensor, datum.left_feat_mask_tensor, _ = img_utils.lcsweep_to_rgbsweep(
    sweep_arr=datum.sweep_arr, dmap_large=datum.left_depth, rgb_intr=datum.large_intr, rgb_size=datum.large_size, lc_intr=datum.K_lc, lc_size=datum.lc_size, M_left2LC=datum.M_left2LC)
datum.right_feat_int_tensor, datum.right_feat_z_tensor, datum.right_mask_tensor, datum.right_feat_mask_tensor, _ = img_utils.lcsweep_to_rgbsweep(
    sweep_arr=datum.sweep_arr, dmap_large=datum.right_depth, rgb_intr=datum.large_intr, rgb_size=datum.large_size, lc_intr=datum.K_lc, lc_size=datum.lc_size, M_left2LC=datum.M_right2LC)

feat_int_tensor = datum.left_feat_int_tensor
feat_z_tensor = datum.left_feat_z_tensor
mask_tensor = datum.left_mask_tensor
feat_mask_tensor = datum.left_feat_mask_tensor
rgb_img = datum.left_img
depth_img = torch.tensor(datum.left_depth)
#############
# feat_int_tensor = datum.right_feat_int_tensor
# feat_z_tensor = datum.right_feat_z_tensor
# mask_tensor = datum.right_mask_tensor
# feat_mask_tensor = datum.right_feat_mask_tensor
# rgb_img = datum.right_img
# depth_img = datum.right_depth

# LC Model Test
d_candi = img_utils.powerf(3, 18, 128, 1.0)
mean_scaling, _  = torch.max(feat_int_tensor, dim=0)
mean_intensities, _ = img_utils.lc_intensities_to_dist(d_candi=feat_z_tensor.permute(1,2,0), placement=depth_img.unsqueeze(-1), 
intensity=0, inten_sigma=0.5, noise_sigma=0.1, mean_scaling=mean_scaling.unsqueeze(-1)/255)
mean_intensities = mean_intensities.permute(2,0,1) # 128, 256, 320

# Instead of d_candi, i want to use the true pixel depth is that possible (yes)

# Plotting
import random
from matplotlib import pyplot as plt
def mouse_callback(event,x,y,flags,param):
    global mouseX,mouseY, combined_image, sweep_arr, dmap_large, feat_int_tensor, feat_mask_tensor, mean_intensities, d_candi
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y
        print(mouseX,mouseY)
        rgb = (random.random(), random.random(), random.random())

        # Extract vals
        disp_z = feat_z_tensor[:,y,x]
        disp_i = feat_int_tensor[:,y,x]/255.
        first_nan = np.isnan(disp_z).argmax(axis=0)
        if first_nan:
            disp_z = disp_z[0:first_nan]
            disp_i = disp_i[0:first_nan]

        plt.figure(0)
        plt.plot(disp_z, disp_i, c=rgb, marker='*')

        #plt.figure(1)
        plt.plot(d_candi, mean_intensities[:,y,x], c=rgb, marker='*')

        plt.pause(0.1)

cv2.namedWindow('rgbimg')
cv2.setMouseCallback('rgbimg',mouse_callback)

while 1:
    cv2.imshow("rgbimg", rgb_img/255. + mask_tensor.squeeze(0).unsqueeze(-1).numpy())
    cv2.waitKey(15)


##########################################################################

stop

#dmap_large = torch.tensor(dmap_large)



stop


# Need to figure out how to load some of the networks too

# Load
sweep_arr = np.load("/media/raaj/Storage/sweep_data/2021_03_05/2021_03_05_drive_0004_sweep/sweep/000021.npy").astype(np.float32)
velodata = np.fromfile("/media/raaj/Storage/sweep_data/2021_03_05/2021_03_05_drive_0004_sweep/lidar/000021.bin", dtype=np.float32).reshape((-1, 4))
rgbimg = cv2.imread("/media/raaj/Storage/sweep_data/2021_03_05/2021_03_05_drive_0004_sweep/left_img/000021.png")
nirimg = cv2.imread("/media/raaj/Storage/sweep_data/2021_03_05/2021_03_05_drive_0004_sweep/nir_img/000021.png")
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
feat_int_tensor, feat_z_tensor, mask_tensor, feat_mask_tensor, combined_image = img_utils.lcsweep_to_rgbsweep(
    sweep_arr=sweep_arr, dmap_large=dmap_large, rgb_intr=large_intr, rgb_size=large_size, lc_intr=K_lc, lc_size=lc_size, M_left2LC=M_left2LC)
end = time.time()
print(end-start)

# Testing
dmap_large = torch.tensor(dmap_large)
print(feat_int_tensor.shape)
print(feat_z_tensor.shape)
print(dmap_large.shape)
"""
Objective is give, the dmap, i want to sample from feat_int_tensor
"""

# Gather Operation
# feat_z_tensor_temp = feat_z_tensor.clone()
# feat_z_tensor_temp[torch.isnan(feat_z_tensor)] = 1000
# inds = torch.argmin(torch.abs(dmap_large.reshape(1, 256, 320) - feat_z_tensor_temp), dim=0)  # (320, 240)
# result = torch.gather(feat_int_tensor, 0, inds.unsqueeze(0))


#result = torch.gather(intensity_values, 0, inds.unsqueeze(0))

# # Reduce Tensor Size here itself?
# print(mask_tensor.shape)
# feat_int_tensor = F.interpolate(feat_int_tensor.unsqueeze(0), size=[64, 80], mode='nearest').squeeze(0)
# feat_z_tensor = F.interpolate(feat_z_tensor.unsqueeze(0), size=[64, 80], mode='nearest').squeeze(0)
# mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=[64, 80], mode='nearest').squeeze(0)
# rgbimg = cv2.resize(rgbimg, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

# def loss_test(output, depth, feat_int, feat_mask, d_candi, position=(100,100)):
#     print(output.shape) # [1, 2, 256, 320]
#     print(depth.shape) # [1, 256, 320])
#     print(feat_int.shape) # [1, 128, 256, 320]
#     print(feat_mask.shape) # [1, 128, 256, 320]
#     print(d_candi.shape)
#     x,y = position

#     """
#     mean_intensities, DPV = img_utils.lc_intensities_to_dist(torch.tensor(self.d_candi).float().to(self.device), z_img, int_img, unc_img*0+0.3, 0.1, 0.6)
#     """

#     for i in range(0, output.shape[0]):

#         mean_intensities, DPV = img_utils.lc_intensities_to_dist(
#             d_candi = torch.tensor(d_candi).float(), 
#             placement = depth[i,:,:].unsqueeze(-1), 
#             intensity = 0, 
#             inten_sigma = output[i, 1, :, :].unsqueeze(-1), # Change
#             noise_sigma = 0.1, 
#             mean_scaling = output[i, 0, :, :].unsqueeze(-1)) # Change
#         mean_intensities = mean_intensities.permute(2,0,1) # 128, 256, 320

#         gt = feat_int[i,:,:,:]/255.
#         pred = mean_intensities
#         mask = feat_mask[i,:,:,:].float()
#         error = torch.sum(((gt-pred)**2)*mask)
#         print(error)

#         gt = feat_int[i,:,y,x]/255
#         pred = mean_intensities[:,y,x]


#         #plt.figure(1)
#         plt.plot(d_candi, gt, c=[0,1,1], marker='.')
#         plt.plot(d_candi, pred, c=[0,1,1], marker='.')

#         print(mean_intensities.shape)


import random
from matplotlib import pyplot as plt
def mouse_callback(event,x,y,flags,param):
    global mouseX,mouseY, combined_image, sweep_arr, dmap_large, feat_int_tensor, feat_mask_tensor
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #x = pixel[1]
        #y = pixel[0]
        mouseX,mouseY = x,y
        print(mouseX,mouseY)
        rgb = (random.random(), random.random(), random.random())

        # Extract vals
        disp_z = feat_z_tensor[:,y,x]
        disp_i = feat_int_tensor[:,y,x]/255.
        print(disp_i)
        #print(disp_z)
        first_nan = np.isnan(disp_z).argmax(axis=0)
        if first_nan:
            disp_z = disp_z[0:first_nan]
            disp_i = disp_i[0:first_nan]

        # # Generate d_candi
        # d_candi = img_utils.powerf(3, 18, 128, 1.0)
        # output = torch.ones(1, 2, 256, 320)
        # output[:,0,:,:] = 0.05
        # output[:,1,:,:] = 1.5
        # depth = torch.tensor(dmap_large).unsqueeze(0)
        # loss_test(output, depth, feat_int_tensor.unsqueeze(0), feat_mask_tensor.unsqueeze(0), d_candi, (mouseX, mouseY))

        #plt.figure(2)
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
