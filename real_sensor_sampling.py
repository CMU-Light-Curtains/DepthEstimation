import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import utils.img_utils as img_utils
import cv2
import matplotlib.pyplot as plt
import time
from lc import light_curtain
import copy
import sys

sys.path.append("external/lcsim/python")
from sim import LCDevice
from planner import PlannerRT
import pylc_lib as pylc

# Light Curtain Params
N = 128
dict = dict()
dict["d_candi"] = img_utils.powerf(3., 38., N, 1.)
dict["d_candi_up"] = dict["d_candi"]
dict["r_candi"] = dict["d_candi"]
dict["r_candi_up"] = dict["d_candi"]
dict["expand_A"] = N
dict["expand_B"] = 128
dict['laser_fov'] = 80.
dict['intr_rgb'] = np.array(
    [[177.43524,   0.,      160.     ],
    [  0.,      178.5432 , 128.     ],
    [  0.,        0. ,       1.     ]]
)
dict['size_rgb'] = [320, 256]
dict['dist_rgb'] = [0., 0., 0., 0., 0.]

# Real LC (Sim) - Replace with real sensor later
LC_SCALE = 1. 
dict['laser_timestep'] = 2.5e-5 / LC_SCALE
dict['intr_lc'] = np.array([
    [446.537*LC_SCALE, 0, 262.073*LC_SCALE],
    [0, 446.589*LC_SCALE, 323.383*LC_SCALE],
    [0, 0, 1]
])
dict['size_lc'] = [int(512*LC_SCALE), int(640*LC_SCALE)]
dict['dist_lc'] = [-0.033918, 0.027494, -0.001691, -0.001078, 0.000000]
dict['lTc'] = np.linalg.inv(np.array([
        [9.9999867e-01,   1.6338158e-03,   1.2624934e-06,  -1.9989257e-01],
        [-1.6338158e-03,   9.9999747e-01,   1.5454519e-03,   0.0000000e+00],
        [1.2624934e-06,  -1.5454519e-03,   9.9999881e-01,   1.4010292e-02],
        [0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   1.0000000e+00]
]))
dict['rTc'] = np.array([
    [ 0.9999845,   0.00197387, -0.00520207,  0.08262175],
    [-0.00208896,  0.9997511 , -0.0222123,  -0.0229844 ],
    [ 0.00515693,  0.02222283,  0.99973977, -0.23375846],
    [ 0.,          0.,          0.,          1.        ]
])
dict['cTr'] = np.linalg.inv(dict['rTc'])
real_lc = light_curtain.LightCurtain()
if not real_lc.initialized:
    real_lc.init(copy.deepcopy(dict))

# Imagine we have this from the sensor
depth_r = np.load('/home/raaj/adapfusion/external/lcsim/python/example/depth_image.npy')
rgb_r = np.load('/home/raaj/adapfusion/external/lcsim/python/example/rgb_image.npy').astype(np.float32)/255.
items = [[250, 300, 50, 50, 10+5], [250, 200, 50, 50, 20.], [250, 100, 50, 50, 10.]]
for item1 in items:
    depth_r[item1[0]:item1[0] + item1[2], item1[1]:item1[1] + item1[3]] = item1[4]
    rgb_r[item1[0]:item1[0] + item1[2], item1[1]:item1[1] + item1[3], 0:2] = 0.1
depth_r = cv2.resize(depth_r, (dict['size_rgb'][0],dict['size_rgb'][1]), interpolation = cv2.INTER_LINEAR)
rgb_r = cv2.resize(rgb_r, (dict['size_rgb'][0],dict['size_rgb'][1]), interpolation = cv2.INTER_LINEAR)
cv2.imshow("depth_r", depth_r/100.)
cv2.imshow("rgb_r", rgb_r)

# Warp Depth Image to LC
pts_rgb = img_utils.depth_to_pts(torch.Tensor(depth_r).unsqueeze(0), dict['intr_rgb'])
pts_rgb = pts_rgb.reshape((pts_rgb.shape[0], pts_rgb.shape[1] * pts_rgb.shape[2]))
pts_rgb = torch.cat([pts_rgb, torch.ones(1, pts_rgb.shape[1])])
pts_rgb = pts_rgb.numpy().T
thick_rgb = np.ones((pts_rgb.shape[0], 1)).astype(np.float32)
uniform_params = {"filtering": 2}
depth_lc, _, _ = pylc.transformPoints(pts_rgb, thick_rgb, dict['intr_lc'], dict['cTr'],
                                      dict['size_lc'][0], dict['size_lc'][1],
                                      uniform_params)

# Pool Depth
pool_val = 4
depth_lc = img_utils.minpool(torch.Tensor(depth_lc).unsqueeze(0).unsqueeze(0), pool_val, 1000).squeeze(0).squeeze(0).numpy()
depth_lc = cv2.resize(depth_lc, (0,0), fx=pool_val, fy=pool_val, interpolation = cv2.INTER_NEAREST)
cv2.imshow("depth_lc", depth_lc/100.)
print(depth_lc.shape)
print(rgb_r.shape)
#cv2.waitKey(0)

# # Resize 
# depth_image = cv2.resize(depth_image, (dict['size_lc'][0],dict['size_lc'][1]), interpolation = cv2.INTER_LINEAR)
# rgb_image = cv2.resize(rgb_image, (dict['size_lc'][0],dict['size_lc'][1]), interpolation = cv2.INTER_LINEAR)
# intr_orig = dict['intr_lc']
# cv2.circle(rgb_image,(int(dict['intr_lc'][0,2]), int(dict['intr_lc'][1,2])), 3, (255,255,0), -1)
# cv2.imshow("depth_image", depth_image/100.)
# # cv2.imshow("rgb_image", rgb_image[:,:,0:3])
# # cv2.waitKey(0)

# Planner LC
LC_SCALE = 0.625 
dict['laser_timestep'] = 2.5e-5 / LC_SCALE
dict['intr_lc'] = np.array([
    [446.537*LC_SCALE, 0, 262.073*LC_SCALE],
    [0, 446.589*LC_SCALE, 323.383*LC_SCALE],
    [0, 0, 1]
])
dict['size_lc'] = [int(512*LC_SCALE), int(640*LC_SCALE)]
dict['dist_lc'] = [-0.033918, 0.027494, -0.001691, -0.001078, 0.000000]
dict['lTc'] = np.linalg.inv(np.array([
        [9.9999867e-01,   1.6338158e-03,   1.2624934e-06,  -1.9989257e-01],
        [-1.6338158e-03,   9.9999747e-01,   1.5454519e-03,   0.0000000e+00],
        [1.2624934e-06,  -1.5454519e-03,   9.9999881e-01,   1.4010292e-02],
        [0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   1.0000000e+00]
]))
dict['rTc'] = np.array([
    [ 0.9999845,   0.00197387, -0.00520207,  0.08262175],
    [-0.00208896,  0.9997511 , -0.0222123,  -0.0229844 ],
    [ 0.00515693,  0.02222283,  0.99973977, -0.23375846],
    [ 0.,          0.,          0.,          1.        ]
])
print(dict['size_lc'])

# Can we crop the top and bottom?
TOP_CUT = 72
BOT_CUT = 72
dict['size_lc'] = [dict['size_lc'][0], dict['size_lc'][1] - TOP_CUT - BOT_CUT]
dict['intr_lc'][1,2] -=  (TOP_CUT/2 + BOT_CUT/2)

# Resize/Rescale the images
# stop
# depth_image_algo = depth_image_algo[TOP_CUT:depth_image_algo.shape[0],:]
# depth_image_algo = depth_image_algo[0:depth_image_algo.shape[0]-BOT_CUT,:]
# rgb_image_algo = rgb_image_algo[TOP_CUT:rgb_image_algo.shape[0],:,:]
# rgb_image_algo = rgb_image_algo[0:rgb_image_algo.shape[0]-BOT_CUT,:,:]
# cv2.circle(rgb_image_algo,(int(dict['intr_lc'][0,2]), int(dict['intr_lc'][1,2])), 3, (255,0,0), -1)
# cv2.imshow("rgb_image_algo", rgb_image_algo[:,:,0:3])
# cv2.waitKey(0)

# Initialize
algo_lc = light_curtain.LightCurtain()
if not algo_lc.initialized:
    algo_lc.init(copy.deepcopy(dict))

# Convert to Tensor
rgb_r_tensor = (torch.tensor(rgb_r).permute(2,0,1)[0:3,:,:])[[2,1,0],:,:].cuda()
depth_r_tensor = torch.tensor(depth_r).unsqueeze(0).cuda()
intr_r_tensor = torch.tensor(dict['intr_rgb']).cuda()

# Viz
viz = None
from external.perception_lib import viewer
viz = viewer.Visualizer("V")
viz.start()

# # Add Cloud
# truth_cloud = img_utils.tocloud(depth_r_tensor, rgb_r_tensor, intr_r_tensor)
# viz.addCloud(truth_cloud, 1)
# viz.swapBuffer()
# cv2.waitKey(0)

# # Make a DPV out of the Depth Map
# mask_real = (depth_r_tensor > 0.).float()
# dpv_real = img_utils.gen_dpv_withmask(depth_r_tensor, mask_real.unsqueeze(0), algo_lc.d_candi, 0.3)
# depth_real = img_utils.dpv_to_depthmap(dpv_real, algo_lc.d_candi, BV_log=False)
# cloud_truth = img_utils.tocloud(depth_truth, rgb_algo_tensor, intr_algo_tensor)

# Make a DPV out of the Depth Map
mask_truth = (depth_r_tensor > 0.).float()
dpv_truth = img_utils.gen_dpv_withmask(depth_r_tensor, mask_truth.unsqueeze(0), algo_lc.d_candi, 0.3)
depth_truth = img_utils.dpv_to_depthmap(dpv_truth, algo_lc.d_candi, BV_log=False)
cloud_truth = img_utils.tocloud(depth_truth, rgb_r_tensor, intr_r_tensor)
depth_x = depth_truth.squeeze(0).cpu().numpy()
if viz is not None: viz.addCloud(cloud_truth, 1)
viz.swapBuffer()
#cv2.waitKey(0)

# Make Unc Field Truth (RGB)
unc_field_truth, debugmap = img_utils.gen_ufield(dpv_truth, algo_lc.d_candi, intr_r_tensor.squeeze(0), BV_log=False, cfg=None)

# Viz Debugmap
rgb_temp = rgb_r.copy()
rgb_temp[:,:,0] += debugmap.squeeze(0).cpu().numpy()
cv2.imshow("debugmap_truth", rgb_temp)
cv2.imshow("unc_field_truth_rgb", unc_field_truth.squeeze(0).cpu().numpy())
#cv2.waitKey(0)

# Unc Field Truth in LC (LC)
unc_field_truth = algo_lc.fw_large.preprocess(unc_field_truth.squeeze(0), algo_lc.d_candi, algo_lc.d_candi_up)
unc_field_truth = algo_lc.fw_large.transformZTheta(unc_field_truth, algo_lc.d_candi_up, algo_lc.d_candi_up, "transform_" + "large").unsqueeze(0)
cv2.imshow("unc_field_truth_lc", unc_field_truth.squeeze(0).cpu().numpy())
cv2.waitKey(0)

# Iterate
visualize = True
mode = "high"
planner = "default"
params = {"step": [0.25, 0.5, 0.75]}
#planner = "m1"
#params = {"step": 5}
iterations = 100
final = torch.log(img_utils.gen_dpv_withmask(depth_truth*0+20, mask_truth.unsqueeze(0)*0+1, algo_lc.d_candi, 6.0))
lc_DPVs = []
for i in range(0, iterations):

    # Generate UField (in RGB)
    unc_field_predicted, _ = img_utils.gen_ufield(final, algo_lc.d_candi, intr_r_tensor.squeeze(0), BV_log=True, cfg=None)

    # Plan (LC Path in LC) (But Field Visual is in LC -> Needs to be in RGB)
    if mode == "high":
        if planner == "default":
            lc_paths, field_visual = algo_lc.plan_default_high(unc_field_predicted.squeeze(0), params)
        elif planner == "m1":
            lc_paths, field_visual = algo_lc.plan_m1_high(unc_field_predicted.squeeze(0), params)
        elif planner == "sweep":
            lc_paths, field_visual = algo_lc.plan_sweep_high(unc_field_predicted.squeeze(0), params)
    elif mode == "low":
        if planner == "default":
            lc_paths, field_visual = algo_lc.plan_default_low(unc_field_predicted.squeeze(0), params)
        elif planner == "m1":
            lc_paths, field_visual = algo_lc.plan_m1_low(unc_field_predicted.squeeze(0), params)

    # # Viz
    if visualize:
        field_visual[:,:,2] = unc_field_truth[0,:,:].cpu().numpy()*3
        cv2.imshow("field_visual", field_visual)
        #cv2.imshow("final_depth", final_depth.squeeze(0).cpu().numpy()/100)
        
    # 3D
    if visualize:
        # Clean
        z = torch.exp(final.squeeze(0))
        d_candi_expanded = torch.tensor(algo_lc.d_candi).unsqueeze(1).unsqueeze(1).cuda()
        mean = torch.sum(d_candi_expanded * z, dim=0)
        variance = torch.sum(((d_candi_expanded - mean) ** 2) * z, dim=0)
        mask_var = (variance < 1.0).float()
        final_temp = final * mask_var.unsqueeze(0)

        if viz is not None:
            cloud_truth = img_utils.tocloud(depth_truth, rgb_r_tensor, intr_r_tensor, rgbr=[255,255,0])
            viz.addCloud(cloud_truth, 1)
            viz.addCloud(img_utils.tocloud(img_utils.dpv_to_depthmap(final_temp, algo_lc.d_candi, BV_log=True), rgb_r_tensor, intr_r_tensor), 3)
            viz.swapBuffer()
        cv2.waitKey(0)

    # # Sensing Fake (in RGB frame)
    # lc_DPVs = []
    # for lc_path in lc_paths:
    #     if mode == "high":
    #         lc_DPV, _, _ = real_lc.sense_high(depth_lc, lc_path, True)
    #         lc_DPVs.append(lc_DPV)

    # In Sensing Real
    for lc_path in lc_paths:
        if mode == "high":
            lc_DPV, _, _ = real_lc.sense_real(depth_lc, lc_path, True)
            lc_DPVs.append(lc_DPV)

    """
    lc_paths consist of 320,2 in LC frame
    Send these points to the LC device and get back a [640x512 image of 4 channels]
    Perform scaling of the tensor down, and then cropping 
    Transform it to the RGB frame and do other lc_dpv generation
    """

    # Keep Renormalize
    curr_dist = torch.clamp(torch.exp(final), img_utils.epsilon, 1.)

    # # Update
    # curr_dist_log = torch.log(curr_dist)
    # for lcdpv in lc_DPVs:
    #     lcdpv = torch.clamp(lcdpv, img_utils.epsilon, 1.)
    #     curr_dist_log += torch.log(lcdpv)
    # curr_dist = torch.exp(curr_dist_log)
    # curr_dist = curr_dist / torch.sum(curr_dist, dim=1).unsqueeze(1)

    # Update
    for lcdpv in lc_DPVs:
        lcdpv = torch.clamp(lcdpv, img_utils.epsilon, 1.)
        curr_dist = curr_dist * lcdpv
        curr_dist = curr_dist / torch.sum(curr_dist, dim=1).unsqueeze(1)

    # #curr_dist = torch.exp(curr_dist_log)
    # curr_dist = curr_dist / torch.sum(curr_dist, dim=1).unsqueeze(1)
    # unc_field_lcdpv, _ = img_utils.gen_ufield(curr_dist, lc.d_candi, intr.squeeze(0), BV_log=False, cfg=self.cfg)
    # _, unc_field_lcdpv = lc.plan_empty_high(unc_field_lcdpv.squeeze(0), {})
    # cv2.imshow("faggot2", unc_field_lcdpv)
    # cv2.waitKey(0)

    # Spread
    for i in range(0, 1):
        curr_dist = img_utils.spread_dpv_hack(curr_dist, 5)

    # Keep Renormalize
    curr_dist = torch.clamp(curr_dist, img_utils.epsilon, 1.)

    # Back to Log space
    final = torch.log(curr_dist)

if viz is not None: viz.swapBuffer()
cv2.waitKey(0)
