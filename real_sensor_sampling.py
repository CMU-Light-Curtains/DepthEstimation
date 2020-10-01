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
import rospy
from external.perception_lib import viewer

sys.path.append("external/lcsim/python")
from sim import LCDevice
from planner import PlannerRT
import pylc_lib as pylc

###########
"""
Put this Stuff into a class that I can call?
"""
###########

class LC:
    def __init__(self, visualize=False):
        self.visualize = visualize

        # Light Curtain Params
        N = 128
        param = dict()
        param["d_candi"] = img_utils.powerf(3., 38., N, 1.)
        param["d_candi_up"] = param["d_candi"]
        param["r_candi"] = param["d_candi"]
        param["r_candi_up"] = param["d_candi"]
        param["expand_A"] = N
        param["expand_B"] = 128
        param['laser_fov'] = 80.
        param['intr_rgb'] = np.array(
            [[177.43524,   0.,      160.     ],
            [  0.,      178.5432 , 128.     ],
            [  0.,        0. ,       1.     ]]
        )
        param['size_rgb'] = [320, 256]
        param['dist_rgb'] = [0., 0., 0., 0., 0.]

        # Real LC (Sim) - Replace with real sensor later
        LC_SCALE = 1. 
        param['laser_timestep'] = 2.5e-5 / LC_SCALE
        param['intr_lc'] = np.array([
            [446.537*LC_SCALE, 0, 262.073*LC_SCALE],
            [0, 446.589*LC_SCALE, 323.383*LC_SCALE],
            [0, 0, 1]
        ])
        param['size_lc'] = [int(512*LC_SCALE), int(640*LC_SCALE)]
        param['dist_lc'] = [-0.033918, 0.027494, -0.001691, -0.001078, 0.000000]
        param['lTc'] = np.linalg.inv(np.array([
                [9.9999867e-01,   1.6338158e-03,   1.2624934e-06,  -1.9989257e-01],
                [-1.6338158e-03,   9.9999747e-01,   1.5454519e-03,   0.0000000e+00],
                [1.2624934e-06,  -1.5454519e-03,   9.9999881e-01,   1.4010292e-02],
                [0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   1.0000000e+00]
        ]))
        param['rTc'] = np.array([
            [ 0.9999845,   0.00197387, -0.00520207,  0.08262175],
            [-0.00208896,  0.9997511 , -0.0222123,  -0.0229844 ],
            [ 0.00515693,  0.02222283,  0.99973977, -0.23375846],
            [ 0.,          0.,          0.,          1.        ]
        ])
        param['cTr'] = np.linalg.inv(param['rTc'])
        self.real_param = copy.deepcopy(param)
        self.real_lc = light_curtain.LightCurtain()
        if not self.real_lc.initialized:
            self.real_lc.init(copy.deepcopy(self.real_param))

        # Planner LC
        LC_SCALE = 0.625 
        param['laser_timestep'] = 2.5e-5 / LC_SCALE
        param['intr_lc'] = np.array([
            [446.537*LC_SCALE, 0, 262.073*LC_SCALE],
            [0, 446.589*LC_SCALE, 323.383*LC_SCALE],
            [0, 0, 1]
        ])
        param['size_lc'] = [int(512*LC_SCALE), int(640*LC_SCALE)]

        # Can we crop the top and bottom?
        TOP_CUT = 72
        BOT_CUT = 72
        param['size_lc'] = [param['size_lc'][0], param['size_lc'][1] - TOP_CUT - BOT_CUT]
        param['intr_lc'][1,2] -=  (TOP_CUT/2 + BOT_CUT/2)

        # Initialize
        self.algo_param = copy.deepcopy(param)
        self.algo_lc = light_curtain.LightCurtain()
        if not self.algo_lc.initialized:
            self.algo_lc.init(copy.deepcopy(self.algo_param))

        # Viz
        self.viz = None
        if self.visualize:
            self.viz = viewer.Visualizer("V")
            self.viz.start()

    def get_rgb_size(self):
        return (self.real_param['size_rgb'][0], self.real_param['size_rgb'][1])

    def set_sim_cam(self, depth_r, rgb_r, pool_val=4):
        self.depth_r = depth_r
        self.rgb_r = rgb_r

        # Warp Depth Image to LC
        pts_rgb = img_utils.depth_to_pts(torch.Tensor(self.depth_r).unsqueeze(0), self.real_param['intr_rgb'])
        pts_rgb = pts_rgb.reshape((pts_rgb.shape[0], pts_rgb.shape[1] * pts_rgb.shape[2]))
        pts_rgb = torch.cat([pts_rgb, torch.ones(1, pts_rgb.shape[1])])
        pts_rgb = pts_rgb.numpy().T
        thick_rgb = np.ones((pts_rgb.shape[0], 1)).astype(np.float32)
        uniform_params = {"filtering": 2}
        depth_lc, _, _ = pylc.transformPoints(pts_rgb, thick_rgb, self.real_param['intr_lc'], self.real_param['cTr'],
                                            self.real_param['size_lc'][0], self.real_param['size_lc'][1],
                                            uniform_params)
        
        depth_lc = img_utils.minpool(torch.Tensor(depth_lc).unsqueeze(0).unsqueeze(0), pool_val, 1000).squeeze(0).squeeze(0).numpy()
        depth_lc = cv2.resize(depth_lc, (0,0), fx=pool_val, fy=pool_val, interpolation = cv2.INTER_NEAREST)
        self.depth_lc = depth_lc

    def process_sim(self):
        # Convert to Tensor
        self.rgb_r_tensor = (torch.tensor(self.rgb_r).permute(2,0,1)[0:3,:,:])[[2,1,0],:,:].cuda()
        self.depth_r_tensor = torch.tensor(self.depth_r).unsqueeze(0).cuda()
        self.intr_r_tensor = torch.tensor(self.real_param['intr_rgb']).cuda()

        # Make a DPV out of the Depth Map
        mask_truth = (self.depth_r_tensor > 0.).float()
        dpv_truth = img_utils.gen_dpv_withmask(self.depth_r_tensor, mask_truth.unsqueeze(0), self.algo_lc.d_candi, 0.3)
        depth_truth = img_utils.dpv_to_depthmap(dpv_truth, self.algo_lc.d_candi, BV_log=False)
        print(depth_truth.shape, depth_truth.dtype)
        cloud_truth = img_utils.tocloud(depth_truth, self.rgb_r_tensor, self.intr_r_tensor)
        depth_x = depth_truth.squeeze(0).cpu().numpy()
        self.depth_truth = depth_truth
        if self.visualize: 
            if self.viz is not None:
                self.viz.addCloud(cloud_truth, 1)
                self.viz.swapBuffer()

        # Make Unc Field Truth (RGB)
        self.unc_field_truth_r, debugmap = img_utils.gen_ufield(dpv_truth, self.algo_lc.d_candi, self.intr_r_tensor.squeeze(0), BV_log=False, cfg=None)

        # Viz Debugmap
        rgb_temp = rgb_r.copy()
        rgb_temp[:,:,0] += debugmap.squeeze(0).cpu().numpy()

        # Unc Field Truth in LC (LC)
        self.algo_lc.fw_large.load_flowfield()
        self.unc_field_truth_lc = self.algo_lc.fw_large.preprocess(self.unc_field_truth_r.squeeze(0), self.algo_lc.d_candi, self.algo_lc.d_candi_up)
        self.unc_field_truth_lc = self.algo_lc.fw_large.transformZTheta(self.unc_field_truth_lc, self.algo_lc.d_candi_up, self.algo_lc.d_candi_up, "transform_" + "large").unsqueeze(0)

        # Viz
        if self.visualize:
            cv2.imshow("debugmap_truth", rgb_temp)
            cv2.imshow("unc_field_truth_rgb", self.unc_field_truth_r.squeeze(0).cpu().numpy())
            cv2.imshow("unc_field_truth_lc", self.unc_field_truth_lc.squeeze(0).cpu().numpy())
    
    def init_unc_field(self):
        init_depth = torch.zeros((1, self.real_param["size_rgb"][1], self.real_param["size_rgb"][0])).cuda() + 20
        self.final = torch.log(img_utils.gen_dpv_withmask(init_depth, init_depth.unsqueeze(0)*0+1, self.algo_lc.d_candi, 6.0))

    def iterate(self, iterations=100):
        # Iterate
        mode = "high"
        planner = "default"
        params = {"step": [0.25, 0.5, 0.75]}
        #planner = "m1"
        #params = {"step": 5}
        for i in range(0, iterations):
            print(i)

            # Generate UField (in RGB)
            start = time.time()
            unc_field_predicted, _ = img_utils.gen_ufield(self.final, self.algo_lc.d_candi, self.intr_r_tensor.squeeze(0), BV_log=True, cfg=None)
            time_gen_ufield = time.time() - start

            # Score (Need to compute in LC space as it is zoomed in sadly)
            start = time.time()
            unc_field_predicted_lc = self.algo_lc.fw_large.preprocess(unc_field_predicted.squeeze(0), self.algo_lc.d_candi, self.algo_lc.d_candi_up)
            unc_field_predicted_lc = self.algo_lc.fw_large.transformZTheta(unc_field_predicted_lc, self.algo_lc.d_candi_up, self.algo_lc.d_candi_up, "transform_" + "large").unsqueeze(0)
            unc_score = img_utils.compute_unc_rmse(self.unc_field_truth_lc, unc_field_predicted_lc, self.algo_lc.d_candi)
            time_score = time.time() - start
            print(unc_score)
            #cv2.imshow("FAGA", self.unc_field_truth_r.squeeze(0).cpu().numpy())
            #cv2.imshow("FAGB", unc_field_predicted.squeeze(0).cpu().numpy())
            
            # Plan (LC Path in LC) (But Field Visual is in LC -> Needs to be in RGB)
            start = time.time()
            if mode == "high":
                if planner == "default":
                    lc_paths, field_visual = self.algo_lc.plan_default_high(unc_field_predicted.squeeze(0), params)
                elif planner == "m1":
                    lc_paths, field_visual = self.algo_lc.plan_m1_high(unc_field_predicted.squeeze(0), params)
                elif planner == "sweep":
                    lc_paths, field_visual = self.algo_lc.plan_sweep_high(unc_field_predicted.squeeze(0), params)
            elif mode == "low":
                if planner == "default":
                    lc_paths, field_visual = self.algo_lc.plan_default_low(unc_field_predicted.squeeze(0), params)
                elif planner == "m1":
                    lc_paths, field_visual = self.algo_lc.plan_m1_low(unc_field_predicted.squeeze(0), params)
            time_plan = time.time() - start

            # Viz
            if self.visualize:
                field_visual[:,:,2] = self.unc_field_truth_lc[0,:,:].cpu().numpy()*3
                cv2.imshow("field_visual", field_visual)
                #cv2.imshow("final_depth", final_depth.squeeze(0).cpu().numpy()/100)
                
            # 3D
            if self.visualize:
                # Clean
                z = torch.exp(self.final.squeeze(0))
                d_candi_expanded = torch.tensor(self.algo_lc.d_candi).unsqueeze(1).unsqueeze(1).cuda()
                mean = torch.sum(d_candi_expanded * z, dim=0)
                variance = torch.sum(((d_candi_expanded - mean) ** 2) * z, dim=0)
                mask_var = (variance < 1.0).float()
                final_temp = self.final * mask_var.unsqueeze(0)

                if self.viz is not None:
                    cloud_truth = img_utils.tocloud(self.depth_truth, self.rgb_r_tensor, self.intr_r_tensor, rgbr=[255,255,0])
                    self.viz.addCloud(cloud_truth, 1)
                    self.viz.addCloud(img_utils.tocloud(img_utils.dpv_to_depthmap(final_temp, self.algo_lc.d_candi, BV_log=True), self.rgb_r_tensor, self.intr_r_tensor), 3)
                    self.viz.swapBuffer()
                cv2.waitKey(0)

            # In Sensing Real
            lc_DPVs = []
            start = time.time()
            for lc_path in lc_paths:
                if mode == "high":
                    lc_DPV, _, _ = self.real_lc.sense_real(self.depth_lc, lc_path, False)
                    lc_DPVs.append(lc_DPV)
            time_sense = time.time() - start

            # Keep Renormalize
            curr_dist = torch.clamp(torch.exp(self.final), img_utils.epsilon, 1.)

            # # Update
            # curr_dist_log = torch.log(curr_dist)
            # for lcdpv in lc_DPVs:
            #     lcdpv = torch.clamp(lcdpv, img_utils.epsilon, 1.)
            #     curr_dist_log += torch.log(lcdpv)
            # curr_dist = torch.exp(curr_dist_log)
            # curr_dist = curr_dist / torch.sum(curr_dist, dim=1).unsqueeze(1)

            # Update
            start = time.time()
            for lcdpv in lc_DPVs:
                lcdpv = torch.clamp(lcdpv, img_utils.epsilon, 1.)
                curr_dist = curr_dist * lcdpv
                curr_dist = curr_dist / torch.sum(curr_dist, dim=1).unsqueeze(1)
            time_update = time.time() - start

            # #curr_dist = torch.exp(curr_dist_log)
            # curr_dist = curr_dist / torch.sum(curr_dist, dim=1).unsqueeze(1)
            # unc_field_lcdpv, _ = img_utils.gen_ufield(curr_dist, lc.d_candi, intr.squeeze(0), BV_log=False, cfg=self.cfg)
            # _, unc_field_lcdpv = lc.plan_empty_high(unc_field_lcdpv.squeeze(0), {})
            # cv2.imshow("faggot2", unc_field_lcdpv)
            # cv2.waitKey(0)

            # Spread
            start = time.time()
            for i in range(0, 1):
                curr_dist = img_utils.spread_dpv_hack(curr_dist, 5)
            time_spread = time.time() - start

            # Keep Renormalize
            curr_dist = torch.clamp(curr_dist, img_utils.epsilon, 1.)

            # Back to Log space
            self.final = torch.log(curr_dist)

            print("time_gen_ufield: " + str(time_gen_ufield))
            print("time_score: " + str(time_score))
            print("time_plan: " + str(time_plan))
            print("time_sense: " + str(time_sense))
            print("time_update: " + str(time_update))
            print("time_spread: " + str(time_spread))
            print(torch.cuda.memory_allocated(0))

# Create LC Object
lc = LC(visualize=False)

# Load some Sim Depth
depth_r = np.load('/home/raaj/adapfusion/external/lcsim/python/example/depth_image.npy')
rgb_r = np.load('/home/raaj/adapfusion/external/lcsim/python/example/rgb_image.npy').astype(np.float32)/255.
items = [[250, 300, 50, 50, 10+5], [250, 200, 50, 50, 20.], [250, 100, 50, 50, 10.]]
for item1 in items:
    depth_r[item1[0]:item1[0] + item1[2], item1[1]:item1[1] + item1[3]] = item1[4]
    rgb_r[item1[0]:item1[0] + item1[2], item1[1]:item1[1] + item1[3], 0:2] = 0.1
depth_r = cv2.resize(depth_r, lc.get_rgb_size(), interpolation = cv2.INTER_LINEAR)
rgb_r = cv2.resize(rgb_r, lc.get_rgb_size(), interpolation = cv2.INTER_LINEAR)
# cv2.imshow("depth_r", depth_r/100.)
# cv2.imshow("rgb_r", rgb_r)

# Set it
lc.set_sim_cam(depth_r, rgb_r)
# cv2.imshow("depth_lc", lc.depth_lc/100.)

# Process Sim
lc.process_sim()

# Initialize the field
lc.init_unc_field()

# Iterate
lc.iterate(100)