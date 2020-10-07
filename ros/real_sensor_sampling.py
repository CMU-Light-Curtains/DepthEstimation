import sys
import os
sys.path.append(os.getcwd())

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

import cv2
import torch
import numpy as np
import pickle
import os
import rospy
import rospkg
import sys
import json
import tf
import copy
import multiprocessing
import time
import shutil

import threading
from collections import deque
import functools
import message_filters
import os
import rospy
from message_filters import ApproximateTimeSynchronizer
import sensor_msgs.msg
import sensor_msgs.srv
from tf.transformations import quaternion_matrix
import image_geometry
import cv_bridge
from cv_bridge import CvBridge
devel_folder = rospkg.RosPack().get_path('params_lib').split("/")
devel_folder = '/'.join(devel_folder[:len(devel_folder)-2]) + "/devel/lib/"
sys.path.append(devel_folder)
import params_lib_python

import rospkg
devel_folder = rospkg.RosPack().get_path('lc_wrapper').split("/")
devel_folder = '/'.join(devel_folder[:len(devel_folder)-3]) + "/devel/lib/"
print(devel_folder)
sys.path.append(devel_folder)
import lc_wrapper_python

class LC:
    def __init__(self, params_file, visualize=False, real_sensor=False):
        self.visualize = visualize

        # Real Device
        self.lc_wrapper = None
        if real_sensor:
            lc_wrapper_python.ros_init("lc_wrapper_example")
            self.lc_wrapper = lc_wrapper_python.LightCurtainWrapper(laser_power=30, dev="/dev/ttyACM0")

        # Light Curtain Params
        with open(params_file) as f:
            param = json.load(f)

        # Precompute additional param
        param["intr_rgb"] = np.array(param["intr_rgb"]).astype(np.float32)
        param["intr_lc"] = np.array(param["intr_lc"]).astype(np.float32)
        param["lTc"] = np.array(param["lTc"]).astype(np.float32)
        param["rTc"] = np.array(param["rTc"]).astype(np.float32)
        N = param["N"]
        self.S_RANGE = param["s_range"]
        self.E_RANGE = param["e_range"]
        param["d_candi"] = img_utils.powerf(self.S_RANGE, self.E_RANGE, N, 1.)
        param["d_candi_up"] = param["d_candi"]
        param["r_candi"] = param["d_candi"]
        param["r_candi_up"] = param["d_candi"]
        param['cTr'] = np.linalg.inv(param['rTc'])
        self.real_param = copy.deepcopy(param)
        self.real_lc = light_curtain.LightCurtain()
        if not self.real_lc.initialized:
            self.real_lc.init(copy.deepcopy(self.real_param))

        # Planner LC
        LC_SCALE = float(param['size_rgb'][0]) / float(param['size_lc'][0]) # 0.625
        param['laser_timestep'] = 2.5e-5 / LC_SCALE
        param['intr_lc'] = np.array([
            [param['intr_lc'][0,0]*LC_SCALE, 0, param['intr_lc'][0,2]*LC_SCALE],
            [0, param['intr_lc'][1,1]*LC_SCALE, param['intr_lc'][1,2]*LC_SCALE],
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
        depth_truth = img_utils.dpv_to_depthmap(dpv_truth, self.algo_lc.d_candi, BV_log=False) * mask_truth
        print(depth_truth.shape, depth_truth.dtype)
        cloud_truth = img_utils.tocloud(depth_truth, self.rgb_r_tensor, self.intr_r_tensor)
        depth_x = depth_truth.squeeze(0).cpu().numpy()
        self.depth_truth = depth_truth
        if self.visualize: 
            if self.viz is not None:
                self.viz.addCloud(cloud_truth, 1)
                self.viz.swapBuffer()

        # Make Unc Field Truth (RGB)
        self.unc_field_truth_r, debugmap = img_utils.gen_ufield(dpv_truth, self.algo_lc.d_candi, self.intr_r_tensor.squeeze(0), BV_log=False, cfgx=self.real_param)

        # Viz Debugmap
        rgb_temp = self.rgb_r.copy()
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
        init_depth = torch.zeros((1, self.real_param["size_rgb"][1], self.real_param["size_rgb"][0])).cuda() + self.E_RANGE / 2.
        self.final = torch.log(img_utils.gen_dpv_withmask(init_depth, init_depth.unsqueeze(0)*0+1, self.algo_lc.d_candi, 0.5))

    def disp_final_cloud(self):
        # Start
        if self.viz == None:
            self.viz = viewer.Visualizer("V")
            self.viz.start()

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

    def iterate(self, iterations=100):
        # Iterate
        mode = "high"
        planner = "default"
        params = {"step": [0.5]}
        #planner = "m1"
        #params = {"step": 3, "interval": 5}
        #planner = "sweep"
        #params = {"start": self.S_RANGE + 1, "end": self.E_RANGE - 1, "step": 0.25}
        for i in range(0, iterations):
            print(i)

            # Generate UField (in RGB)
            start = time.time()
            unc_field_predicted, _ = img_utils.gen_ufield(self.final, self.algo_lc.d_candi, self.intr_r_tensor.squeeze(0), BV_log=True, cfgx=self.real_param)
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
            field_visual[:,:,2] = self.unc_field_truth_lc[0,:,:].cpu().numpy()*3
            if self.visualize:
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
                    lc_DPV, _, _ = self.real_lc.sense_real(self.depth_lc, lc_path, self.lc_wrapper)
                    lc_DPVs.append(lc_DPV)
            time_sense = time.time() - start

            # # Other
            # start = time.time()
            # lc_DPVs = self.real_lc.sense_real_batch(self.depth_lc, lc_paths)
            # time_sense = time.time() - start

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
            # unc_field_lcdpv, _ = img_utils.gen_ufield(curr_dist, lc.d_candi, intr.squeeze(0), BV_log=False, cfgx=self.real_param)
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
            self.field_visual = field_visual

            print("time_gen_ufield: " + str(time_gen_ufield))
            print("time_score: " + str(time_score))
            print("time_plan: " + str(time_plan))
            print("time_sense: " + str(time_sense))
            print("time_update: " + str(time_update))
            print("time_spread: " + str(time_spread))
            print(torch.cuda.memory_allocated(0))

class ConsumerThread(threading.Thread):
    def __init__(self, queue, function):
        threading.Thread.__init__(self)
        self.queue = queue
        self.function = function

    def run(self):
        while True:
            # wait for an image (could happen at the very beginning when the queue is still empty)
            while len(self.queue) == 0:
                time.sleep(0.1)
            self.function(self.queue[0])

class Ros():
    def __init__(self, params_file, visualize, real_sensor):
        self.transforms = None
        self.index = 0
        self.prev_index = -1
        self.bridge = CvBridge()
        self.prev_seq = None
        self.just_started = True

        self.q_msg = deque([], 1)
        lth = ConsumerThread(self.q_msg, self.handle_msg)
        lth.setDaemon(True)
        lth.start()

        self.queue_size = 5
        self.sync = functools.partial(ApproximateTimeSynchronizer, slop=0.01)
        self.left_camsub = message_filters.Subscriber('/left_camera_resized/image_color_rect', sensor_msgs.msg.Image)
        self.right_camsub = message_filters.Subscriber('right_camera_resized/image_color_rect', sensor_msgs.msg.Image)
        self.depthsub = message_filters.Subscriber('/left_camera_resized/depth', sensor_msgs.msg.Image)
        self.ts = self.sync([self.left_camsub, self.right_camsub, self.depthsub], self.queue_size)
        self.ts.registerCallback(self.callback)

        self.lc = LC(params_file=params_file, visualize=visualize, real_sensor=real_sensor)

    def destroy(self):
        self.left_camsub.unregister()
        self.right_camsub.unregister()
        self.depthsub.unregister()
        del self.ts

    def get_transform(self, from_frame, to_frame):
        listener = tf.TransformListener()
        listener.waitForTransform(to_frame, from_frame, rospy.Time(), rospy.Duration(4.0))
        (trans,rot) = listener.lookupTransform(to_frame, from_frame, rospy.Time(0))
        matrix = quaternion_matrix(rot)
        matrix[0:3, 3] = trans
        return matrix.astype(np.float32)

    def convert_imgmsg(self, cammsg, caminfomsg):
        cvImg = self.bridge.imgmsg_to_cv2(cammsg, desired_encoding='passthrough')
        cam_model = image_geometry.PinholeCameraModel()
        cam_model.fromCameraInfo(caminfomsg)
        cam_model.rectifyImage(cvImg, 0)
        cvImg = cv2.remap(cvImg, cam_model.mapx, cam_model.mapy, 1)
        return cvImg, cam_model

    def callback(self, left_cammsg, right_cammsg, depth_cammsg):
        # Append msg
        self.q_msg.append((left_cammsg, right_cammsg, depth_cammsg))

        # Index
        self.index += 1

    def handle_msg(self, msg):
        if self.prev_index == self.index:
            time.sleep(0.00001)
            return
        self.prev_index = self.index
        left_cammsg, right_cammsg, depth_cammsg = msg
        print("enter")

        # Convert
        left_img = self.bridge.imgmsg_to_cv2(left_cammsg, desired_encoding='passthrough').astype(np.float32)/255.
        right_img = self.bridge.imgmsg_to_cv2(right_cammsg, desired_encoding='passthrough').astype(np.float32)/255.
        depth_img = self.bridge.imgmsg_to_cv2(depth_cammsg, desired_encoding='passthrough').astype(np.float32)/1000.

        # # Viz
        # depth_disp = np.repeat(depth_img[:, :, np.newaxis], 3, axis=2)
        # stack = np.vstack((depth_disp,left_img,right_img))
        # cv2.imshow("win", stack)
        # cv2.waitKey(1)

        # Model here

        # Set inputs
        start = time.time()
        self.lc.set_sim_cam(depth_img, left_img, pool_val=4)
        self.lc.process_sim()
        process_time = time.time() - start
        print("process_time: " + str(process_time))
        iterations = 1

        # Strategy to init from Scratch
        if self.just_started:
            self.lc.init_unc_field()
            self.just_started = False
            iterations = 1

        # Iterate
        self.lc.iterate(iterations)

        # Display
        cv2.imshow("field_visual", self.lc.field_visual)
        #self.lc.disp_final_cloud()
        key = cv2.waitKey(15)
        if key == 27:
            self.destroy()
            del self.lc.real_lc
            sys.exit(0)


        # # Save
        # np.save('/home/raaj/rgb.npy', left_img)
        # np.save('/home/raaj/depth.npy', depth_img)
        # stop

# class Ros:
#     def __init__(self):
#         pass

rospy.init_node('real_sensor', anonymous=False)
ros = Ros(params_file="basement_sensor.json", visualize=True, real_sensor=True)
rospy.spin()
stop

# Create LC Object
lc = LC(params_file="toy_sensor.json", visualize=True)

# Initialize the field
lc.init_unc_field()

# Iterate
for i in range(0, 100):
    # Load some Sim Depth
    depth_r = np.load('/home/raaj/adapfusion/external/lcsim/python/example/depth_image.npy')
    rgb_r = np.load('/home/raaj/adapfusion/external/lcsim/python/example/rgb_image.npy').astype(np.float32)/255.
    items = [[250, 300, 50, 50, 10+5], [250, 200, 50, 50, 20.], [250, 100 + i*1, 50, 50, 10.]]
    for item1 in items:
        depth_r[item1[0]:item1[0] + item1[2], item1[1]:item1[1] + item1[3]] = item1[4] + float(i)*0.5
        rgb_r[item1[0]:item1[0] + item1[2], item1[1]:item1[1] + item1[3], 0:2] = 0.1
    depth_r = cv2.resize(depth_r, lc.get_rgb_size(), interpolation = cv2.INTER_LINEAR)
    rgb_r = cv2.resize(rgb_r, lc.get_rgb_size(), interpolation = cv2.INTER_LINEAR)
    # cv2.imshow("depth_r", depth_r/100.)
    # cv2.imshow("rgb_r", rgb_r)

    # Set it
    lc.set_sim_cam(depth_r, rgb_r, pool_val=4)
    # cv2.imshow("depth_lc", lc.depth_lc/100.)

    # Process Sim
    lc.process_sim()

    # Iterate
    if i == 0:
        lc.iterate(10)
    else:
        lc.iterate(5)
