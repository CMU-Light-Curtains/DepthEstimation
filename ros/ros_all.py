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

from easydict import EasyDict
from models.get_model import get_model
from utils.torch_utils import bias_parameters, weight_parameters, \
    load_checkpoint, save_checkpoint, AdamW
import warping.view as View
import torchvision.transforms as transforms

from sensor_bridge.msg import TensorMsg
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import zlib
from threading import Thread, Lock

import rospkg
devel_folder = rospkg.RosPack().get_path('lc_wrapper').split("/")
devel_folder = '/'.join(devel_folder[:len(devel_folder)-3]) + "/devel/lib/"
print(devel_folder)
sys.path.append(devel_folder)
import lc_wrapper_python

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

class Planner():
    def __init__(self, mode, params_file):
        self.mode = mode
        self.counter = 0
        self.just_started = True

        if self.mode == "real":
            lc_wrapper_python.ros_init("lc_wrapper_example")
            self.lc_wrapper = lc_wrapper_python.LightCurtainWrapper(laser_power=100, dev="/dev/ttyACM0")

        #  Params
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
        param["device"] = torch.device(0)
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

        # Load Flow Field
        self.algo_lc.fw_large.load_flowfield()

        # Pin Memory
        self.intr_r_tensor = torch.tensor(self.real_param['intr_rgb']).cuda()

        # Init Field
        self.init_unc_field()

        # Gather
        self.unc_scores = []

    def init_unc_field(self):
        init_depth = torch.zeros((1, self.real_param["size_rgb"][1], self.real_param["size_rgb"][0])).cuda() + 5.
        self.final = torch.log(img_utils.gen_dpv_withmask(init_depth, init_depth.unsqueeze(0)*0+1, self.algo_lc.d_candi, 10.0))

    def integrate(self, DPVs):
        # Keep Renormalize
        curr_dist = torch.clamp(torch.exp(self.final), img_utils.epsilon, 1.)

        # # Update
        # for dpv in DPVs:
        #     dpv = torch.clamp(dpv, img_utils.epsilon, 1.)
        #     curr_dist = curr_dist * dpv
        #     curr_dist = curr_dist / torch.sum(curr_dist, dim=1).unsqueeze(1)

        # Update
        curr_dist_log = torch.log(curr_dist)
        for dpv in DPVs:
            dpv = torch.clamp(dpv, img_utils.epsilon, 1.)
            curr_dist_log += torch.log(dpv)
        curr_dist = torch.exp(curr_dist_log)
        curr_dist = curr_dist / torch.sum(curr_dist, dim=1).unsqueeze(1)

        # Spread
        for i in range(0, 1):
            curr_dist = img_utils.spread_dpv_hack(curr_dist, 5)
        # if self.counter < 20:
        #     for i in range(0, 1):
        #         curr_dist = img_utils.spread_dpv_hack(curr_dist, 5)
        # else:
        #     for i in range(0, 3):
        #         curr_dist = img_utils.spread_dpv_hack(curr_dist, 3)
        
        # Keep Renormalize
        curr_dist = torch.clamp(curr_dist, img_utils.epsilon, 1.)

        # Back to Log space
        self.final = torch.log(curr_dist)

    def get_depth_lc(self, depth_r, pool_val=4):
        # Warp Depth Image to LC
        pts_rgb = img_utils.depth_to_pts(torch.Tensor(depth_r).unsqueeze(0), self.real_param['intr_rgb'])
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
        return depth_lc

    def run(self, dpv_r_tensor, depth_r_tensor):
        self.counter+=1
        print(self.counter)
        global_time = time.time()
        start = time.time()

        # Generate Depth LC
        depth_lc = None
        if self.mode == "sim":
            depth_lc = self.get_depth_lc(depth_r_tensor.squeeze(0).cpu().numpy())

        # Copy to GPU
        # self.unc_pinned[:] = unc_img[:]
        # unc_field_predicted_r = self.unc_pinned.cuda()
        intr_r_tensor = self.intr_r_tensor

        # Make a DPV out of the Depth Map
        mask_truth = (depth_r_tensor > 0.).float()
        dpv_truth = img_utils.gen_dpv_withmask(depth_r_tensor, mask_truth.unsqueeze(0), self.algo_lc.d_candi, 0.3)
        depth_truth = img_utils.dpv_to_depthmap(dpv_truth, self.algo_lc.d_candi, BV_log=False) * mask_truth

        # Make Unc Field Truth (RGB) (LC)
        unc_field_truth_r, debugmap = img_utils.gen_ufield(dpv_truth, self.algo_lc.d_candi, intr_r_tensor.squeeze(0), BV_log=False, cfgx=self.real_param)
        unc_field_truth_lc = self.algo_lc.fw_large.preprocess(unc_field_truth_r.squeeze(0), self.algo_lc.d_candi, self.algo_lc.d_candi_up)
        unc_field_truth_lc = self.algo_lc.fw_large.transformZTheta(unc_field_truth_lc, self.algo_lc.d_candi_up, self.algo_lc.d_candi_up, "transform_" + "large").unsqueeze(0)
        unc_field_truth_lc[:,:,0:50] = np.nan
        unc_field_truth_lc[:,:,-50:-1] = np.nan
        preprocess_time = time.time() - start
        print("preprocess_time: " + str(preprocess_time)) # 30ms

        # cv2.imshow("win", unc_field_truth_lc.squeeze(0).cpu().numpy()*3)
        # cv2.waitKey(1)
        # return

        # Get Planner
        planner = "default"
        start = time.time()
        if planner == "default":
            params = {"step": [0.5], "std_div": 5.}
            plan_func = self.algo_lc.plan_default
        elif planner == "m1":
            params = {"step": 3, "interval": 5, "std_div": 1.}
            plan_func = self.algo_lc.plan_m1
        elif planner == "sweep":
            params = {"start": self.S_RANGE + 1, "end": self.E_RANGE - 1, "step": 0.25}
            plan_func = self.algo_lc.plan_sweep
        elif planner == "empty":
            params = {}
            plan_func = self.algo_lc.plan_empty
        setup_time = time.time() - start
        #print("setup_time: " + str(setup_time)) # 30ms

        # Expand DPV
        start = time.time()
        if dpv_r_tensor is not None:
            #self.final = img_utils.upsample_dpv(dpv_r_tensor, N=self.real_lc.expand_A, BV_log=True)
            dpv_r_tensor = torch.exp(img_utils.upsample_dpv(dpv_r_tensor, N=self.real_lc.expand_A, BV_log=True))
            for i in range(0,3):
                self.integrate([dpv_r_tensor])
        dpv_int_time = time.time() - start
        print("dpv_int_time: " + str(dpv_int_time)) # 30ms

        # Iterations
        if self.just_started:
            iterations = 1
            self.just_started = False
        else:
            iterations = 1

        # Iterations
        print("***********************")
        for i in range(0, iterations):

            # Generate UField (in RGB)
            start = time.time()
            unc_field_predicted_r, _ = img_utils.gen_ufield(self.final, self.algo_lc.d_candi, intr_r_tensor.squeeze(0), BV_log=True, cfgx=self.real_param)
            time_gen_ufield = time.time() - start
            #print("time_gen_ufield: " + str(time_gen_ufield)) # 30ms

            # Hack
            #unc_field_predicted_r[:,-10:-1] = 0

            # Score (Need to compute in LC space as it is zoomed in sadly)
            start = time.time()
            unc_field_predicted_lc = self.algo_lc.fw_large.preprocess(unc_field_predicted_r.squeeze(0), self.algo_lc.d_candi, self.algo_lc.d_candi_up)
            unc_field_predicted_lc = self.algo_lc.fw_large.transformZTheta(unc_field_predicted_lc, self.algo_lc.d_candi_up, self.algo_lc.d_candi_up, "transform_" + "large").unsqueeze(0)
            unc_score = img_utils.compute_unc_rmse(unc_field_truth_lc, unc_field_predicted_lc, self.algo_lc.d_candi, False)
            time_score = time.time() - start
            #print("time_score: " + str(time_gen_ufield)) # 30ms
            print(unc_score)
            self.unc_scores.append(unc_score.item())
            if self.counter % 20 == 0:
                print(self.unc_scores)
            # cv2.imshow("FAGA", unc_field_truth_lc.squeeze(0).cpu().numpy())
            # cv2.imshow("FAGB", unc_field_predicted_lc.squeeze(0).cpu().numpy())
            # cv2.waitKey(1)

            # Plan and Sense
            if self.mode == "real":
            
                # Send Curtains while planning
                sensed_arr_all = []
                start = time.time()
                i=-1
                for lc_path in plan_func(unc_field_predicted_r.squeeze(0), self.algo_lc.planner_large, self.algo_lc.fw_large, "high", params, yield_mode=True):
                    i+=1
                    lc_path = lc_path.astype(np.float32)

                    """
                    send
                    if first:
                        continue
                    getLastCurtain()
                    
                    outside of loop, call getLastCurtain() again
                    """

                    # Send Curtain
                    self.lc_wrapper.sendCurtain(lc_path)
                    if i==0:
                        continue

                    # Receive Curtain
                    output_lc, thickness_lc = self.lc_wrapper.receiveCurtainAndProcess()[1]
                    output_lc[np.isnan(output_lc[:, :, 0])] = 0
                    thickness_lc[np.isnan(thickness_lc[:, :])] = 0

                    # Transform to RGB
                    sensed_arr = self.real_lc.transform_measurement(output_lc, thickness_lc)
                    sensed_arr_all.append(sensed_arr)

                    # # Once I receive a curtain I send it out
                    # start = time.time()
                    # output_lc, thickness_lc = self.lc_wrapper.sendAndWait(lc_path)
                    # output_lc[np.isnan(output_lc[:, :, 0])] = 0
                    # thickness_lc[np.isnan(thickness_lc[:, :])] = 0
                    # curtains.append([output_lc, thickness_lc])
                    # measure_time = time.time() - start

                # Receive Curtain
                output_lc, thickness_lc = self.lc_wrapper.receiveCurtainAndProcess()[1]
                output_lc[np.isnan(output_lc[:, :, 0])] = 0
                thickness_lc[np.isnan(thickness_lc[:, :])] = 0

                # Transform to RGB
                sensed_arr = self.real_lc.transform_measurement(output_lc, thickness_lc)
                sensed_arr_all.append(sensed_arr)

                base_time = time.time() - start
                print("base_time: " + str(base_time)) # 30ms

                # Generate DPV
                start = time.time()
                lc_DPVs = []
                for sensed_arr in sensed_arr_all:

                    # Gen DPV
                    lc_DPV = self.real_lc.gen_lc_dpv(sensed_arr, params["std_div"])
                    lc_DPVs.append(lc_DPV)

                dpv_time = time.time() - start
                #print("dpv_time: " + str(dpv_time)) # 30ms

            # Plane and Sense
            elif self.mode == "sim":
                lc_paths = list(plan_func(unc_field_predicted_r.squeeze(0), self.algo_lc.planner_large, self.algo_lc.fw_large, "high", params, yield_mode=True))
                field_visual = self.algo_lc.field_visual
                field_visual[:,:,2] = unc_field_truth_lc[0,:,:].cpu().numpy()*3
                #cv2.imshow("field_visual", field_visual)
                #cv2.waitKey(1)

                # In Sensing Real
                lc_DPVs = []
                for lc_path in lc_paths:
                    
                    # Take Measurement
                    start = time.time()
                    output_lc, thickness_lc = self.real_lc.lightcurtain_large.get_return(depth_lc, lc_path, True)
                    output_lc[np.isnan(output_lc[:, :, 0])] = 0
                    thickness_lc[np.isnan(thickness_lc[:, :])] = 0
                    measure_time = time.time() - start

                    # Transform to LC
                    start = time.time()
                    sensed_arr = self.real_lc.transform_measurement(output_lc, thickness_lc)
                    transform_time = time.time() - start

                    # Gen DPV
                    start = time.time()
                    lc_DPV = self.real_lc.gen_lc_dpv(sensed_arr, 10.)
                    lc_DPVs.append(lc_DPV)
                    dpv_time = time.time() - start

                    print((measure_time, transform_time, dpv_time))

            # Integrate Measurement
            start = time.time()
            self.integrate(lc_DPVs)
            int_time = time.time() - start
            #print("int_time: " + str(int_time)) # 30ms

            # Gen Depth removing uncertainties
            z = torch.exp(self.final.squeeze(0))
            d_candi_expanded = torch.tensor(self.algo_lc.d_candi).unsqueeze(1).unsqueeze(1).cuda()
            mean = torch.sum(d_candi_expanded * z, dim=0)
            variance = torch.sum(((d_candi_expanded - mean) ** 2) * z, dim=0)
            mask_var = (variance < 2.0).float()
            final_depth = (mean * mask_var).cpu().numpy()

            print("****global time: " + str(time.time() - global_time))

            # Return
            field_visual = self.algo_lc.field_visual
            field_visual[:,:,2] = unc_field_truth_lc[0,:,:].cpu().numpy()*3
        
        return field_visual, final_depth

class RosAll():
    def __init__(self):
        self.transforms = None
        self.index = 0
        self.prev_index = -1
        self.bridge = CvBridge()
        self.prev_seq = None
        self.just_started = True
        self.mode = rospy.get_param('~mode', 'sim')
        self.counter = 0
        self.mutex = Lock()
        self.new_dpv = False
        print(self.mode)

        self.planner = Planner(mode = self.mode, params_file='real_sensor.json')

        # Pin Memory
        self.depth_pinned = torch.zeros((int(self.planner.real_param["size_rgb"][1]), int(self.planner.real_param["size_rgb"][0]))).float().pin_memory()
        self.dpv_pinned = torch.zeros(1,64, int(self.planner.real_param["size_rgb"][1]), int(self.planner.real_param["size_rgb"][0])).float().pin_memory()
        self.dpv_r_tensor = None

        # ROS
        self.queue_size = 1
        self.sync = functools.partial(ApproximateTimeSynchronizer, slop=0.01)
        self.depth_sub = message_filters.Subscriber('/left_camera_resized/depth', sensor_msgs.msg.Image, queue_size=self.queue_size, buff_size=2**24)
        self.dpv_sub = message_filters.Subscriber('ros_net/dpv_pub', TensorMsg, queue_size=self.queue_size)
        #self.ts = self.sync([self.depth_sub, self.dpv_sub], self.queue_size)
        self.depth_sub.registerCallback(self.depth_callback)
        self.dpv_sub.registerCallback(self.dpv_callback)
        self.debug_pub = rospy.Publisher('ros_planner/debug', sensor_msgs.msg.Image, queue_size=self.queue_size)
        self.depth_pub = rospy.Publisher('ros_planner/depth', sensor_msgs.msg.Image, queue_size=self.queue_size)

    def destroy(self):
        self.depth_sub.unregister()
        self.dpv_sub.unregister()

    def dpv_callback(self, msg):
        dpv_msg = msg
        dpv_img = torch.tensor(np.frombuffer(dpv_msg.data, np.dtype(np.float32)).reshape(dpv_msg.shape))
        self.dpv_pinned[:] = dpv_img[:]
        self.mutex.acquire()
        self.dpv_r_tensor = self.dpv_pinned.cuda()
        self.new_dpv = True
        self.mutex.release()

    def depth_callback(self, msg):
        self.counter+=1
        print(self.counter)
        global_time = time.time()

        # Get DPV from RGB
        self.mutex.acquire()
        dpv_r_tensor = None
        if self.new_dpv:
            dpv_r_tensor = self.dpv_r_tensor.clone()
            self.new_dpv = False
            print("gotdpv")
        self.mutex.release()

        # # Hack
        # dpv_r_tensor = None
        # depth_r = np.load('/home/raaj/adapfusion/external/lcsim/python/example/depth_image.npy')
        # items = [[250, 300, 50, 50, 10+5], [250, 200, 50, 50, 20.], [250, 100, 50, 50, 10.]]
        # for item1 in items:
        #     depth_r[item1[0]:item1[0] + item1[2], item1[1]:item1[1] + item1[3]] = item1[4] + float(self.counter * 0.1)
        # depth_img = torch.tensor(cv2.resize(depth_r, self.get_rgb_size(), interpolation = cv2.INTER_LINEAR))

        # Convert to Tensor
        start = time.time()
        depth_msg = msg
        depth_img = torch.tensor(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)/1000.)
        self.depth_pinned[:] = depth_img[:]
        depth_r_tensor = self.depth_pinned.cuda(non_blocking=True).unsqueeze(0)

        # Call Planner
        field_visual, final_depth = self.planner.run(dpv_r_tensor, depth_r_tensor)

        # Viz
        if self.debug_pub.get_num_connections():
            start = time.time()
            ros_debug = self.bridge.cv2_to_imgmsg((field_visual*255).astype(np.uint8), encoding="bgr8")
            ros_debug.header = msg.header
            self.debug_pub.publish(ros_debug)
            pub_time = time.time() - start
            #print("pub_time: " + str(int_time)) # 30ms

        if self.depth_pub.get_num_connections():
            ros_depth = (final_depth * 1000).astype(np.uint16)
            ros_depth = self.bridge.cv2_to_imgmsg(ros_depth)
            ros_depth.header = depth_msg.header
            ros_depth.header.stamp = rospy.Time.now()
            self.depth_pub.publish(ros_depth)

if __name__ == '__main__':
    rospy.init_node('ros_all', anonymous=False)
    rosall = RosAll()
    rospy.spin()