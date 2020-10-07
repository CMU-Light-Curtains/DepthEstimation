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

import rospkg
devel_folder = rospkg.RosPack().get_path('lc_wrapper').split("/")
devel_folder = '/'.join(devel_folder[:len(devel_folder)-3]) + "/devel/lib/"
print(devel_folder)
sys.path.append(devel_folder)
import lc_wrapper_python

class RosLC():
    def __init__(self):
        self.transforms = None
        self.index = 0
        self.prev_index = -1
        self.bridge = CvBridge()
        self.prev_seq = None
        self.just_started = True
        self.mode = rospy.get_param('~mode', 'sim')
        print(self.mode)

        if self.mode == "real":
            lc_wrapper_python.ros_init("lc_wrapper_example")
            self.lc_wrapper = lc_wrapper_python.LightCurtainWrapper(laser_power=30, dev="/dev/ttyACM0")

        self.queue_size = 5
        self.sync = functools.partial(ApproximateTimeSynchronizer, slop=0.01)
        if self.mode == "sim":
            self.depth_sub = message_filters.Subscriber('/left_camera_resized/depth', sensor_msgs.msg.Image)
            self.plan_sub = message_filters.Subscriber('ros_planner/plan_pub', TensorMsg)
            self.ts = self.sync([self.depth_sub, self.plan_sub], self.queue_size)
            self.ts.registerCallback(self.sim_callback)
            self.lc_pub = rospy.Publisher('ros_lc/lc_pub', TensorMsg, queue_size=5)
        elif self.mode == "real":
            self.plan_sub = message_filters.Subscriber('ros_planner/plan_pub', TensorMsg)
            self.plan_sub.registerCallback(self.real_callback)

        #  Params
        with open('real_sensor.json') as f:
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

        # Load Flow Field
        self.algo_lc.fw_large.load_flowfield()

        print("Ready")

    def destroy(self):
        self.plan_sub.unregister()
        if self.mode == "sim":
            self.depth_sub.unregister()
            del self.ts
        
    def sim_callback(self, depth_msg, plan_msg):
        print("sim")

        depth_r = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)/1000.
        depth_lc = self.get_depth_lc(depth_r)
        plan_result = torch.tensor(np.frombuffer(plan_msg.data, np.dtype(np.float32)).reshape(plan_msg.shape))

        joint_all_arr = []
        for design_pts_lc in plan_result:
            output_lc, thickness_lc = self.real_lc.lightcurtain_large.get_return(depth_lc, design_pts_lc, True)
            output_lc[np.isnan(output_lc[:, :, 0])] = 0
            thickness_lc[np.isnan(thickness_lc[:, :])] = 0

            pts = output_lc.reshape((output_lc.shape[0] * output_lc.shape[1], 4))
            thickness = thickness_lc.flatten()
            depth_sensed, int_sensed, thickness_sensed = pylc.transformPoints(pts, thickness, self.real_lc.PARAMS['intr_rgb'],
                                                                              self.real_lc.PARAMS['rTc'],
                                                                              self.real_lc.PARAMS['size_rgb'][0],
                                                                              self.real_lc.PARAMS['size_rgb'][1],
                                                                              {"filtering": 0})

            sensed_arr = np.array([depth_sensed, int_sensed, thickness_sensed]).astype(np.float32)
            joint_all_arr.append(sensed_arr)

        joint_all_arr = np.array(joint_all_arr).astype(np.float32)
        print(joint_all_arr.shape)

        ros_lc = TensorMsg()
        ros_lc.header = plan_msg.header
        ros_lc.shape = joint_all_arr.shape
        ros_lc.data = joint_all_arr.tostring()
        self.lc_pub.publish(ros_lc)

        # Index
        self.index += 1

    def real_callback(self, plan_msg):
        print("real")

        plan_result = torch.tensor(np.frombuffer(plan_msg.data, np.dtype(np.float32)).reshape(plan_msg.shape))

        output_lc, thickness_lc = lc_wrapper.sendAndWait(plan_result)
        output_lc[np.isnan(output_lc[:, :, 0])] = 0
        thickness_lc[np.isnan(thickness_lc[:, :])] = 0

        pts = output_lc.reshape((output_lc.shape[0] * output_lc.shape[1], 4))
        thickness = thickness_lc.flatten()
        depth_sensed, int_sensed, thickness_sensed = pylc.transformPoints(pts, thickness, self.real_lc.PARAMS['intr_rgb'],
                                                                            self.real_lc.PARAMS['rTc'],
                                                                            self.real_lc.PARAMS['size_rgb'][0],
                                                                            self.real_lc.PARAMS['size_rgb'][1],
                                                                            {"filtering": 0})

        sensed_arr = np.array([depth_sensed, int_sensed, thickness_sensed]).astype(np.float32)

        ros_lc = TensorMsg()
        ros_lc.header = plan_msg.header
        ros_lc.shape = sensed_arr.shape
        ros_lc.data = sensed_arr.tostring()
        self.lc_pub.publish(ros_lc)

        # Index
        self.index += 1

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


rospy.init_node('ros_lc', anonymous=False)
roslc = RosLC()
rospy.spin()