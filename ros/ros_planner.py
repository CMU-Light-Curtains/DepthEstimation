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

class RosPlanner():
    def __init__(self):
        self.transforms = None
        self.index = 0
        self.prev_index = -1
        self.bridge = CvBridge()
        self.prev_seq = None
        self.just_started = True
        self.mode = rospy.get_param('~mode', 'sim')
        print(self.mode)

        self.q_msg = deque([], 1)
        lth = ConsumerThread(self.q_msg, self.handle_msg)
        lth.setDaemon(True)
        lth.start()

        self.queue_size = 5
        self.sync = functools.partial(ApproximateTimeSynchronizer, slop=0.01)
        self.depth_sub = message_filters.Subscriber('/left_camera_resized/depth', sensor_msgs.msg.Image)
        self.unc_sub = message_filters.Subscriber('ros_net/unc_pub', TensorMsg)
        self.ts = self.sync([self.depth_sub, self.unc_sub], self.queue_size)
        self.ts.registerCallback(self.callback)
        self.plan_pub = rospy.Publisher('ros_planner/plan_pub', TensorMsg, queue_size=5)
        self.debug_pub = rospy.Publisher('ros_planner/debug', sensor_msgs.msg.Image, queue_size=5)

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

        # Pin Memory
        self.intr_r_tensor = torch.tensor(self.real_param['intr_rgb']).cuda()
        self.param = param
        self.depth_pinned = torch.zeros((int(self.param["size_rgb"][1]), int(self.param["size_rgb"][0]))).float().pin_memory()
        self.unc_pinned = torch.zeros(1,64, int(self.param["size_rgb"][0])).float().pin_memory()
        print("Ready")

    def destroy(self):
        self.depth_sub.unregister()
        self.unc_sub.unregister()
        del self.ts
        
    def callback(self, depth_msg, unc_msg):
        # Append msg
        self.q_msg.append((depth_msg, unc_msg))

        # Index
        self.index += 1

    # def get_depth_lc(self, depth_r, pool_val=4):
    #     # Warp Depth Image to LC
    #     pts_rgb = img_utils.depth_to_pts(torch.Tensor(depth_r).unsqueeze(0), self.real_param['intr_rgb'])
    #     pts_rgb = pts_rgb.reshape((pts_rgb.shape[0], pts_rgb.shape[1] * pts_rgb.shape[2]))
    #     pts_rgb = torch.cat([pts_rgb, torch.ones(1, pts_rgb.shape[1])])
    #     pts_rgb = pts_rgb.numpy().T
    #     thick_rgb = np.ones((pts_rgb.shape[0], 1)).astype(np.float32)
    #     uniform_params = {"filtering": 2}
    #     depth_lc, _, _ = pylc.transformPoints(pts_rgb, thick_rgb, self.real_param['intr_lc'], self.real_param['cTr'],
    #                                         self.real_param['size_lc'][0], self.real_param['size_lc'][1],
    #                                         uniform_params)
        
    #     depth_lc = img_utils.minpool(torch.Tensor(depth_lc).unsqueeze(0).unsqueeze(0), pool_val, 1000).squeeze(0).squeeze(0).numpy()
    #     depth_lc = cv2.resize(depth_lc, (0,0), fx=pool_val, fy=pool_val, interpolation = cv2.INTER_NEAREST)
    #     return depth_lc

    def handle_msg(self, msg):
        if self.prev_index == self.index:
            time.sleep(0.00001)
            return
        self.prev_index = self.index
        print("enter")

        start = time.time()
        depth_msg, unc_msg = msg
        depth_img = torch.tensor(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)/1000.)
        unc_img = torch.tensor(np.frombuffer(unc_msg.data, np.dtype(np.float32)).reshape(unc_msg.shape))

        # Copy to GPU
        self.unc_pinned[:] = unc_img[:]
        unc_field_predicted_r = self.unc_pinned.cuda()
        self.depth_pinned[:] = depth_img[:]
        depth_r_tensor = self.depth_pinned.cuda().unsqueeze(0)

        # Upsample
        unc_field_predicted_r = img_utils.upsample_dpv(unc_field_predicted_r.unsqueeze(2), N=self.algo_lc.expand_A, BV_log=False).squeeze(2)

        # Make a DPV out of the Depth Map
        mask_truth = (depth_r_tensor > 0.).float()
        dpv_truth = img_utils.gen_dpv_withmask(depth_r_tensor, mask_truth.unsqueeze(0), self.algo_lc.d_candi, 0.3)
        depth_truth = img_utils.dpv_to_depthmap(dpv_truth, self.algo_lc.d_candi, BV_log=False) * mask_truth

        # Make Unc Field Truth (RGB)
        unc_field_truth_r, debugmap = img_utils.gen_ufield(dpv_truth, self.algo_lc.d_candi, self.intr_r_tensor.squeeze(0), BV_log=False, cfgx=self.real_param)

        # Unc Field Truth in LC (LC)
        unc_field_truth_lc = self.algo_lc.fw_large.preprocess(unc_field_truth_r.squeeze(0), self.algo_lc.d_candi, self.algo_lc.d_candi_up)
        unc_field_truth_lc = self.algo_lc.fw_large.transformZTheta(unc_field_truth_lc, self.algo_lc.d_candi_up, self.algo_lc.d_candi_up, "transform_" + "large").unsqueeze(0)

        # Unc Field Predicted in LC (LC)
        unc_field_predicted_lc = self.algo_lc.fw_large.preprocess(unc_field_predicted_r.squeeze(0), self.algo_lc.d_candi, self.algo_lc.d_candi_up)
        unc_field_predicted_lc = self.algo_lc.fw_large.transformZTheta(unc_field_predicted_lc, self.algo_lc.d_candi_up, self.algo_lc.d_candi_up, "transform_" + "large").unsqueeze(0)

        # Get Planner
        planner = "m1"
        start = time.time()
        if planner == "default":
            params = {"step": [0.5]}
            plan_func = self.algo_lc.plan_default
        elif planner == "m1":
            params = {"step": 3}
            plan_func = self.algo_lc.plan_m1
        elif planner == "sweep":
            params = {"start": self.S_RANGE + 1, "end": self.E_RANGE - 1, "step": 0.25}
            plan_func = self.algo_lc.plan_sweep
        elif planner == "empty":
            params = {}
            plan_func = self.algo_lc.plan_empty
        time_plan = time.time() - start

        # Plan
        if self.mode == "real":
            for lc_path in plan_func(unc_field_predicted_r.squeeze(0), self.algo_lc.planner_large, self.algo_lc.fw_large, "high", params, yield_mode=True):
                lc_path = lc_path.astype(np.float32)
                ros_path = TensorMsg()
                ros_path.header = unc_msg.header
                ros_path.shape = lc_path.shape
                ros_path.data = lc_path.tostring()
                self.plan_pub.publish(ros_path)
        elif self.mode == "sim":
            lc_paths = list(plan_func(unc_field_predicted_r.squeeze(0), self.algo_lc.planner_large, self.algo_lc.fw_large, "high", params, yield_mode=True))
            lc_paths = np.array(lc_paths).astype(np.float32)
            ros_path = TensorMsg()
            ros_path.header = unc_msg.header
            ros_path.shape = lc_paths.shape
            ros_path.data = lc_paths.tostring()
            self.plan_pub.publish(ros_path)

        # Viz
        if self.debug_pub.get_num_connections():
            field_visual = self.algo_lc.field_visual
            field_visual[:,:,2] = unc_field_truth_lc[0,:,:].cpu().numpy()*3
            ros_debug = self.bridge.cv2_to_imgmsg((field_visual*255).astype(np.uint8), encoding="bgr8")
            ros_debug.header = unc_msg.header
            self.debug_pub.publish(ros_debug)

rospy.init_node('ros_planner', anonymous=False)
rosplanner = RosPlanner()
rospy.spin()