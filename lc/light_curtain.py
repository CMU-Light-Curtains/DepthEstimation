# Python
import numpy as np
import time
import sys
import os

import torch
import torch.nn.functional as F

sys.path.append("external/lcsim/python")
from sim import LCDevice
from planner import PlannerRT
import pylc_lib as pylc

import utils.img_utils as img_utils
import cv2

class FieldWarp:
    def __init__(self, intr_input, dist_input, size_input, intr_output, dist_output, size_output, output2input, name, device):
        # Assign
        self.intr_input = intr_input
        self.dist_input = dist_input
        self.size_input = size_input
        self.intr_output = intr_output
        self.dist_output = dist_output
        self.size_output = size_output
        self.output2input = output2input
        self.name = name
        self.device = device

        # Compute Scaled
        self.intr_input_scaled = img_utils.intr_scale(self.intr_input, self.size_input, self.size_output)
        self.dist_input_scaled = self.dist_input
        self.size_input_scaled = self.size_output

        # Compute angles
        self.angles_input = pylc.generateCameraAngles(self.intr_input, self.dist_input, self.size_input[0],
                                                      self.size_input[1])
        self.angles_input_scaled = pylc.generateCameraAngles(self.intr_input_scaled, self.dist_input_scaled,
                                                             self.size_input_scaled[0], self.size_input_scaled[1])
        self.angles_output = pylc.generateCameraAngles(self.intr_output, self.dist_output, self.size_output[0],
                                                       self.size_output[1])

        # Previous Fields
        self.flowfields = dict()

    def warp(self, input, flowfield):
        gridfield = torch.zeros(flowfield.shape).to(input.device)
        ax = torch.arange(0, input.shape[0]).float().to(input.device)
        bx = torch.arange(0, input.shape[1]).float().to(input.device)
        yv, xv = torch.meshgrid([ax, bx])
        ystep = 2. / float(input.shape[0] - 1)
        xstep = 2. / float(input.shape[1] - 1)
        gridfield[0, :, :, 0] = -1 + xv * xstep - flowfield[0, :, :, 0] * xstep
        gridfield[0, :, :, 1] = -1 + yv * ystep - flowfield[0, :, :, 1] * ystep
        input = input.unsqueeze(0).unsqueeze(0)
        output = F.grid_sample(input, gridfield, mode='bilinear').squeeze(0).squeeze(0)
        return output

    def digitize_soft(self, input, array):
        position = np.digitize([input], array)[0] - 1
        lp = array[position]
        # if position == len(array) - 1 or position == -1:
        #     return position

        if abs(input - array[-1]) < 1e-7:
            return len(array) - 1
        elif abs(input - array[0]) < 1e-7:
            return 0
        elif position == len(array) - 1:
            return 100000000
        elif position == -1:
            return -100000000

        rp = array[position + 1]
        soft_position = position + 1. * (float(input - lp) / float(rp - lp))
        return soft_position

    def preprocess(self, field, candi_input, candi_output):
        assert field.shape[0] == len(candi_input)
        assert field.shape[1] == self.size_input[0]
        field = field.unsqueeze(0).unsqueeze(0)
        field = F.upsample(field, size=[len(candi_output), self.size_input_scaled[0]], scale_factor=None,
                           mode='bilinear').squeeze(0).squeeze(0)
        return field

    def _ztheta2zrange(self, field, angles, d_candi, r_candi):
        r_field = torch.zeros(field.shape).to(field.device)
        flowfield = torch.zeros((1, field.shape[0], field.shape[1], 2)).to(field.device)
        assert r_field.shape[1] == len(angles)
        assert r_field.shape[0] == len(d_candi)
        assert r_field.shape[0] == len(r_candi)

        for r in range(0, r_field.shape[0]):
            for c in range(0, r_field.shape[1]):
                # Extract Values
                rng = r_candi[r]
                theta = angles[c]

                # Compute XYZ
                yval = 0.
                xval = rng * np.sin(np.radians(theta))
                zval = rng * np.cos(np.radians(theta))
                pt = np.array([xval, yval, zval, 1.]).reshape((4, 1))

                zbin = self.digitize_soft(zval, d_candi)
                thetabin = self.digitize_soft(theta, angles)

                # Set
                # r_field[r,c] = field[zbin, thetabin]
                flowfield[0, r, c, 0] = c - thetabin
                flowfield[0, r, c, 1] = r - zbin

        r_field = self.warp(field, flowfield)

        return r_field, flowfield

    def _transformZTheta(self, field, angles_input, d_candi_input, angles_output, d_candi_output, output2input):
        assert field.shape[1] == len(angles_input)
        assert field.shape[1] == len(angles_output)
        assert field.shape[0] == len(d_candi_input)
        assert field.shape[0] == len(d_candi_output)
        flowfield = torch.zeros((1, field.shape[0], field.shape[1], 2)).to(field.device)

        for r in range(0, field.shape[0]):
            for c in range(0, field.shape[1]):
                # r controls the d_candi_lc
                # c controls the angle
                zval = d_candi_output[r]
                theta = angles_output[c]

                # Compute XYZ
                rng = np.sqrt((np.power(zval, 2)) / (1 - np.power(np.sin(np.radians(theta)), 2)))
                xval = rng * np.sin(np.radians(theta))
                yval = 0
                pt = np.array([xval, yval, zval, 1.]).reshape((4, 1))

                # Transform
                tpt = np.matmul(output2input, pt)[:, 0]

                # Compute RTheta
                rng = np.sqrt(tpt[0] * tpt[0] + tpt[1] * tpt[1] + tpt[2] * tpt[2])
                zval = tpt[2]
                xval = tpt[0]
                theta = np.degrees(np.arcsin(xval / rng))

                zbin = self.digitize_soft(zval, d_candi_input)
                thetabin = self.digitize_soft(theta, angles_input)

                flowfield[0, r, c, 0] = c - thetabin
                flowfield[0, r, c, 1] = r - zbin

        n_field = self.warp(field, flowfield)

        return n_field, flowfield

    def ztheta2zrange_input(self, field, d_candi, r_candi, name):
        if name in self.flowfields.keys():
            output = self.warp(field, self.flowfields[name])
            return output
        else:
            print("Recomputing ztheta2zrange_input")
            output, flowfield = self._ztheta2zrange(field, self.angles_input_scaled, d_candi, r_candi)
            self.flowfields[name] = flowfield
            return output

    def ztheta2zrange_output(self, field, d_candi, r_candi, name):
        if name in self.flowfields.keys():
            output = self.warp(field, self.flowfields[name])
            return output
        else:
            print("Recomputing ztheta2zrange_output")
            output, flowfield = self._ztheta2zrange(field, self.angles_output, d_candi, r_candi)
            self.flowfields[name] = flowfield
            return output

    def transformZTheta(self, field, d_candi_input, d_candi_output, name):
        if name in self.flowfields.keys():
            output = self.warp(field, self.flowfields[name])
            return output
        else:
            print("Recomputing transformZTheta")
            output, flowfield = self._transformZTheta(field, self.angles_input_scaled, d_candi_input,
                                                      self.angles_output, d_candi_output, self.output2input)
            self.flowfields[name] = flowfield
            return output

    def save_flowfields(self):
        if str(self.device) != "cuda:0":
            return
        if os.path.isfile(self.name + "_lc_flowfields.npy"):
            return
        print("Saving Flowfields " + str(self.device))
        np.save(self.name + "_lc_flowfields.npy", self.flowfields)

    def load_flowfield(self):
        if len(self.flowfields.keys()):
            return
        if os.path.isfile(self.name + "_lc_flowfields.npy") and os.access(self.name + "_lc_flowfields.npy", os.R_OK):
            self.flowfields = np.load(self.name + "_lc_flowfields.npy", allow_pickle=True).item()

            # Store on correct GPU
            for key in self.flowfields.keys():
                self.flowfields[key] = self.flowfields[key].to(self.device)


def normalize(field):
    minv, _ = field.min(1)  # [1,384]
    maxv, _ = field.max(1)  # [1,384]
    return (field - minv) / (maxv - minv)


def create_mean_kernel(N):
    kernel = torch.Tensor(np.zeros((N, N)).astype(np.float32))
    kernel[:, int(N / 2)] = 1 / float(N)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    params = {'weight': kernel, 'padding': N // 2}
    return params


def invert(x, p=0.5):
    #efield = -((1 / np.sqrt(0.5) * (x - 0.5)) ** 2) + 0.5
    efield = ((x**p)*((1-x)**(1-p)))/((p**p)*((1-p)**(1-p)))
    #efield = efield*0.5
    return efield


def mapping(x):
    # https://www.desmos.com/calculator/htpohhqx1a

    def ma(x, m):
        A = -1. / ((m) * ((0.5 / m) + x)) + 1.
        return A

    def mb(x, m, f):
        c = m / ((m * f + 0.5) ** 2)
        y = c * x + (1 - c)
        return y

    m = 20
    f = 0.5
    mask = x > f
    y = ~mask * ma(x, m=m) + mask * mb(x, m=m, f=f)
    return y


def mixed_model(d_candi, z_img, unc_img, A, B):
    mixed_dist = img_utils.gen_soft_label_torch(d_candi, z_img, unc_img, zero_invalid=True,
                                                pow=2.) * A + img_utils.gen_uniform(d_candi, z_img) * B
    mixed_dist = torch.clamp(mixed_dist, 0, np.inf)
    mixed_dist = mixed_dist / torch.sum(mixed_dist, dim=0)
    return mixed_dist


class LightCurtain:
    def __init__(self):
        self.lightcurtain = None
        self.initialized = False

    # intr_rgb, dist_rgb, size_rgb, intr_lc, dist_lc, size_lc, lc2rgb, lc2laser, laser_fov

    def get_flat(self, z):
        points = []
        for x in np.arange(-10., 10., 0.01):
            points.append([x, z])
        return np.array(points).astype(np.float32)

    def init(self, PARAMS):
        self.PARAMS = PARAMS
        CAMERA_PARAMS_LARGE = {
            'width': PARAMS["size_lc"][0],
            'height': PARAMS["size_lc"][1],
            'matrix': PARAMS["intr_lc"],
            'distortion': PARAMS["dist_lc"],
            'hit_mode': 1,
            'hit_noise': 0.01
        }
        CAMERA_PARAMS_SMALL = {
            'width': int(PARAMS["size_lc"][0] / 4),
            'height': int(PARAMS["size_lc"][1] / 4),
            'matrix': img_utils.intr_scale_unit(PARAMS["intr_lc"], 1 / 4.),
            'distortion': PARAMS["dist_lc"],
            'hit_mode': 1,
            'hit_noise': 0.01,
        }
        LASER_PARAMS_LARGE = {
            'lTc': PARAMS["lTc"],
            'fov': PARAMS["laser_fov"],
            'laser_timestep': PARAMS['laser_timestep']
        }
        LASER_PARAMS_SMALL = {
            'lTc': PARAMS["lTc"],
            'fov': PARAMS["laser_fov"],
            'laser_timestep': PARAMS['laser_timestep'] * 4
        }
        PARAMS["intr_rgb_small"] = img_utils.intr_scale_unit(PARAMS["intr_rgb"], 1 / 4.)
        PARAMS["intr_lc_small"] = img_utils.intr_scale_unit(PARAMS["intr_lc"], 1 / 4.)
        PARAMS["size_rgb_small"] = (int(PARAMS["size_rgb"][0] / 4), int(PARAMS["size_rgb"][1] / 4))
        PARAMS["size_lc_small"] = (int(PARAMS["size_lc"][0] / 4), int(PARAMS["size_lc"][1] / 4))
        self.lightcurtain_large = LCDevice(CAMERA_PARAMS=CAMERA_PARAMS_LARGE, LASER_PARAMS=LASER_PARAMS_LARGE)
        self.lightcurtain_small = LCDevice(CAMERA_PARAMS=CAMERA_PARAMS_SMALL, LASER_PARAMS=LASER_PARAMS_SMALL)
        self.planner_large = PlannerRT(self.lightcurtain_large, PARAMS["r_candi_up"], PARAMS["size_lc"][0], debug=False)
        self.planner_small = PlannerRT(self.lightcurtain_small, PARAMS["r_candi_up"], PARAMS["size_lc_small"][0],
                                       debug=False)
        dist_rgb = np.array(PARAMS["dist_rgb"]).astype(np.float32).reshape((1, 5))
        dist_lc = np.array(PARAMS["dist_lc"]).astype(np.float32).reshape((1, 5))
        self.fw_large = FieldWarp(PARAMS["intr_rgb"], dist_rgb, PARAMS["size_rgb"],
                                  PARAMS["intr_lc"], dist_lc, PARAMS["size_lc"],
                                  PARAMS["rTc"], PARAMS["name"], PARAMS["device"])
        self.fw_small = FieldWarp(PARAMS["intr_rgb_small"], dist_rgb, PARAMS["size_rgb_small"],
                                  PARAMS["intr_lc_small"], dist_lc, PARAMS["size_lc_small"],
                                  PARAMS["rTc"], PARAMS["name"], PARAMS["device"])
        self.d_candi = PARAMS["d_candi"]
        self.r_candi = PARAMS["r_candi"]
        self.d_candi_up = PARAMS["d_candi_up"]
        self.r_candi_up = PARAMS["r_candi_up"]
        self.PARAMS['cTr'] = np.linalg.inv(PARAMS["rTc"])
        self.expand_A = PARAMS["expand_A"]
        self.expand_B = PARAMS["expand_B"]
        self.initialized = True
        self.sensed_arr = None
        self.device = PARAMS["device"]

        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))) or \
        self.PARAMS["size_rgb"][0] != self.PARAMS["size_lc"][0] or \
        self.PARAMS["size_rgb"][1] != self.PARAMS["size_lc"][1]:
            self.transform_needed = True
        else:
            self.transform_needed = False

    def expand_params(self, PARAMS, cfg, expand_A, expand_B):
        d_candi_expand = img_utils.powerf(cfg.var.d_min, cfg.var.d_max, expand_A, cfg.var.qpower)
        d_candi_expand_upsample = img_utils.powerf(cfg.var.d_min, cfg.var.d_max, expand_B, cfg.var.qpower)
        PARAMS["d_candi"] = d_candi_expand
        PARAMS["r_candi"] = PARAMS["d_candi"]
        PARAMS["d_candi_up"] = d_candi_expand_upsample
        PARAMS["r_candi_up"] = PARAMS["d_candi_up"]
        PARAMS["expand_A"] = expand_A
        PARAMS["expand_B"] = expand_B
        return PARAMS

    def gen_params_from_model_input(self, model_input):
        PARAMS = {
            "intr_rgb": model_input["intrinsics_up"][0, :, :].cpu().numpy(),
            "dist_rgb": [0., 0., 0., 0., 0.],
            "size_rgb": [model_input["rgb"].shape[4], model_input["rgb"].shape[3]],
            "intr_lc": model_input["intrinsics_up"][0, :, :].cpu().numpy(),
            "dist_lc": [0., 0., 0., 0., 0.],
            "size_lc": [model_input["rgb"].shape[4], model_input["rgb"].shape[3]],
            "rTc": np.array([
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ]).astype(np.float32),
            "lTc": np.array([
                [1., 0., 0., 0.2],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ]).astype(np.float32),
            "laser_fov": 80.,
            "d_candi": model_input["d_candi"],
            "r_candi": model_input["d_candi"],
            "d_candi_up": model_input["d_candi_up"],
            "r_candi_up": model_input["d_candi_up"],
            "name": "default"
        }
        PARAMS['laser_timestep'] = 3.5e-5
        PARAMS["device"] = model_input["intrinsics"].device

        return PARAMS

    def plan_default_high(self, field, cfg):
        return list(self.plan_default(field, self.planner_large, self.fw_large, "high", cfg, yield_mode=False))[0]

    def plan_default_low(self, field, cfg):
        return list(self.plan_default(field, self.planner_small, self.fw_small, "low", cfg, yield_mode=False))[0]

    def plan_m1_high(self, field, cfg):
        return list(self.plan_m1(field, self.planner_large, self.fw_large, "high", cfg, yield_mode=False))[0]

    def plan_m1_low(self, field, cfg):
        return list(self.plan_m1(field, self.planner_small, self.fw_small, "low", cfg, yield_mode=False))[0]

    def plan_sweep_high(self, field, cfg):
        return list(self.plan_sweep(field, self.planner_large, self.fw_large, "high", cfg, yield_mode=False))[0]

    def plan_empty_high(self, field, cfg):
        return list(self.plan_empty(field, self.planner_large, self.fw_large, "high", cfg, yield_mode=False))[0]

    def plan_empty_low(self, field, cfg):
        return list(self.plan_empty(field, self.planner_small, self.fw_small, "low", cfg, yield_mode=False))[0]

    def plan_empty(self, field, planner, fw, kw, cfg, yield_mode):
        start = time.time()
        #fw = self.fw_large
        #planner = self.planner_large

        fw.load_flowfield()

        # Fix Weird Side bug
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

        # Preprocess to the right size
        field_preprocessed = fw.preprocess(field, self.d_candi, self.d_candi_up)

        # Apply smoothing Kernel
        mean_kernel = create_mean_kernel(5)
        mean_kernel["weight"] = mean_kernel["weight"].to(field.device)
        field_preprocessed = F.conv2d(field_preprocessed.unsqueeze(0).unsqueeze(0), **mean_kernel).squeeze(0).squeeze(0)

        # Transform RGB to LC
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
            field_preprocessed = fw.transformZTheta(field_preprocessed, self.d_candi_up, self.d_candi_up, "transform_" + kw)

        # Normalize 0 to 1
        field_preprocessed = normalize(field_preprocessed.unsqueeze(0)).squeeze(0)
        #field_preprocessed = field_preprocessed*0
        #field_preprocessed[50:55, :] = 1.

        # Warp from Z to theta
        field_preprocessed_range = fw.ztheta2zrange_output(field_preprocessed, self.d_candi_up, self.r_candi_up,
                                                          "z2rwarp_" + kw)

        fw.save_flowfields()

        # Fix Weird Side bug
        field_preprocessed_range[:, 0] = field_preprocessed_range[:, 1]
        field_preprocessed_range[:, -1] = field_preprocessed_range[:, -2]

        # Generate CV
        field_visual = np.repeat(field_preprocessed_range.cpu().numpy()[:, :, np.newaxis], 3, axis=2)

        self.field_visual = field_visual
        if not yield_mode:
            yield(pts_planned_all, field_visual)

    def plan_sweep(self, field, planner, fw, kw, cfg, yield_mode):
        start = time.time()
        #fw = self.fw_large
        #planner = self.planner_large

        fw.load_flowfield()

        # Fix Weird Side bug
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

        # Preprocess to the right size
        field_preprocessed = fw.preprocess(field, self.d_candi, self.d_candi_up)

        # Apply smoothing Kernel
        mean_kernel = create_mean_kernel(5)
        mean_kernel["weight"] = mean_kernel["weight"].to(field.device)
        field_preprocessed = F.conv2d(field_preprocessed.unsqueeze(0).unsqueeze(0), **mean_kernel).squeeze(0).squeeze(0)

        # Transform RGB to LC
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
            field_preprocessed = fw.transformZTheta(field_preprocessed, self.d_candi_up, self.d_candi_up, "transform_" + kw)

        # Normalize 0 to 1
        field_preprocessed = normalize(field_preprocessed.unsqueeze(0)).squeeze(0)
        #field_preprocessed = field_preprocessed*0
        #field_preprocessed[50:55, :] = 1.

        # Warp from Z to theta
        field_preprocessed_range = fw.ztheta2zrange_output(field_preprocessed, self.d_candi_up, self.r_candi_up,
                                                          "z2rwarp_" + kw)

        fw.save_flowfields()

        # Fix Weird Side bug
        field_preprocessed_range[:, 0] = field_preprocessed_range[:, 1]
        field_preprocessed_range[:, -1] = field_preprocessed_range[:, -2]

        # Generate CV
        field_visual = np.repeat(field_preprocessed_range.cpu().numpy()[:, :, np.newaxis], 3, axis=2)

        pts_planned_all = []
        # THIS Z VAL NEEDS TO BE BASED ON D_CANDI!!!
        #stop
        for z in np.arange(cfg["start"], cfg["end"], cfg["step"]):
            pts_planned = self.get_flat(z)
            if yield_mode: yield(pts_planned)
            pts_planned_all.append(pts_planned)

            # Draw
            pixels = np.array([np.digitize(pts_planned[:, 1], self.d_candi_up) - 1, range(0, pts_planned.shape[0])]).T
            indices = (pixels[:,0] < 256) & (pixels[:,0] >= 0) & (pixels[:,1] >= 0) & (pixels[:,1] < field_visual.shape[1])

            # print(field_visual.shape)
            #
            pixels = pixels[indices]
            # print(pixels.shape)
            # stop

            field_visual[pixels[:, 0], pixels[:, 1], :] = [1, 0, 1]

        self.field_visual = field_visual
        if not yield_mode:
            yield(pts_planned_all, field_visual)

    def plan_m1(self, field, planner, fw, kw, cfg, yield_mode):
        start = time.time()
        #fw = self.fw_large
        #planner = self.planner_large

        fw.load_flowfield()

        # Fix Weird Side bug
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

        # Force fields to back
        for i in range(0, field.shape[1]):
            ray_dist = field[:, i]
            if(torch.isnan(ray_dist).any()):
                field[:, i] = img_utils.epsilon
                field[-1, i] = 1.
        #field[torch.isnan(field)] = img_utils.epsilon

        # Preprocess to the right size
        field_preprocessed = fw.preprocess(field, self.d_candi, self.d_candi_up)

        # Apply smoothing Kernel
        mean_kernel = create_mean_kernel(5)
        mean_kernel["weight"] = mean_kernel["weight"].to(field.device)
        field_preprocessed = F.conv2d(field_preprocessed.unsqueeze(0).unsqueeze(0), **mean_kernel).squeeze(0).squeeze(0)

        # Transform RGB to LC
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
            field_preprocessed = fw.transformZTheta(field_preprocessed, self.d_candi_up, self.d_candi_up, "transform_" + kw)

        # Normalize 0 to 1
        #field_preprocessed = normalize(field_preprocessed.unsqueeze(0)).squeeze(0)
        #field_preprocessed[torch.isnan(field_preprocessed)] = img_utils.epsilon
        #field_preprocessed = field_preprocessed*0
        #field_preprocessed[50:55, :] = 1.

        # Warp from Z to theta
        field_preprocessed_range = fw.ztheta2zrange_output(field_preprocessed, self.d_candi_up, self.r_candi_up,
                                                          "z2rwarp_" + kw)

        fw.save_flowfields()

        # Fix Weird Side bug
        field_preprocessed_range[:, 0] = field_preprocessed_range[:, 1]
        field_preprocessed_range[:, -1] = field_preprocessed_range[:, -2]

        # Generate CV
        field_visual = np.repeat(normalize(field_preprocessed.unsqueeze(0)).squeeze(0).cpu().numpy()[:, :, np.newaxis], 3, axis=2)

        #########

        # Plan
        pts_main = planner.get_design_points(field_preprocessed_range.cpu().numpy())
        if yield_mode: yield(pts_main)

        pixels = np.array([np.digitize(pts_main[:, 1], self.d_candi_up) - 1, range(0, pts_main.shape[0])]).T
        field_visual[pixels[:, 0], pixels[:, 1], :] = [1, 0, 1]

        if yield_mode:
            yield pts_main

        # Try few times
        pts_planned_all = [pts_main]
        for i in range(0, cfg["step"]):

            # Copy
            field_towork = field_preprocessed_range.cpu().numpy()

            # Through each ray sample pt
            field_preprocessed_range_temp = field_towork.copy()
            #field_preprocessed_range_temp[field_preprocessed_range_temp < 0.05] = 1e-5

            # # Nan Test
            # if np.isnan(field_preprocessed_range_temp).any():
            #     raise Exception("NAN Found")

            # Sample Based Strategy
            sampled_vals = []
            sampled_pts = []
            for c in range(0, field_preprocessed_range_temp.shape[1], cfg["interval"]):
                ray_dist = field_preprocessed_range_temp[:,c]
                ray_dist[np.isnan(ray_dist)] = 1e-5
                ray_dist = ray_dist / np.sum(ray_dist)
                sampled = np.random.choice(ray_dist.shape[0], 1, p=ray_dist)
                sampled_vals.append(sampled[0])
                sampled_pts.append([c, sampled[0], 5.])

            # Variance based? Compute var/mean for each ray then pick extent

            # Generate Spline?
            sampled_pts = np.array(sampled_pts).astype(np.float32)
            splineParams = pylc.SplineParamsVec()
            spline = pylc.fitBSpline(sampled_pts, splineParams, True).astype(np.long)
            spline = spline[(spline[:,0] >= 0) & (spline[:,0] < field_visual.shape[1])
                            & (spline[:,1] >= 0) & (spline[:,1] < field_visual.shape[0])]

            # Draw Spline
            # for spixel in spline: field_visual[spixel[1], spixel[0]] = (0,1,1)

            # Create Empty Field
            empty_field = field_preprocessed_range_temp*0
            empty_field[spline[:, 1], spline[:, 0]] = 1.

            # N = 5
            # xdir_gauss = cv2.getGaussianKernel(N, 1.0).astype(np.float32)
            # kernel = {'weight': torch.tensor(np.multiply(xdir_gauss.T, xdir_gauss)).unsqueeze(0).unsqueeze(0), 'padding': N // 2}
            # for i in range(0, 4):
            #     kernel = F.conv2d(kernel["weight"], **kernel)
            #     kernel = {'weight': kernel, 'padding': N // 2}
            # empty_field = F.conv2d(torch.tensor(empty_field).unsqueeze(0).unsqueeze(0), **kernel).squeeze(0).squeeze(0)
            # empty_field = empty_field.numpy()
            for i in range(0, 3):
                empty_field = cv2.GaussianBlur(empty_field, (5, 5), 1)

            # Fuse
            empty_field = empty_field / np.sum(empty_field, axis=0)
            multiply = field_towork * empty_field
            field_towork = multiply / np.sum(multiply, axis=0)
            field_towork[np.isnan(field_towork)] = 0.

            #end = time.time()
            #print(end-start)

            # Plan
            pts_planned = planner.get_design_points(field_towork)
            pts_planned_all.append(pts_planned)
            if yield_mode: yield(pts_planned)

            # Draw
            pixels = np.array([np.digitize(pts_planned[:, 1], self.d_candi_up) - 1, range(0, pts_planned.shape[0])]).T
            field_visual[pixels[:, 0], pixels[:, 1], :] = [1, 0, 1]

            # Draw Pixels
            #for spixel in pixels: cv2.circle(field_visual, (spixel[1], spixel[0]), 1, (0, 255, 255), -1)

        self.field_visual = field_visual
        if not yield_mode:
            yield(pts_planned_all, field_visual)

    def plan_default(self, field, planner, fw, kw, cfg, yield_mode):
        # cv2.imshow("field", field.cpu().numpy())

        start = time.time()

        fw.load_flowfield()

        # Fix Weird Side bug
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

        # Force field to back
        for i in range(0, field.shape[1]):
            ray_dist = field[:, i]
            if(torch.isnan(ray_dist).any()):
                field[:, i] = img_utils.epsilon
                field[-1, i] = 1.
        #field[torch.isnan(field)] = img_utils.epsilon

        # Preprocess to the right size
        field_preprocessed = fw.preprocess(field, self.d_candi, self.d_candi_up)

        # Apply smoothing Kernel
        mean_kernel = create_mean_kernel(5)
        mean_kernel["weight"] = mean_kernel["weight"].to(field.device)
        field_preprocessed = F.conv2d(field_preprocessed.unsqueeze(0).unsqueeze(0), **mean_kernel).squeeze(0).squeeze(0)

        # Transform RGB to LC
        #cv2.imshow("A", field_preprocessed.cpu().numpy()*100)
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
            field_preprocessed = fw.transformZTheta(field_preprocessed, self.d_candi_up, self.d_candi_up, "transform_" + kw)
        #cv2.imshow("B", field_preprocessed.cpu().numpy()*100)

        # Normalize 0 to 1
        field_preprocessed = normalize(field_preprocessed.unsqueeze(0)).squeeze(0)
        field_preprocessed[torch.isnan(field_preprocessed)] = img_utils.epsilon

        # Warp from Z to theta
        field_preprocessed_range = fw.ztheta2zrange_output(field_preprocessed, self.d_candi_up, self.r_candi_up,
                                                          "z2rwarp_" + kw)

        # Main Field
        pts_main = planner.get_design_points(field_preprocessed_range.cpu().numpy())
        if yield_mode: yield(pts_main)
        field_visual = np.repeat(field_preprocessed.cpu().numpy()[:, :, np.newaxis], 3, axis=2)
        pixels = np.array([np.digitize(pts_main[:, 1], self.d_candi_up) - 1, range(0, pts_main.shape[0])]).T
        field_visual[pixels[:, 0], pixels[:, 1], :] = [1, 0, 0]
        all_pts = [pts_main]

        # Generate Peak Fields
        left_field = field_preprocessed_range.clone()
        right_field = field_preprocessed_range.clone()
        values, indices = torch.max(field_preprocessed_range, 0)
        # Extremely slow needs to be in CUDA (Takes 30ms?)
        for i in range(0, indices.shape[0]):
            maxind = indices[i]
            left_field[0:maxind, i] = 1.
            right_field[maxind:len(self.r_candi_up), i] = 1.

        for pval in cfg["step"]:
            # Invert the Fields
            left_field_inv = invert(left_field, p=pval)
            right_field_inv = invert(right_field, p=pval)

            #combined_test = (left_field_inv + right_field_inv).cpu().numpy()
            #cv2.imshow("win", combined_test); cv2.waitKey(0)

            # Plan
            pts_up = planner.get_design_points(left_field_inv.cpu().numpy())
            if yield_mode: yield(pts_up)
            pts_down = planner.get_design_points(right_field_inv.cpu().numpy())
            if yield_mode: yield(pts_down)

            # Visual
            pixels = np.array([np.digitize(pts_up[:, 1], self.d_candi_up) - 1, range(0, pts_up.shape[0])]).T
            field_visual[pixels[:, 0], pixels[:, 1], :] = [0, 1, 0]
            pixels = np.array([np.digitize(pts_down[:, 1], self.d_candi_up) - 1, range(0, pts_down.shape[0])]).T
            field_visual[pixels[:, 0], pixels[:, 1], :] = [0, 1, 0]

            # Append
            all_pts.append(pts_up)
            all_pts.append(pts_down)

        fw.save_flowfields()

        #print("Plan: " + str(time.time() - start))
        # cv2.imshow("field", field.cpu().numpy())
        # cv2.imshow("field_preprocessed", field_preprocessed.cpu().numpy())
        # cv2.imshow("field_visual", field_visual)
        # cv2.waitKey(0)

        self.field_visual = field_visual
        if not yield_mode:
            yield(all_pts, field_visual)

    def sense_low(self, depth_rgb, design_pts_lc, visualizer=None):
        start = time.time()

        # Warp depthmap to LC frame
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
            pts_rgb = img_utils.depth_to_pts(torch.Tensor(depth_rgb).unsqueeze(0), self.PARAMS['intr_rgb'])
            pts_rgb = pts_rgb.reshape((pts_rgb.shape[0], pts_rgb.shape[1] * pts_rgb.shape[2]))
            pts_rgb = torch.cat([pts_rgb, torch.ones(1, pts_rgb.shape[1])])
            pts_rgb = pts_rgb.numpy().T
            thick_rgb = np.ones((pts_rgb.shape[0], 1)).astype(np.float32)
            depth_lc, _, _ = pylc.transformPoints(pts_rgb, thick_rgb, self.PARAMS['intr_lc_small'], self.PARAMS['cTr'],
                                                  self.PARAMS['size_lc_small'][0], self.PARAMS['size_lc_small'][1],
                                                  {"filtering": 0})
        else:
            depth_lc = depth_rgb

        # Sense
        output_lc, thickness_lc = self.lightcurtain_small.get_return(depth_lc, design_pts_lc, True)
        output_lc[np.isnan(output_lc[:, :, 0])] = 0
        thickness_lc[np.isnan(thickness_lc[:, :])] = 0

        # Warp output to RGB frame
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
            pts = output_lc.reshape((output_lc.shape[0] * output_lc.shape[1], 4))
            thickness = thickness_lc.flatten()
            depth_sensed, int_sensed, thickness_sensed = pylc.transformPoints(pts, thickness, self.PARAMS['intr_rgb_small'],
                                                                              self.PARAMS['rTc'],
                                                                              self.PARAMS['size_rgb_small'][0],
                                                                              self.PARAMS['size_rgb_small'][1],
                                                                              {"filtering": 0})
        else:
            int_sensed = output_lc[:, :, 3]
            depth_sensed = output_lc[:, :, 2]
            thickness_sensed = thickness_lc

        # Transfer to CUDA (How to know device?)
        mask_sense = torch.tensor(depth_rgb > 0).float().to(self.device)
        depth_sensed = torch.tensor(depth_sensed).to(self.device) * mask_sense
        thickness_sensed = torch.tensor(thickness_sensed).to(self.device) * mask_sense
        int_sensed = torch.tensor(int_sensed).to(self.device) * mask_sense

        # Compute DPV
        z_img = depth_sensed
        int_img = int_sensed / 255.
        unc_img = (thickness_sensed / 10.) ** 2
        A = mapping(int_img)
        # Try fucking with 1 in the 1-A value
        DPV = mixed_model(self.d_candi, z_img, unc_img, A, 1. - A)

        # Generate XYZ version for viz
        output_rgb = None
        if visualizer:
            pts_sensed = img_utils.depth_to_pts(depth_sensed.unsqueeze(0), self.PARAMS['intr_rgb_small']).cpu()
            output_rgb = np.zeros(output_lc.shape).astype(np.float32)
            output_rgb[:, :, 0] = pts_sensed[0, :, :]
            output_rgb[:, :, 1] = pts_sensed[1, :, :]
            output_rgb[:, :, 2] = pts_sensed[2, :, :]
            output_rgb[:, :, 3] = int_sensed.cpu()
            output_rgb[np.isnan(output_rgb[:, :, 0])] = 0

        return DPV, output_rgb

        # Generate XYZ version for viz
        # pts_sensed = util.depth_to_pts(torch.Tensor(depth_sensed).unsqueeze(0), self.PARAMS['intr_rgb'])
        # output_rgb = np.zeros(output_lc.shape).astype(np.float32)
        # output_rgb[:, :, 0] = pts_sensed[0, :, :]
        # output_rgb[:, :, 1] = pts_sensed[1, :, :]
        # output_rgb[:, :, 2] = pts_sensed[2, :, :]
        # output_rgb[:, :, 3] = int_sensed

        # cv2.imshow("depth_rgb", depth_rgb / 100.)
        # cv2.imshow("depth_lc", depth_lc / 100.)
        # cv2.imshow("depth_sensed_orig", output_lc[:,:,2] / 100.)
        # cv2.imshow("int_sensed_orig", output_lc[:, :, 3] / 255.)
        # cv2.imshow("depth_sensed", depth_sensed/100.)
        # cv2.imshow("int_sensed", int_sensed/255.)
        # cv2.waitKey(0)


    def sense_high(self, depth_rgb, design_pts_lc, visualizer=None):
        start = time.time()

        # Warp depthmap to LC frame
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
            pts_rgb = img_utils.depth_to_pts(torch.Tensor(depth_rgb).unsqueeze(0), self.PARAMS['intr_rgb'])
            pts_rgb = pts_rgb.reshape((pts_rgb.shape[0], pts_rgb.shape[1] * pts_rgb.shape[2]))
            pts_rgb = torch.cat([pts_rgb, torch.ones(1, pts_rgb.shape[1])])
            pts_rgb = pts_rgb.numpy().T
            thick_rgb = np.ones((pts_rgb.shape[0], 1)).astype(np.float32)
            uniform_params = {"filtering": 2}
            depth_lc, _, _ = pylc.transformPoints(pts_rgb, thick_rgb, self.PARAMS['intr_lc'], self.PARAMS['cTr'],
                                                  self.PARAMS['size_lc'][0], self.PARAMS['size_lc'][1],
                                                  uniform_params)

            # # Pool Depth
            # pool_val = 2
            # depth_lc = img_utils.minpool(torch.Tensor(depth_lc).unsqueeze(0).unsqueeze(0), pool_val, 1000).squeeze(0).squeeze(0).numpy()
            # depth_lc = cv2.resize(depth_lc, (0,0), fx=pool_val, fy=pool_val, interpolation = cv2.INTER_NEAREST)
        else:
            depth_lc = depth_rgb

        # cv2.imshow("depth_rgb", depth_rgb/100.)
        # cv2.imshow("depth_lc", depth_lc/100.)

        # Sense
        output_lc, thickness_lc = self.lightcurtain_large.get_return(depth_lc, design_pts_lc, True)
        output_lc[np.isnan(output_lc[:, :, 0])] = 0
        thickness_lc[np.isnan(thickness_lc[:, :])] = 0

        # Warp output to RGB frame
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
            pts = output_lc.reshape((output_lc.shape[0] * output_lc.shape[1], 4))
            thickness = thickness_lc.flatten()
            depth_sensed, int_sensed, thickness_sensed = pylc.transformPoints(pts, thickness, self.PARAMS['intr_rgb'],
                                                                              self.PARAMS['rTc'],
                                                                              self.PARAMS['size_rgb'][0],
                                                                              self.PARAMS['size_rgb'][1],
                                                                              {"filtering": 0})
        else:
            int_sensed = output_lc[:, :, 3]
            depth_sensed = output_lc[:, :, 2]
            thickness_sensed = thickness_lc

        # depth_lc_x = output_lc[:, :, 2]
        # depth_rgb_x = depth_sensed
        # cv2.imshow("depth_rgb_x", depth_rgb_x/100.)
        # cv2.imshow("depth_lc_x", depth_lc_x/100.)

        # Transfer to CUDA (How to know device?)
        mask_sense = torch.tensor(depth_rgb > 0).float().to(self.device)
        depth_sensed = torch.tensor(depth_sensed).to(self.device) * mask_sense
        thickness_sensed = torch.tensor(thickness_sensed).to(self.device) * mask_sense
        int_sensed = torch.tensor(int_sensed).to(self.device) * mask_sense

        # Compute DPV
        z_img = depth_sensed
        int_img = int_sensed / 255.
        unc_img = (thickness_sensed / 10.) ** 2
        A = mapping(int_img)
        # Try fucking with 1 in the 1-A value
        DPV = mixed_model(self.d_candi, z_img, unc_img, A, 1. - A)

        # Save information at pixel wanted
        pixel_wanted = [150, 66]
        debug_data = dict()
        # debug_data["gt"] = depth_rgb[pixel_wanted[0], pixel_wanted[1]]
        # debug_data["z_img"] = z_img[pixel_wanted[0], pixel_wanted[1]]
        # debug_data["int_img"] = int_img[pixel_wanted[0], pixel_wanted[1]]
        # debug_data["thickness_sensed"] = thickness_sensed[pixel_wanted[0], pixel_wanted[1]]
        # debug_data["dist"] = DPV[:, pixel_wanted[0], pixel_wanted[1]].cpu().numpy()

        # Generate XYZ version for viz
        output_rgb = None
        if visualizer:
            pts_sensed = img_utils.depth_to_pts(depth_sensed.unsqueeze(0), self.PARAMS['intr_rgb']).cpu()
            output_rgb = np.zeros((depth_sensed.shape[0], depth_sensed.shape[1], 4)).astype(np.float32)
            output_rgb[:, :, 0] = pts_sensed[0, :, :]
            output_rgb[:, :, 1] = pts_sensed[1, :, :]
            output_rgb[:, :, 2] = pts_sensed[2, :, :]
            output_rgb[:, :, 3] = int_sensed.cpu()
            output_rgb[np.isnan(output_rgb[:, :, 0])] = 0

        return DPV, output_rgb, debug_data

        # Generate XYZ version for viz
        # pts_sensed = util.depth_to_pts(torch.Tensor(depth_sensed).unsqueeze(0), self.PARAMS['intr_rgb'])
        # output_rgb = np.zeros(output_lc.shape).astype(np.float32)
        # output_rgb[:, :, 0] = pts_sensed[0, :, :]
        # output_rgb[:, :, 1] = pts_sensed[1, :, :]
        # output_rgb[:, :, 2] = pts_sensed[2, :, :]
        # output_rgb[:, :, 3] = int_sensed

        # cv2.imshow("depth_rgb", depth_rgb / 100.)
        # cv2.imshow("depth_lc", depth_lc / 100.)
        # cv2.imshow("depth_sensed_orig", output_lc[:,:,2] / 100.)
        # cv2.imshow("int_sensed_orig", output_lc[:, :, 3] / 255.)
        # cv2.imshow("depth_sensed", depth_sensed/100.)
        # cv2.imshow("int_sensed", int_sensed/255.)
        # cv2.waitKey(0)

    def sense_real_batch(self, depth_lc, design_pts_lc_arr):

        pass

        # Sense (Replace with Real Sensor)
        start = time.time()
        output_lc_arr = []
        thickness_lc_arr = []
        for design_pts_lc in design_pts_lc_arr:
            output_lc, thickness_lc = self.lightcurtain_large.get_return(depth_lc, design_pts_lc, True)
            output_lc[np.isnan(output_lc[:, :, 0])] = 0
            thickness_lc[np.isnan(thickness_lc[:, :])] = 0     
            output_lc_arr.append(output_lc.reshape((output_lc.shape[0] * output_lc.shape[1], 4)))
            thickness_lc_arr.append(thickness_lc.flatten())
        sense_time = time.time() - start

        
        # Warp to RGB
        start = time.time()
        sensed_arr = pylc.transformPointsBatch(output_lc_arr, thickness_lc_arr, self.PARAMS['intr_rgb'],
                                         self.PARAMS['rTc'],
                                         self.PARAMS['size_rgb'][0],
                                         self.PARAMS['size_rgb'][1],
                                         {"filtering": 0})
        warp_time = time.time() - start

        # Put in CUDA
        start = time.time()
        depth_sensed = torch.tensor(sensed_arr[0]).cuda()
        int_sensed = torch.tensor(sensed_arr[1]).cuda()
        thickness_sensed = torch.tensor(sensed_arr[2]).cuda()
        mask_sensed = torch.tensor(depth_sensed > 0).float().cuda()
        cuda_time = time.time() - start

        # Compute DPV
        start = time.time()
        z_img = depth_sensed
        int_img = int_sensed / 255.
        unc_img = (thickness_sensed / 10.) ** 2
        A = mapping(int_img)
        DPVs = []
        for i in range(0, A.shape[0]):
            DPV = mixed_model(self.d_candi, z_img[i,:,:], unc_img[i,:,:], A[i,:,:], 1. - A[i,:,:])
            DPVs.append(DPV)
        dpv_time = time.time() - start

        #mixed_model(self.d_candi, z_img, unc_img, A, 1. - A)


        print("---")
        print(" sense_time: " + str(sense_time))
        print(" warp_time: " + str(warp_time))
        print(" cuda_time: " + str(cuda_time))
        print(" dpv_time: " + str(dpv_time))

        return DPVs

    def sense_real(self, depth_lc, design_pts_lc, lc_wrapper=None):
        start = time.time()

        """
        Can we speed up this processing somehow so it takes in multiple points?
        No point optimizing for the get_return part

        The Warping can be done with the warp tool? Or just do using openmp?
        Warp tool too difficult due to scaling etc.
        """

        # Sense (Replace with Real Sensor)
        if lc_wrapper is not None:
            start = time.time()
            output_lc, thickness_lc = lc_wrapper.sendAndWait(design_pts_lc)
            output_lc[np.isnan(output_lc[:, :, 0])] = 0
            thickness_lc[np.isnan(thickness_lc[:, :])] = 0
            time_sense = time.time() - start
        else:
            start = time.time()
            output_lc, thickness_lc = self.lightcurtain_large.get_return(depth_lc, design_pts_lc, True)
            output_lc[np.isnan(output_lc[:, :, 0])] = 0
            thickness_lc[np.isnan(thickness_lc[:, :])] = 0
            time_sense = time.time() - start

        # Warp output to RGB frame
        start = time.time()
        if not np.all(np.equal(self.PARAMS["rTc"], np.eye(4))):
            pts = output_lc.reshape((output_lc.shape[0] * output_lc.shape[1], 4))
            thickness = thickness_lc.flatten()
            depth_sensed, int_sensed, thickness_sensed = pylc.transformPoints(pts, thickness, self.PARAMS['intr_rgb'],
                                                                              self.PARAMS['rTc'],
                                                                              self.PARAMS['size_rgb'][0],
                                                                              self.PARAMS['size_rgb'][1],
                                                                              {"filtering": 0})
        else:
            int_sensed = output_lc[:, :, 3]
            depth_sensed = output_lc[:, :, 2]
            thickness_sensed = thickness_lc
        time_warp = time.time() - start

        # depth_lc_x = output_lc[:, :, 2]
        # depth_rgb_x = depth_sensed
        # cv2.imshow("depth_rgb_x", depth_rgb_x/100.)
        # cv2.imshow("depth_lc_x", depth_lc_x/100.)

        # Transfer to CUDA (How to know device?)
        start = time.time()
        mask_sense = torch.tensor(depth_sensed > 0).float().cuda()
        depth_sensed = torch.tensor(depth_sensed).cuda() * mask_sense
        thickness_sensed = torch.tensor(thickness_sensed).cuda() * mask_sense
        int_sensed = torch.tensor(int_sensed).cuda() * mask_sense

        # Div
        if lc_wrapper is not None:
            std_div = 10.
        else:
            std_div = 10.

        # Compute DPV
        z_img = depth_sensed
        int_img = int_sensed / 255.
        unc_img = (thickness_sensed / std_div) ** 2
        A = mapping(int_img)
        # Try fucking with 1 in the 1-A value
        DPV = mixed_model(self.d_candi, z_img, unc_img, A, 1. - A)
        time_dpv = time.time() - start

        print("---")
        print(" time_sense: " + str(time_sense))
        print(" time_warp: " + str(time_warp))
        print(" time_dpv: " + str(time_dpv))

        return DPV, None, None

        # Generate XYZ version for viz
        # pts_sensed = util.depth_to_pts(torch.Tensor(depth_sensed).unsqueeze(0), self.PARAMS['intr_rgb'])
        # output_rgb = np.zeros(output_lc.shape).astype(np.float32)
        # output_rgb[:, :, 0] = pts_sensed[0, :, :]
        # output_rgb[:, :, 1] = pts_sensed[1, :, :]
        # output_rgb[:, :, 2] = pts_sensed[2, :, :]
        # output_rgb[:, :, 3] = int_sensed

        # cv2.imshow("depth_rgb", depth_rgb / 100.)
        # cv2.imshow("depth_lc", depth_lc / 100.)
        # cv2.imshow("depth_sensed_orig", output_lc[:,:,2] / 100.)
        # cv2.imshow("int_sensed_orig", output_lc[:, :, 3] / 255.)
        # cv2.imshow("depth_sensed", depth_sensed/100.)
        # cv2.imshow("int_sensed", int_sensed/255.)
        # cv2.waitKey(0)

    def transform_measurement(self, output_lc, thickness_lc):
        pts = output_lc.reshape((output_lc.shape[0] * output_lc.shape[1], 4))
        thickness = thickness_lc.flatten()
        depth_sensed, int_sensed, thickness_sensed = pylc.transformPoints(pts, thickness, self.PARAMS['intr_rgb'],
                                                                            self.PARAMS['rTc'],
                                                                            self.PARAMS['size_rgb'][0],
                                                                            self.PARAMS['size_rgb'][1],
                                                                            {"filtering": 0})

        sensed_arr = torch.tensor(np.array([depth_sensed, int_sensed, thickness_sensed]).astype(np.float32))
        if self.sensed_arr == None:
            self.sensed_arr = sensed_arr.pin_memory()
        self.sensed_arr[:] = sensed_arr[:]
        return self.sensed_arr.cuda(non_blocking=True)

    def gen_lc_dpv(self, sensed_arr, std_div):
        depth_sensed = sensed_arr[0,:,:]
        mask_sense = (depth_sensed > 0).float()
        thickness_sensed = sensed_arr[2,:,:] * mask_sense
        int_sensed = sensed_arr[1,:,:] * mask_sense

        # Compute DPV
        z_img = depth_sensed
        int_img = int_sensed / 255.
        unc_img = (thickness_sensed / std_div) ** 2
        A = mapping(int_img)
        DPV = mixed_model(self.d_candi, z_img, unc_img, A, 1. - A)

        return DPV
