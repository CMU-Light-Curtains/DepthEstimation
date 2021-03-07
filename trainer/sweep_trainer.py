import time
import torch
import copy
import json
import os
from .base_trainer import BaseTrainer
from utils.flow_utils import evaluate_flow
from utils.misc_utils import AverageMeter
import utils.img_utils as img_utils

from external.perception_lib import viewer
from kittiloader import batch_scheduler
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import sys

class SweepTrainer(BaseTrainer):
    def __init__(self, id, model, loss_func, _log, save_root, config, shared):
        super(SweepTrainer, self).__init__(id, model, loss_func, _log, save_root, config, shared)
        self.cfg = config
        self.id = id

        # JSON
        if self.id == 0:
            self.foutput = {
                "name": self.cfg.data.exp_name,
                "error": []
            }

        # Create all variables here
        self.d_candi = img_utils.powerf(self.cfg.var.d_min, self.cfg.var.d_max, self.cfg.var.ndepth,
                                        self.cfg.var.qpower)
        self.d_candi_up = img_utils.powerf(self.cfg.var.d_min, self.cfg.var.d_max, self.cfg.var.ndepth * 2,
                                           self.cfg.var.qpower)

        # Create Data Loaders over here
        if self.cfg.mp.enabled:
            total_processes = self.cfg.mp.workers
            process_num = self.id
            batch_size = int(self.cfg.train.batch_size / total_processes)
        else:
            total_processes = 1
            process_num = 0
            batch_size = self.cfg.train.batch_size
        train_loader_params = {
            "pytorch_scaling": True,
            "velodyne_depth": True,
            "dataset_path": self.cfg.data.dataset_path,
            "split_path": self.cfg.data.dataset_split,
            "total_processes": total_processes,
            "process_num": process_num,
            "t_win_r": self.cfg.var.t_win,
            "img_size": self.cfg.var.img_size,
            "crop_w": self.cfg.var.crop_w,
            "d_candi": self.d_candi,
            "d_candi_up": self.d_candi_up,
            "dataset_name": "kitti",
            "hack_num": 0,
            "batch_size": batch_size,
            "n_epoch": 1,
            "qmax": 3,
            "mode": "train",
            "cfg": self.cfg
        }
        val_loader_params = copy.deepcopy(train_loader_params)
        val_loader_params["mode"] = "val"
        self._log.info(self.id, "=> Setting Batch Size {}.".format(batch_size))
        self.train_loader = batch_scheduler.BatchSchedulerMP(train_loader_params, self.cfg.var.mload)
        self.val_loader = batch_scheduler.BatchSchedulerMP(val_loader_params, self.cfg.var.mload)
        self.prev_output = None
        self.prev_lc = None
        self.first_run = True

    def __del__(self):
        pass

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()
        key_meter_names = ['Loss', 'l_ph', 'l_sm', 'flow_mean']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        # Zero out shared
        self.shared[:] = 0

        # Model Train
        self.model.train()

        # Iterate Train Loader
        early_stop = False
        for items in self.train_loader.enumerate():
            if early_stop: break

            # ** Signal **
            if self.cfg.mp.enabled:
                signal = torch.tensor([1]).to(self.device)
                dist.all_reduce(signal)
                if signal.item() < self.cfg.mp.workers:
                    #self._log.info(self.id, "EXIT: " + str(signal.item()))
                    self.train_loader.stop()
                    early_stop = True
                    continue

            # Get data
            local_info, batch_length, batch_idx, frame_count, frame_length, iepoch = items
            if local_info['is_valid'].sum() == 0:
                loader_str = 'Corrupted Data! batch %d / %d, iter: %d, frame_count: %d / %d; Epoch: %d / %d' \
                % (batch_idx + 1, batch_length, self.i_iter, frame_count + 1, frame_length, self.i_epoch + 1, self.cfg.train.epoch_num)
                self._log.info(self.id, loader_str)
                continue

            # Reset Stage
            if frame_count == 0:
                self._log.info(self.id, "Reset Previous Output")
                self.prev_output = {"left": None, "right": None}

            # Create inputs
            local_info["d_candi"] = self.d_candi
            local_info["d_candi_up"] = self.d_candi_up
            if "stereo" in self.cfg["var"]:
                model_input_left, gt_input_left = batch_scheduler.generate_stereo_input(self.id, local_info, self.cfg, "left")
                model_input_right, gt_input_right = batch_scheduler.generate_stereo_input(self.id, local_info, self.cfg, "right")
            else:
                model_input_left, gt_input_left = batch_scheduler.generate_model_input(self.id, local_info, self.cfg, "left")
                model_input_right, gt_input_right = batch_scheduler.generate_model_input(self.id, local_info, self.cfg, "right")
            model_input_left["prev_output"] = self.prev_output["left"]
            model_input_right["prev_output"] = self.prev_output["right"]
            model_input_left["epoch"] = self.i_epoch; model_input_right["epoch"] = self.i_epoch

            # Model
            output_left, output_right = self.model([model_input_left, model_input_right])

            # Set Prev from last one
            output_left_intp = F.interpolate(output_left["output_refined"][-1].detach(), scale_factor=0.25, mode='nearest')
            output_right_intp = F.interpolate(output_right["output_refined"][-1].detach(), scale_factor=0.25, mode='nearest')
            self.prev_output = {"left": output_left_intp, "right": output_right_intp}

            # Loss Function
            loss = self.loss_func([output_left, output_right], [gt_input_left, gt_input_right])

            # Opt
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # String
            loader_str = 'Train batch %d / %d, iter: %d, frame_count: %d / %d; Epoch: %d / %d, loss = %.5f' \
            % (batch_idx + 1, batch_length, self.i_iter, frame_count + 1, frame_length, self.i_epoch + 1, self.cfg.train.epoch_num, loss)
            self._log.info(self.id, loader_str)
            self.i_iter += 1

        # ** Signal **
        if self.cfg.mp.enabled:
            #self._log.info(self.id, "AR: " + str(signal.item()))
            if signal.item() >= self.cfg.mp.workers:
                dist.all_reduce(torch.tensor([0]).to(self.device))
            #dist.barrier()

        self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):
        batch_time = AverageMeter()
        if self.cfg.mp.enabled:
            dist.barrier()

        # Zero out shared
        self.shared[:] = 0

        # Eval Mode
        self.model.eval()

        # Variables
        errors = []

        # Iterate Train Loader
        for items in self.val_loader.enumerate():

            # Get data
            local_info, batch_length, batch_idx, frame_count, frame_length, iepoch = items
            if local_info['is_valid'].sum() == 0:
                loader_str = 'Corrupted Data! Val batch %d / %d, frame_count: %d / %d' \
                % (batch_idx + 1, batch_length, frame_count + 1, frame_length)
                self._log.info(self.id, loader_str)
                continue

            # Reset Stage
            if frame_count == 0:
                self._log.info(self.id, "Reset Previous Output")
                self.prev_output = {"left": None, "right": None}

            # Create inputs
            local_info["d_candi"] = self.d_candi
            local_info["d_candi_up"] = self.d_candi_up
            if "stereo" in self.cfg["var"]:
                model_input_left, gt_input_left = batch_scheduler.generate_stereo_input(self.id, local_info, self.cfg, "left")
            else:
                model_input_left, gt_input_left = batch_scheduler.generate_model_input(self.id, local_info, self.cfg, "left")
            model_input_left["prev_output"] = self.prev_output["left"]
            model_input_left["epoch"] = self.i_epoch # Not sure if this will work during runtime/eval

            # Model
            start = time.time()
            output_left = self.model([model_input_left])[0]
            #output_right = self.model(model_input_right)
            print("Forward: " + str(time.time() - start))

            # Set Prev
            output_left_intp = F.interpolate(output_left["output_refined"][-1].detach(), scale_factor=0.25, mode='nearest')
            self.prev_output = {"left": output_left_intp, "right": None}

            # Visualization
            if self.cfg.var.viz:
                self.visualize(model_input_left, gt_input_left, output_left)

            # Eval
            
            for b in range(0, output_left["output"][-1].shape[0]):
                output_large = output_left["output_refined"][-1][b, :, :, :]
                gt_large = gt_input_left["feat_int_tensor"][b, :, :, :]
                mask_large = gt_input_left["feat_mask_tensor"][b, :, :, :]
                depth_large = gt_input_left["dmap_imgsizes"][b, :, :]
                d_candi = torch.tensor(gt_input_left["d_candi"]).float().to(output_large.device)

                # Run Model
                mean_intensities, _ = img_utils.lc_intensities_to_dist(
                    d_candi = d_candi, 
                    placement = depth_large[:,:].unsqueeze(-1), 
                    intensity = 0, 
                    inten_sigma = output_large[1, :, :].unsqueeze(-1), # Change
                    noise_sigma = 0.1, 
                    mean_scaling = output_large[0, :, :].unsqueeze(-1)) # Change
                mean_intensities = mean_intensities.permute(2,0,1) # 128, 256, 320

                # Compute Error
                gt = gt_large[:,:,:]/255.
                pred = mean_intensities
                mask = mask_large[:,:,:].float()
                count = torch.sum(mask) + 1
                loss += (torch.sum(((gt-pred)**2)*mask) / count)*255

                errors.append(loss)

            # String
            loader_str = 'Val batch %d / %d, frame_count: %d / %d' \
            % (batch_idx + 1, batch_length, frame_count + 1, frame_length)
            self._log.info(self.id, loader_str)
            self.i_iter += 1

        # Evaluate Errors
        error = torch.mean(torch.tensor(errors))
        error_keys = ["error"]
        error_list = [error]

        # Copy to Shared
        for i, e in enumerate(error_list):
            self.shared[self.id, i] = e

        # Mean Error Compute Only First Process
        if self.cfg.mp.enabled:
            dist.barrier()
        error_list = torch.mean(self.shared, dim=0)
        if self.cfg.eval:
            print(error_list)

        # Save Model (Only First ID)
        self.save_model(error, self.cfg.data.exp_name)

        # Log
        if self.id == 0:
            json_loc = str(self.save_root) + "/" + self.cfg.data.exp_name + '.json'

            # First Run
            if self.first_run:
                # Remove Results JSON if first epoch
                if self.i_epoch == 1:
                    if os.path.exists(json_loc):
                        os.remove(json_loc)
                # If not first epoch, then load past results
                else:
                    if os.path.isfile(json_loc):
                        with open(json_loc) as f:
                            self.foutput = json.load(f)

            # Save
            for value, name in zip(error_list, error_keys):
                self.foutput[name].append(value.item())
            with open(json_loc, 'w') as f:
                json.dump(self.foutput, f)

            # Tensorboard
            if self.summary_writer is not None:
                for value, name in zip(error_list, error_keys):
                    self.summary_writer.add_scalar(name, value, self.i_epoch)
                self.summary_writer.flush()

        # Set fr
        self.first_run = False

        return error_list, error_keys
        
    def visualize(self, model_input, gt_input, output):
        import cv2

        # Eval
        for b in range(0, output["output"][-1].shape[0]):
            pass
