import time
import torch
import copy
import json
from .base_trainer import BaseTrainer
from utils.flow_utils import evaluate_flow
from utils.misc_utils import AverageMeter
import utils.img_utils as img_utils

from external.perception_lib import viewer
from kittiloader import batch_scheduler
import torch.distributed as dist

class DefaultTrainer(BaseTrainer):
    def __init__(self, id, model, loss_func, _log, save_root, config, shared):
        super(DefaultTrainer, self).__init__(id, model, loss_func, _log, save_root, config, shared)
        self.cfg = config
        self.id = id

        # JSON
        if self.id == 0:
            self.foutput = {
                "name": self.cfg.data.exp_name,
                "rmse": [], "rmse_refined": [], "sil": [], "sil_refined": []
            }

        # Visualizer
        self.viz = None
        if self.cfg.var.viz and self.id == 0:
            self.viz = viewer.Visualizer("V")
            self.viz.start()

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
            "qmax": 1,
            "mode": "train"
        }
        val_loader_params = copy.deepcopy(train_loader_params)
        val_loader_params["mode"] = "val"
        self._log.info(self.id, "=> Setting Batch Size {}.".format(batch_size))
        self.train_loader = batch_scheduler.BatchSchedulerMP(train_loader_params, self.cfg.var.mload)
        self.val_loader = batch_scheduler.BatchSchedulerMP(val_loader_params, self.cfg.var.mload)
        self.prev_output = None

        # PIN MEMORY?

    def __del__(self):
        if self.viz is not None:
            self.viz.kill_received = True

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

            # Get data
            local_info, batch_length, batch_idx, frame_count, frame_length, iepoch = items
            if local_info['is_valid'].sum() == 0:
                raise Exception("Not supposed to happen")

            # Reset Stage
            if frame_count == 0:
                self._log.info(self.id, "Reset Previous Output")
                self.prev_output = {"left": None, "right": None}

            # Create inputs
            local_info["d_candi"] = self.d_candi
            model_input_left, gt_input_left = batch_scheduler.generate_model_input(self.id, local_info, self.cfg, "left")
            model_input_right, gt_input_right = batch_scheduler.generate_model_input(self.id, local_info, self.cfg, "right")
            model_input_left["prev_output"] = self.prev_output["left"]
            model_input_right["prev_output"] = self.prev_output["right"]

            # Model
            output_left = self.model(model_input_left)
            output_right = self.model(model_input_right)

            # Set Prev from last one
            self.prev_output = {"left": output_left["output"][-1].detach(), "right": output_right["output"][-1].detach()}

            # Loss Function
            loss = self.loss_func([output_left, output_right], [gt_input_left, gt_input_right])

            # ** Signal **
            if self.cfg.mp.enabled:
                signal = torch.tensor([1]).to(self.device)
                work = dist.all_reduce(signal, async_op=True)
                work.wait()
                #self._log.info(self.id, "SIG: " + str(signal.item()))
                if signal.item() < self.cfg.mp.workers:
                    #self._log.info(self.id, "EXIT: " + str(signal.item()))
                    self.train_loader.stop()
                    early_stop = True
                    continue

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
            signal = torch.tensor([0]).to(self.device)
            sflag = self.shared[0, -1]
            if sflag == 0:
                self.shared[0, -1] = 1
                dist.all_reduce(signal)

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
        errors_refined = []

        # Iterate Train Loader
        for items in self.val_loader.enumerate():

            # Get data
            local_info, batch_length, batch_idx, frame_count, frame_length, iepoch = items
            if local_info['is_valid'].sum() == 0:
                raise Exception("Not supposed to happen")

            # Reset Stage
            if frame_count == 0:
                self._log.info(self.id, "Reset Previous Output")
                self.prev_output = {"left": None, "right": None}

            # Create inputs
            local_info["d_candi"] = self.d_candi
            model_input_left, gt_input_left = batch_scheduler.generate_model_input(self.id, local_info, self.cfg, "left")
            model_input_right, gt_input_right = batch_scheduler.generate_model_input(self.id, local_info, self.cfg, "right")
            model_input_left["prev_output"] = self.prev_output["left"]
            model_input_right["prev_output"] = self.prev_output["right"]

            # Model
            output_left = self.model(model_input_left)
            output_right = self.model(model_input_right)

            # Set Prev
            self.prev_output = {"left": output_left["output"][-1].detach(), "right": output_right["output"][-1].detach()}

            # Eval
            for b in range(0, output_left["output"][-1].shape[0]):
                dpv_predicted = output_left["output"][-1][b, :, :, :].unsqueeze(0)
                dpv_refined_predicted = output_left["output_refined"][-1][b, :, :, :].unsqueeze(0)
                depth_predicted = img_utils.dpv_to_depthmap(dpv_predicted, self.d_candi, BV_log=True)
                depth_refined_predicted = img_utils.dpv_to_depthmap(dpv_refined_predicted, self.d_candi, BV_log=True)
                mask = gt_input_left["masks"][b, :, :, :]
                mask_refined = gt_input_left["masks_imgsizes"][b, :, :, :]
                depth_truth = gt_input_left["dmaps"][b, :, :].unsqueeze(0)
                depth_refined_truth = gt_input_left["dmap_imgsizes"][b, :, :].unsqueeze(0)

                # Eval
                depth_truth_eval = depth_truth.clone()
                depth_truth_eval[depth_truth_eval >= self.d_candi[-1]] = self.d_candi[-1]
                depth_refined_truth_eval = depth_refined_truth.clone()
                depth_refined_truth_eval[depth_refined_truth_eval >= self.d_candi[-1]] = self.d_candi[-1]
                depth_predicted_eval = depth_predicted.clone()
                depth_predicted_eval = depth_predicted_eval * mask
                depth_refined_predicted_eval = depth_refined_predicted.clone()
                depth_refined_predicted_eval = depth_refined_predicted_eval * mask_refined
                errors.append(img_utils.depth_error(depth_predicted_eval.squeeze(0).cpu().numpy(), depth_truth_eval.squeeze(0).cpu().numpy()))
                errors_refined.append(img_utils.depth_error(depth_refined_predicted_eval.squeeze(0).cpu().numpy(), depth_refined_truth_eval.squeeze(0).cpu().numpy()))

            # String
            loader_str = 'Val batch %d / %d, frame_count: %d / %d' \
            % (batch_idx + 1, batch_length, frame_count + 1, frame_length)
            self._log.info(self.id, loader_str)
            self.i_iter += 1

        # Evaluate Errors
        results = img_utils.eval_errors(errors)
        results_refined = img_utils.eval_errors(errors_refined)
        sil = results["scale invariant log"][0]
        sil_refined = results_refined["scale invariant log"][0]
        rmse = results["rmse"][0]
        rmse_refined = results_refined["rmse"][0]
        error_keys = ["rmse", "rmse_refined", "sil", "sil_refined"]
        error_list = [rmse, rmse_refined, sil, sil_refined]

        # Copy to Shared
        for i, e in enumerate(error_list):
            self.shared[self.id, i] = e

        # Mean Error Compute Only First Process
        if self.cfg.mp.enabled:
            dist.barrier()
        error_list = torch.mean(self.shared, dim=0)

        # Save Model (Only First ID)
        self.save_model(rmse_refined, self.cfg.data.exp_name)

        # Log
        if self.id == 0:
            for value, name in zip(error_list, error_keys):
                self.foutput[name].append(value.item())
            with open(str(self.save_root) + "/" + self.cfg.data.exp_name + '.json', 'w') as f:
                json.dump(self.foutput, f)

        return error_list, error_keys

        # if type(self.valid_loader) is not list:
        #     self.valid_loader = [self.valid_loader]
        #
        # # only use the first GPU to run validation, multiple GPUs might raise error.
        # # https://github.com/Eromera/erfnet_pytorch/issues/2#issuecomment-486142360
        # self.model = self.model.module
        # self.model.eval()
        #
        # end = time.time()
        #
        # all_error_names = []
        # all_error_avgs = []
        #
        # n_step = 0
        # for i_set, loader in enumerate(self.valid_loader):
        #     error_names = ['EPE']
        #     error_meters = AverageMeter(i=len(error_names))
        #     for i_step, data in enumerate(loader):
        #         img1, img2 = data['img1'], data['img2']
        #         img_pair = torch.cat([img1, img2], 1).to(self.device)
        #         gt_flows = data['target']['flow'].numpy().transpose([0, 2, 3, 1])
        #
        #         # compute output
        #         flows = self.model(img_pair)['flows_fw']
        #         pred_flows = flows[0].detach().cpu().numpy().transpose([0, 2, 3, 1])
        #
        #         es = evaluate_flow(gt_flows, pred_flows)
        #         error_meters.update([l.item() for l in es], img_pair.size(0))
        #
        #         # measure elapsed time
        #         batch_time.update(time.time() - end)
        #         end = time.time()
        #
        #         if i_step % self.cfg.print_freq == 0 or i_step == len(loader) - 1:
        #             self._log.info('Test: {0}[{1}/{2}]\t Time {3}\t '.format(
        #                 i_set, i_step, self.cfg.valid_size, batch_time) + ' '.join(
        #                 map('{:.2f}'.format, error_meters.avg)))
        #
        #         if i_step > self.cfg.valid_size:
        #             break
        #     n_step += len(loader)
        #
        #     # write error to tf board.
        #     for value, name in zip(error_meters.avg, error_names):
        #         self.summary_writer.add_scalar(
        #             'Valid_{}_{}'.format(name, i_set), value, self.i_epoch)
        #
        #     all_error_avgs.extend(error_meters.avg)
        #     all_error_names.extend(['{}_{}'.format(name, i_set) for name in error_names])
        #
        # self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # # In order to reduce the space occupied during debugging,
        # # only the model with more than cfg.save_iter iterations will be saved.
        # if self.i_iter > self.cfg.save_iter:
        #     self.save_model(all_error_avgs[0] + all_error_avgs[1], name='Sintel')
        #
        # return all_error_avgs, all_error_names
