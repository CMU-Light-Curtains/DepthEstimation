import time
import torch
import copy
from .base_trainer import BaseTrainer
from utils.flow_utils import evaluate_flow
from utils.misc_utils import AverageMeter
import utils.img_utils as img_utils

from external.perception_lib import viewer
from kittiloader import batch_scheduler


class DefaultTrainer(BaseTrainer):
    def __init__(self, id, model, loss_func, _log, save_root, config):
        super(DefaultTrainer, self).__init__(id, model, loss_func, _log, save_root, config)
        self.cfg = config
        self.id = id

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
        self.train_loader = batch_scheduler.BatchSchedulerMP(train_loader_params, 1)
        self.val_loader = batch_scheduler.BatchSchedulerMP(val_loader_params, 1)

        # Create Visualizer?

    def __del__(self):
        if self.viz is not None:
            self.viz.kill_received = True

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l_ph', 'l_sm', 'flow_mean']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model.train()
        end = time.time()

        #self.i_iter += 1
        #self.i_epoch += 1

        for items in self.train_loader.enumerate():

            # Get data
            local_info, batch_length, batch_idx, frame_count, ref_indx, iepoch = items

            # String
            loader_str = 'video batch %d / %d, iter: %d, frame_count: %d; Epoch: %d / %d, loss = %.5f' \
                  % (batch_idx + 1, batch_length, 0, frame_count, iepoch + 1, 0, 0)

            self._log.info(self.id, loader_str)

            self.i_iter += 1

        self.i_epoch += 1


        # for i_step, data in enumerate(self.train_loader):
        #     if i_step > self.cfg.epoch_size:
        #         break
        #     # read data to device
        #     img1, img2 = data['img1'], data['img2']
        #     print(img1.shape)
        #     img_pair = torch.cat([img1, img2], 1).to(self.device)
        #
        #     # measure data loading time
        #     am_data_time.update(time.time() - end)
        #
        #     # torch.Size([2, 3, 384, 832])
        #     # torch.Size([2, 3, 384, 832])
        #
        #     # compute output
        #     res_dict = self.model(img_pair, with_bk=True)
        #     flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
        #     flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
        #              zip(flows_12, flows_21)]
        #
        #     # flows has 5 items for the diff scales
        #     #   each has B items for batch
        #     #        Each one is [4, H, W] It got 4xed resized concat of 12 and 21
        #     # img_pair is torch.Size([2, 6, 384, 832])
        #
        #     loss, l_ph, l_sm, flow_mean = self.loss_func(flows, img_pair)
        #
        #     # update meters
        #     key_meters.update([loss.item(), l_ph.item(), l_sm.item(), flow_mean.item()],
        #                       img_pair.size(0))
        #
        #     # compute gradient and do optimization step
        #     self.optimizer.zero_grad()
        #     # loss.backward()
        #
        #     scaled_loss = 1024. * loss
        #     scaled_loss.backward()
        #
        #     for param in [p for p in self.model.parameters() if p.requires_grad]:
        #         param.grad.data.mul_(1. / 1024)
        #
        #     self.optimizer.step()
        #
        #     # measure elapsed time
        #     am_batch_time.update(time.time() - end)
        #     end = time.time()
        #
        #     if self.i_iter % self.cfg.record_freq == 0:
        #         for v, name in zip(key_meters.val, key_meter_names):
        #             self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)
        #
        #     if self.i_iter % self.cfg.print_freq == 0:
        #         istr = '{}:{:04d}/{:04d}'.format(
        #             self.i_epoch, i_step, self.cfg.epoch_size) + \
        #                ' Time {} Data {}'.format(am_batch_time, am_data_time) + \
        #                ' Info {}'.format(key_meters)
        #         self._log.info(istr)
        #
        #     self.i_iter += 1
        # self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):
        batch_time = AverageMeter()

        return [0], ["error"]

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
