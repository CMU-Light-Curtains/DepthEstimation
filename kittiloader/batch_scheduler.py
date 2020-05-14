# Python
import numpy as np
import time
import cv2

# Custom
try:
    import kittiloader.kitti as kitti
    import kittiloader.batch_loader as batch_loader
except:
    import kitti
    import batch_loader

# Data Loading Module
import torch.multiprocessing
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Value, cpu_count
import utils.img_utils as img_utils
import utils.misc_utils as misc_utils

def generate_model_input(id, local_info_valid, cfg, camside="left"):
    # Device
    n_gpu_use = cfg.train.n_gpu
    device = torch.device('cuda:' + str(id) if n_gpu_use > 0 else 'cpu')

    # Ensure same size
    valid = (len(local_info_valid[camside + "_cam_intrins"]) == len(
        local_info_valid[camside + "_src_cam_poses"]) == len(local_info_valid["src_dats"]))
    if not valid:
        raise Exception('Batch size invalid')

    # Keep to middle only
    midval = int(len(local_info_valid["src_dats"][0]) / 2)
    preval = 0

    # Grab ground truth digitized map
    dmap_imgsize_digit_arr = []
    dmap_digit_arr = []
    dmap_imgsize_arr = []
    dmap_arr = []
    dmap_imgsize_prev_arr = []
    dmap_prev_arr = []
    for i in range(0, len(local_info_valid["src_dats"])):
        dmap_imgsize_digit = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_imgsize_digit"]
        dmap_imgsize_digit_arr.append(dmap_imgsize_digit)
        dmap_digit = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap"]
        dmap_imgsize = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_imgsize"]
        dmap = local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_raw"]
        dmap_digit_arr.append(dmap_digit)
        dmap_imgsize_arr.append(dmap_imgsize)
        dmap_arr.append(dmap)
        dmap_imgsize_prev = local_info_valid["src_dats"][i][preval][camside + "_camera"]["dmap_imgsize"]
        dmap_prev = local_info_valid["src_dats"][i][preval][camside + "_camera"]["dmap_raw"]
        dmap_imgsize_prev_arr.append(dmap_imgsize_prev)
        dmap_prev_arr.append(dmap_prev)
    dmap_imgsize_digits = torch.cat(dmap_imgsize_digit_arr).to(device)  # [B,256,384] uint64
    dmap_digits = torch.cat(dmap_digit_arr).to(device)  # [B,64,96] uint64
    dmap_imgsizes = torch.cat(dmap_imgsize_arr).to(device)
    dmaps = torch.cat(dmap_arr).to(device)
    dmap_imgsizes_prev = torch.cat(dmap_imgsize_prev_arr).to(device)
    dmaps_prev = torch.cat(dmap_prev_arr).to(device)

    intrinsics_arr = []
    intrinsics_up_arr = []
    unit_ray_arr = []
    for i in range(0, len(local_info_valid[camside + "_cam_intrins"])):
        intr = local_info_valid[camside + "_cam_intrins"][i]["intrinsic_M_cuda"]
        intr_up = intr * 4;
        intr_up[2, 2] = 1;
        intrinsics_arr.append(intr.unsqueeze(0))
        intrinsics_up_arr.append(intr_up.unsqueeze(0))
        unit_ray_arr.append(local_info_valid[camside + "_cam_intrins"][i]["unit_ray_array_2D"].unsqueeze(0))
    intrinsics = torch.cat(intrinsics_arr).to(device)
    intrinsics_up = torch.cat(intrinsics_up_arr).to(device)
    unit_ray = torch.cat(unit_ray_arr).to(device)

    src_cam_poses_arr = []
    for i in range(0, len(local_info_valid[camside + "_src_cam_poses"])):
        pose = local_info_valid[camside + "_src_cam_poses"][i]
        src_cam_poses_arr.append(pose[:, 0:midval + 1, :, :])  # currently [1x3x4x4]
    src_cam_poses = torch.cat(src_cam_poses_arr).to(device)
    if cfg.var.pnoise: src_cam_poses = img_utils.add_noise2pose(src_cam_poses, cfg.var.pnoise).to(device)

    mask_imgsize_arr = []
    mask_arr = []
    rgb_arr = []
    debug_path = []
    for i in range(0, len(local_info_valid["src_dats"])):
        rgb_set = []
        debug_path_int = []
        for j in range(0, len(local_info_valid["src_dats"][i])):
            rgb_set.append(local_info_valid["src_dats"][i][j][camside + "_camera"]["img"])
            debug_path_int.append(local_info_valid["src_dats"][i][j][camside + "_camera"]["img_path"])
            if j == midval: break
        rgb_arr.append(torch.cat(rgb_set).unsqueeze(0))
        debug_path.append(debug_path_int)
        mask_imgsize_arr.append(local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_mask_imgsize"])
        mask_arr.append(local_info_valid["src_dats"][i][midval][camside + "_camera"]["dmap_mask"])
    rgb = torch.cat(rgb_arr).to(device)
    masks_imgsize = torch.cat(mask_imgsize_arr).float().to(device)
    masks = torch.cat(mask_arr).float().to(device)

    # Create Soft Label
    d_candi = local_info_valid["d_candi"]
    if cfg.var.softce:
        soft_labels_imgsize = []
        soft_labels = []
        variance = torch.tensor(cfg.var.softce)
        for i in range(0, dmap_imgsizes.shape[0]):
            # Clamping
            dmap_imgsize = dmap_imgsizes[i, :, :].clamp(d_candi[0], d_candi[-1]) * masks_imgsize[i, 0, :, :]
            dmap = dmaps[i, :, :].clamp(d_candi[0], d_candi[-1]) * masks[i, 0, :, :]
            soft_labels_imgsize.append(
                img_utils.gen_soft_label_torch(d_candi, dmap_imgsize, variance, zero_invalid=True))
            soft_labels.append(img_utils.gen_soft_label_torch(d_candi, dmap, variance, zero_invalid=True))
            # Generate Fake one with all 1
            # soft_labels.append(util.digitized_to_dpv(dmap_digits[i,:,:].unsqueeze(0), len(d_candi)).squeeze(0).cuda())
            # soft_labels_imgsize.append(util.digitized_to_dpv(dmap_imgsize_digits[i,:,:].unsqueeze(0), len(d_candi)).squeeze(0).cuda())
    else:
        soft_labels_imgsize = []
        soft_labels = []

    model_input = {
        "intrinsics": intrinsics,
        "intrinsics_up": intrinsics_up,
        "unit_ray": unit_ray,
        "src_cam_poses": src_cam_poses,
        "rgb": rgb,
        "prev_output": None,  # Has to be [B, 64, H, W],
        "dmaps": dmaps,
        "masks": masks,
        "d_candi": d_candi,
    }

    gt_input = {
        "masks_imgsizes": masks_imgsize,
        "masks": masks,
        "dmap_imgsize_digits": dmap_imgsize_digits,
        "dmap_digits": dmap_digits,
        "dmap_imgsizes": dmap_imgsizes,
        "dmaps": dmaps,
        "dmap_imgsizes_prev": dmap_imgsizes_prev,
        "dmaps_prev": dmaps_prev,
        "soft_labels_imgsize": soft_labels_imgsize,
        "soft_labels": soft_labels,
        "d_candi": d_candi,
        "T_left2right": local_info_valid["T_left2right"],
        "rgb": rgb,
        "intrinsics": intrinsics,
        "intrinsics_up": intrinsics_up,
    }

    return model_input, gt_input

class BatchSchedulerMP:
    def __init__(self, inputs, mload):
        self.mload = mload
        self.inputs = inputs
        self.qmax = self.inputs["qmax"]

    def stop(self):
        if self.mload:
            self.control.value = 0
        else:
            self.control = 0

    def enumerate(self):
        if self.mload:
            smp = mp.get_context('spawn')
            queue = smp.Queue(maxsize=self.qmax)
            self.control = smp.Value('i', 1)
            mp.spawn(self.worker, nprocs=1, args=(self.inputs, queue, self.control), join=False)
            #self.process = Process(target=self.worker, args=(0, self.inputs, self.queue, self.control))
            #self.process.start()
            while 1:
                if self.control.value == 0:
                    _ = queue.get()
                    break
                items = queue.get()
                if items is None: break
                yield items
        else:
            self.control = 1
            for items in self.single(self.inputs):
                if self.control == 0:
                    break
                if items is None: break
                yield items

    def load(self, inputs):
        dataset_path = inputs["dataset_path"]
        t_win_r = inputs["t_win_r"]
        img_size = inputs["img_size"]
        crop_w = inputs["crop_w"]
        d_candi = inputs["d_candi"]
        d_candi_up = inputs["d_candi_up"]
        dataset_name = inputs["dataset_name"]
        hack_num = inputs["hack_num"]
        batch_size = inputs["batch_size"]
        qmax = inputs["qmax"]
        n_epoch = inputs["n_epoch"]
        mode = inputs["mode"]
        pytorch_scaling = inputs["pytorch_scaling"]
        velodyne_depth = inputs["velodyne_depth"]
        split_path = inputs["split_path"]
        total_processes = inputs["total_processes"]
        process_num = inputs["process_num"]
        cfg = inputs["cfg"]
        if mode == "train":
            split_txt = split_path + '/training.txt'
        elif mode == "val":
            split_txt = split_path + '/testing.txt'

        # Split the information here
        scene_names = misc_utils.read_split_file(split_txt)
        scene_names_split = dict()
        for i in range(0, total_processes):
            scene_names_split[i] = []
        for i in range(0, len(scene_names)):
            vi = i % total_processes
            if len(scene_names[i]) == 0: continue
            scene_names_split[vi].append(scene_names[i])

        dataset_init = kitti.KITTI_dataset
        fun_get_paths = lambda traj_indx: kitti.get_paths(traj_indx, scene_names=scene_names_split[process_num],
                                                          mode=mode,
                                                          database_path_base=dataset_path, t_win=t_win_r)

        # Load Dataset
        n_scenes, _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)
        fldr_path, img_paths, dmap_paths, poses, intrin_path = fun_get_paths(0)

        #print(len(dmap_paths))
        #stop

        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path=intrin_path, img_size=img_size, digitize=True,
                               d_candi=d_candi, d_candi_up=d_candi_up, resize_dmap=.25,
                               crop_w=crop_w, velodyne_depth=velodyne_depth, pytorch_scaling=pytorch_scaling, cfg=cfg)
        BatchScheduler = batch_loader.Batch_Loader(
            batch_size=batch_size, fun_get_paths=fun_get_paths,
            dataset_traj=dataset, nTraj=len(traj_Indx), dataset_name=dataset_name, t_win_r=t_win_r,
            hack_num=hack_num)
        return BatchScheduler

    def worker(self, id, inputs, queue, control):
        qmax = inputs["qmax"]
        n_epoch = inputs["n_epoch"]

        # Iterate batch
        broken = False
        for iepoch in range(n_epoch):
            BatchScheduler = self.load(inputs)
            for batch_idx in range(len(BatchScheduler)):
                start = time.time()
                for frame_count, ref_indx in enumerate(range(BatchScheduler.traj_len)):
                    local_info = BatchScheduler.local_info_full()

                    # Put in Q
                    queue.put([local_info, len(BatchScheduler), batch_idx, frame_count, BatchScheduler.traj_len, iepoch])

                    # Break
                    if control.value == 0:
                        broken = True
                        break

                    # Update dat_array
                    if frame_count < BatchScheduler.traj_len - 1:
                        BatchScheduler.proceed_frame()

                    # print(batch_idx, frame_count)
                    if broken: break

                if broken: break
                BatchScheduler.proceed_batch()

            if broken: break

        # If we prematurely stopped the process, then empty the queue
        if control.value == 0:
            time.sleep(1)
            while not queue.empty():
                _ = queue.get()
        # If it naturally ended, we need to wait for queue to clear
        else:
            while queue.qsize():
                time.sleep(0.1)
            time.sleep(1)
            queue.put(None)

        #print(queue.qsize())
        print("DataLoader End")

    def single(self, inputs):
        qmax = inputs["qmax"]
        n_epoch = inputs["n_epoch"]

        # Iterate batch
        broken = False
        for iepoch in range(n_epoch):
            BatchScheduler = self.load(inputs)
            for batch_idx in range(len(BatchScheduler)):
                start = time.time()
                for frame_count, ref_indx in enumerate(range(BatchScheduler.traj_len)):
                    local_info = BatchScheduler.local_info_full()
                    #local_info = 0

                    # if frame_count == 0:
                    #   print(local_info["src_dats"][0][0]["left_camera"]["img_path"])

                    # Put in Q
                    yield [local_info, len(BatchScheduler), batch_idx, frame_count, BatchScheduler.traj_len, iepoch]

                    # Update dat_array
                    start = time.time()
                    if frame_count < BatchScheduler.traj_len - 1:
                        BatchScheduler.proceed_frame()

                    # print(batch_idx, frame_count)
                    if broken: break

                if broken: break
                BatchScheduler.proceed_batch()

            if broken: break
        yield None

if __name__ == "__main__":
    testing_inputs = {
        "pytorch_scaling": True,
        "velodyne_depth": True,
        #"dataset_path": "/media/raaj/Storage/kitti/",
        #"split_path": "./kittiloader/k1/",
        "dataset_path": "/home/raaj/adapfusion/kittiloader/kitti/",
        "split_path": "./kittiloader/k2/",
        "total_processes": 1,
        "process_num": 0,
        "t_win_r": 1,
        "img_size": [768, 256],
        "crop_w": 384,
        "d_candi": img_utils.powerf(5., 40., 64, 1.5),
        "d_candi_up": img_utils.powerf(5., 40., 64, 1.5),
        "dataset_name": "kitti",
        "hack_num": 0,
        "batch_size": 4,
        "n_epoch": 1,
        "qmax": 1,
        "mode": "val",
        "cfg": None
    }

    # Add feature to control lidar params

    # train regular on other server

    # add lidar control stuff?

    # Should we clamp the count?? so every batch looks at mixed or max no of images

    # Save Every 5k iter

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from models import models
    x = models.BaseDecoder(3,3,3)

    bs = BatchSchedulerMP(testing_inputs, True) # Multiprocessing misses last image for some reason

    counter = 0
    early_stop = False
    for epoch in range(0, 1):
        print("Epoch: " + str(epoch))
        if early_stop: break

        for items in bs.enumerate():

            # Get data
            local_info, batch_length, batch_idx, frame_count, frame_length, iepoch = items

            # Print
            print('video batch %d / %d, iter: %d, frame_count: %d / %d; Epoch: %d / %d, loss = %.5f' \
                  % (batch_idx + 1, batch_length, 0, frame_count + 1, frame_length, iepoch + 1, 0, 0))

            time.sleep(1)

            # # Test Stop
            # counter += 1
            # if counter == 6:
            #     bs.stop()
            #     early_stop = True

            # # # Visualize
            # global_item = []
            # for b in range(0, len(local_info["src_dats"])):
            #     batch_item = []
            #     for i in range(0, len(local_info["src_dats"][b])):
            #         if i > 1: continue
            #         batch_item.append(local_info["src_dats"][b][i]["left_camera"]["img"])
            #     batch_item = torch.cat(batch_item, dim = 3)
            #     global_item.append(batch_item)
            # global_item = torch.cat(global_item, dim=2)
            # cv2.imshow("win", img_utils.torchrgb_to_cv2(global_item.squeeze(0)))
            # cv2.waitKey(15)

    print("Ended")

    # counter = 0
    # for epoch in range(0, 1):
    #     print("Epoch: " + str(epoch))
    #
    #     for items in bs.enumerate():
    #
    #         # Get data
    #         local_info, batch_length, batch_idx, frame_count, frame_length, iepoch = items
    #
    #         # Print
    #         print('video batch %d / %d, iter: %d, frame_count: %d / %d; Epoch: %d / %d, loss = %.5f' \
    #               % (batch_idx + 1, batch_length, 0, frame_count + 1, frame_length, iepoch + 1, 0, 0))
    #
    #         time.sleep(1)
    #
    #         # Test Stop
    #         counter += 1
    #         if counter == 6:
    #             bs.stop()
    #
    # print("Ended")