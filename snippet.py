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

# import matplotlib.pyplot as plt
# dict = np.load("prev_dpv.npy", allow_pickle=True).item()
# tofuse_dpv = dict["prev_dpv"].cuda()
#
# #tofuse_dpv = tofuse_dpv[0,:,:,:].unsqueeze(0)
#
# H, W = [41,45]
#
# #ray_dist = tofuse_dpv[0,:, H,W]
# #plt.plot(ray_dist.cpu().numpy())
#
# # Algo
# ray_dist_prev = tofuse_dpv[0,:, H,W]
# tofuse_dpv = img_utils.spread_dpv(tofuse_dpv, 5)
#
# ray_dist = tofuse_dpv[0,:, H,W]
# plt.plot(ray_dist_prev.cpu().numpy())
# plt.plot(ray_dist.cpu().numpy())
# plt.show()

######################

dict = np.load("test.npy", allow_pickle=True).item()
dpv_refined_predicted = dict["dpv_refined_predicted"]
depth_refined_truth_eval = dict["depth_refined_truth_eval"]
img_refined = dict["img_refined"]
img = F.interpolate(img_refined.unsqueeze(0), scale_factor=0.25, mode='bilinear').squeeze(0)
d_candi = dict["d_candi"]
intr_refined = dict["intr_refined"]
depthmap_truth_np = dict["depthmap_truth_np"]
depthmap_truth_refined_np = dict["depthmap_truth_refined_np"]
intr = torch.tensor(img_utils.intr_scale_unit(intr_refined.cpu().numpy(), 0.25))

# Viz
from external.perception_lib import viewer
viz = viewer.Visualizer("V")
viz.start()

# Hack the d_candi and upsample?
N = 128
dict["d_candi"] = img_utils.powerf(5., 40., N, 1.)
dict["d_candi_up"] = dict["d_candi"]
dict["r_candi"] = dict["d_candi"]
dict["r_candi_up"] = dict["d_candi"]
d_candi = dict["d_candi"]
dpv_refined_predicted = torch.exp(dpv_refined_predicted)
dpv_refined_predicted = dpv_refined_predicted.permute(0,2,1,3)
dpv_refined_predicted = F.upsample(dpv_refined_predicted, size=[N,384], mode='bilinear')
dpv_refined_predicted = dpv_refined_predicted.permute(0,2,1,3)
dpv_refined_predicted = dpv_refined_predicted / torch.sum(dpv_refined_predicted, dim=1).unsqueeze(1)
dpv_refined_predicted = torch.log(dpv_refined_predicted)

# LC
from lc import light_curtain
lc = light_curtain.LightCurtain()
# Initialize
if not lc.initialized:
    lc.init(dict)

# cloud_refined_truth = img_utils.tocloud(img_utils.dpv_to_depthmap(dpv_refined_predicted, d_candi, BV_log=True), img_utils.demean(img_refined), intr_refined)
# viz.addCloud(cloud_refined_truth)
# viz.swapBuffer()
# print(dict.keys())
# stop

# Hack
depth_larger = np.load("a1.npy", allow_pickle=True).item()["depth_refined_predicted"].squeeze(0).cpu().numpy()
#d_candi_ultra = img_utils.powerf(5., 40., 256, 1.)

# # Upsample?
# scaler=4
# depth_smaller = img_utils.minpool(depth_refined_truth_eval, scaler, 1000)
# depth_larger = F.interpolate(depth_smaller.unsqueeze(0), scale_factor=scaler, mode='nearest').squeeze(0).squeeze(0).cpu().numpy()




# Convert the DPV into a depthmap and cloud
depth_refined_predicted = img_utils.dpv_to_depthmap(dpv_refined_predicted, d_candi, BV_log=True)
cloud_refined_predicted = img_utils.tocloud(depth_refined_predicted, img_utils.demean(img_refined), intr_refined)

# Low Res DPV
dpv_predicted = F.interpolate(dpv_refined_predicted, scale_factor=0.25, mode='nearest')
depth_predicted = img_utils.dpv_to_depthmap(dpv_predicted, d_candi, BV_log=True)
cloud_predicted = img_utils.tocloud(depth_predicted, img_utils.demean(img), intr)

# # UField Low
# # Low Res?
# uncfield_predicted, _ = img_utils.gen_ufield(dpv_predicted, d_candi, intr.squeeze(0))
# lc_paths, field_visual = lc.plan_m1_low(uncfield_predicted.squeeze(0), {})
#
# # Sensing Low
# lc_outputs = []
# lc_DPVs = []
# for lc_path in lc_paths:
#     print(lc_path.shape)
#     lc_DPV, output = lc.sense_low(depthmap_truth_np, lc_path, True)
#     lc_outputs.append(output)
#     lc_DPVs.append(lc_DPV)
#     viz.addCloud(img_utils.lcoutput_to_cloud(output), 3)

# GT
truth_dpv = img_utils.gen_soft_label_torch(d_candi, torch.tensor(depthmap_truth_refined_np).cuda(), torch.tensor(0.15)).unsqueeze(0)
truth_uncfield, _ = img_utils.gen_ufield(truth_dpv, d_candi, intr_refined.squeeze(0), None, None, False, False)
# cv2.imshow("truth_uncfield", truth_uncfield.squeeze(0).cpu().numpy()*100)
# cv2.waitKey(0)
# print(truth_uncfield.shape)
# stop

def overlay_truth(input_field, truth_field):
    truth_field = truth_field.squeeze(0).cpu().numpy()
    input_field[:,:,2] += truth_field*255

# # Testing spread
# H, W = [41*4,48*4]
# dpv = torch.exp(dpv_refined_predicted)
# ray_dist_old = dpv[0,:, H,W]
# plt.plot(ray_dist_old.cpu().numpy())
#
# plt.ion()
# for i in range(0, 10):
#     dpv = img_utils.spread_dpv_hack(dpv, 15)
#     ray_dist_new = dpv[0,:, H,W]
#     plt.plot(ray_dist_new.cpu().numpy())
#     plt.pause(0.5)
#
# # N = 21
# # kernel = []
# # for i in range(0, N):
# #     kernel.append(1.*(1/float(N)))
# # print(kernel)
# # ray_test = np.convolve(ray_dist_old.cpu().numpy(), np.array(kernel), "same")
# # print(ray_test.shape)
# # ray_test = ray_test / np.sum(ray_test)
# # plt.plot(ray_test)
# plt.show()

# # Hack to spread?
# dpv_temp = torch.exp(dpv_refined_predicted)
# for i in range(0, 15):
#     dpv_temp = img_utils.spread_dpv_hack(dpv_temp, 15)
# dpv_refined_predicted = torch.log(dpv_temp)

final = dpv_refined_predicted # Cant start with all uniform cos gen_ufield will be 0 - Can we fix this, or can we spread it somehow?
for i in range(0,15):
    # UField High
    uncfield_refined_predicted, _ = img_utils.gen_ufield(final, d_candi, intr_refined.squeeze(0))
    #cv2.imshow("uncfield_refined_predicted", uncfield_refined_predicted.squeeze(0).cpu().numpy()*10)
    #cv2.waitKey(0)
    #lc_paths_refined, field_visual_refined = lc.plan_sweep_high(uncfield_refined_predicted.squeeze(0), {"step": 1.0})
    lc_paths_refined, field_visual_refined = lc.plan_default_high(uncfield_refined_predicted.squeeze(0), {"step": [0.25, 0.5, 0.75]})
    #lc_paths_refined, field_visual_refined = lc.plan_m1_high(uncfield_refined_predicted.squeeze(0), {"step": 5})
    #lc_paths_refined, field_visual_refined = lc.plan_empty_high(uncfield_refined_predicted.squeeze(0), {})
    # cv2.imshow("field_visual_refined", field_visual_refined)
    # cv2.waitKey(0)

    # Sensing High
    lc_outputs = []
    lc_DPVs = []
    i=0
    debug_datas = []
    for lc_path in lc_paths_refined:
        lc_DPV, output, debug_data = lc.sense_high(depth_larger, lc_path, True) # wrong res
        lc_outputs.append(output)
        lc_DPVs.append(lc_DPV)
        debug_datas.append(debug_data)
        #viz.addCloud(img_utils.lcoutput_to_cloud(output), 3)
        #viz.swapBuffer()
        #print(output.shape)
        #print(output[150,66,3])
        #cv2.imshow("int", output[:,:,3]/255.)
        #cv2.waitKey(1)
        #intensities.append(output[150,66,3]/255.)
        i+=1

    # FUNCTION TO VIZ THE GT ON THE MAP

    # FUNCTION TO RESPREAD THE GAUSSIAN?

    # HAVE THE OPTION TO TURN OFF INVERTED GAUSSIAN!?
    # Yes SHOWS SIGNS OF WORKING WITH SWEEP HIGH

    # # Debug
    #
    # sense_depths = []
    # intensities = []
    # thicknesses = []
    # dists = []
    # gt = debug_datas[0]["gt"]
    # for debug_data in debug_datas:
    #     sense_depths.append(debug_data["z_img"])
    #     intensities.append(debug_data["int_img"])
    #     thicknesses.append(debug_data["thickness_sensed"])
    #     dists.append(debug_data["dist"])
    #     print(debug_data["gt"])
    # plt.axvline(x=gt)
    # #plt.plot(sense_depths, thicknesses)
    # print("----")
    # index = 15
    # print(thicknesses[index])
    # print(intensities[index])
    # plt.plot(d_candi, dists[index])
    # print("----")
    # index = 16
    # print(thicknesses[index])
    # print(intensities[index])
    # plt.plot(d_candi, dists[index])
    # plt.pause(100)
    # stop

    # Keep Renormalize
    curr_dist = torch.clamp(torch.exp(final), img_utils.epsilon, 1.)
    i=0
    for lcdpv in lc_DPVs:
        #break
        prior_viz = curr_dist[0, :,150,66].cpu().numpy()
        lcdpv = torch.clamp(lcdpv, img_utils.epsilon, 1.)
        measure_viz = lcdpv[:, 150, 66].cpu().numpy()
        curr_dist = torch.exp(torch.log(lcdpv) + torch.log(curr_dist))
        curr_dist = curr_dist / torch.sum(curr_dist, dim=1).unsqueeze(1)
        curr_viz = curr_dist[0, :,150,66].cpu().numpy()

        # plt.plot(d_candi, prior_viz)
        # plt.plot(d_candi, measure_viz)
        # plt.plot(d_candi, curr_viz)
        # plt.pause(1)
        # # if i == 15:
        # #     plt.pause(100)
        # # if i == 16:
        # #     plt.pause(100)
        # plt.clf()
        # #plt.xlim([8, 15])
        # plt.ylim([0, 0.05])
        # i+=1

    # # Testing
    # measures = [lc_DPVs[10], lc_DPVs[11], lc_DPVs[12], lc_DPVs[13], lc_DPVs[14], lc_DPVs[15], lc_DPVs[16]]
    # for measure in measures:
    #     measure_viz = measure[:, 150, 66].cpu().numpy()
    #     plt.plot(d_candi, measure_viz)
    # plt.xlim([8, 15])
    # plt.ylim([0, 0.4])
    # plt.pause(100)
    # plt.clf()

    # Even with a higher res, overwrite issue exists!

    # # Add over
    # logsum = torch.log(torch.clamp(torch.exp(dpv_refined_predicted), img_utils.epsilon, 1.))
    # for lcdpv in lc_DPVs:
    #     lcdpv = torch.clamp(lcdpv, img_utils.epsilon, 1.)
    #     logsum += torch.log(lcdpv)
    # curr_dist = torch.exp(logsum)
    # curr_dist = curr_dist / torch.sum(curr_dist, dim=1).unsqueeze(1)


    # #print(final[0, :, 150, 66])
    # final = torch.exp(logsum)
    # final = final / torch.sum(final, dim=1).unsqueeze(1)
    # final = torch.log(final)

    # # Respread?
    # for i in range(0, 3):
    #     curr_dist = img_utils.spread_dpv_hack(curr_dist, 5)

    # H, W = [41*4,48*4]
    # ray_dist_old = curr_dist[0,:, H,W]
    # plt.plot(ray_dist_old.cpu().numpy())
    # plt.pause(0.5)


    final = torch.log(curr_dist)

    # ISSUES
    # Need a larger dim space instead of 64? - Can we temporarily interpolate over it?

    # Objects that are closer to cam suffer more via this method

    # WE NEED TO FIRST GET A REGULAR SWEEP TO WORK WELL

    # How to

    # UField High
    final_ufield, _ = img_utils.gen_ufield(final, d_candi, intr_refined.squeeze(0))
    _, field_visual_final = lc.plan_empty_high(final_ufield.squeeze(0), {})
    overlay_truth(field_visual_final, truth_uncfield)
    cv2.imshow("field_visual_final", field_visual_final)

    #cloud_refined_truth = img_utils.tocloud(depth_refined_truth_eval, img_utils.demean(img_refined), intr_refined)
    #viz.addCloud(cloud_refined_truth)
    #viz.addCloud(cloud_refined_predicted)
    #viz.addCloud(cloud_predicted, 3)
    viz.addCloud(img_utils.tocloud(img_utils.dpv_to_depthmap(final, d_candi, BV_log=True), img_utils.demean(img_refined), intr_refined), 1)
    viz.swapBuffer()
    overlay_truth(field_visual_refined, truth_uncfield)
    cv2.imshow("field_visual_refined", field_visual_refined)
    #cv2.imshow("uncfield_refined_predicted", uncfield_refined_predicted.squeeze(0).cpu().numpy())
    print("waiting")
    cv2.waitKey(0)
    print("end")


#CAN WE FAKE UPSAMPLE THE DEPTH CHECK HERE (Or use a network to do this??)

# DO THE REGRESS EXP

# print(depth_refined_truth_eval.shape)
# scaler = 4
# depth_smaller = img_utils.minpool(depth_refined_truth_eval, scaler, 1000)
# depth_larger = F.interpolate(depth_smaller.unsqueeze(0), scale_factor=scaler, mode='nearest').squeeze(0)
# print(depth_smaller.shape)
# print(depth_larger.shape)
# #viz.addCloud(img_utils.tocloud(depth_refined_truth_eval, img_utils.demean(img_refined), intr_refined))
# viz.addCloud(img_utils.tocloud(depth_smaller, img_utils.demean(img), intr))
# #viz.addCloud(img_utils.tocloud(depth_larger, img_utils.demean(img_refined), intr_refined))
# viz.swapBuffer()
# #cv2.imshow("win", depth_larger.squeeze(0).cpu().numpy()/100.)
# #cv2.imshow("win2", depth_smaller.squeeze(0).cpu().numpy()/100.)
# cv2.waitKey(0)
# stop


# import cv2
# cloud_refined_truth = img_utils.tocloud(depth_refined_truth_eval, img_utils.demean(img_refined), intr_refined)
# viz.addCloud(cloud_refined_truth)
# #viz.addCloud(cloud_refined_predicted)
# #viz.addCloud(cloud_predicted, 3)
# viz.swapBuffer()
# cv2.imshow("field_visual_refined", field_visual_refined)
# #cv2.imshow("uncfield_refined_predicted", uncfield_refined_predicted.squeeze(0).cpu().numpy())
# cv2.waitKey(0)


################

# # UField High
# uncfield_refined_predicted, _ = img_utils.gen_ufield(dpv_refined_predicted, d_candi, intr_refined.squeeze(0))
# lc_paths_refined, field_visual_refined = lc.plan_m1_high(uncfield_refined_predicted.squeeze(0))
#
# import cv2
# cv2.imshow("field_visual_refined", field_visual_refined)
# cv2.imshow("uncfield_refined_predicted", uncfield_refined_predicted.squeeze(0).cpu().numpy())
# cv2.waitKey(0)

# Two strategies..add the noise control thing..

# or sample random choice in each


# Lets just compute all the stuff here and visualize! It should make the unc field shrink?

    #start_dpv = torch.clamp(dpv_refined_predicted, img_utils.epsilon, 1.)
    #logsum = final
    #logsum += final
    # logsum = img_utils.gen_uniform(d_candi, torch.tensor(depth_larger)).unsqueeze(0).cuda()
    # i=0
    # for lcdpv in lc_DPVs:
    #     #logsum = torch.log(torch.clamp(torch.exp(logsum), img_utils.epsilon, 1.))
    #     lcdpv = torch.clamp(lcdpv, img_utils.epsilon, 1.)
    #
    #     sample_prior = torch.exp(logsum); sample_prior = sample_prior / torch.sum(sample_prior, dim=1).unsqueeze(1);
    #     sample_prior = sample_prior[0,:,150,66].cpu().numpy()
    #     sample_measurement = lcdpv[:,150,66].cpu().numpy()
    #
    #     logsum += torch.log(lcdpv.unsqueeze(0))
    #
    #     print(i)
    #
    #
    #     print(logsum[0,:,150,66])
    #     mf = torch.exp(logsum);
    #     print(mf[0, :, 150, 66])
    #     mf = mf / torch.sum(mf, dim=1).unsqueeze(1);
    #     print(mf[0, :, 150, 66])
    #     sample_mf = mf[0, :,150,66].cpu().numpy()
    #
    #     plt.plot(d_candi, sample_prior)
    #     plt.plot(d_candi, sample_measurement)
    #     plt.plot(d_candi, sample_mf)
    #
    #     #print(sample_prior)
    #     #print(sample_measurement)
    #     #print(sample_mf)
    #     print("--")
    #
    #     if i == 26:
    #         stop
    #         plt.pause(25)
    #     plt.pause(0.1)
    #     plt.clf()
    #     plt.ylim([0, 1.0])
    #
    #     i+=1


print("END")
print(dict["dpv_refined_predicted"].shape)