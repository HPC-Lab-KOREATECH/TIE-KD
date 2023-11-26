from collections import OrderedDict

import mmcv
import numpy as np
# import torch
import sys
from PIL import Image
from depth.utils import colorize

def bin_feature_fn(input, bin_edge):   
    bin_edge_shape = bin_edge.shape   
           
    bin_data = None      
    for j in range(bin_edge_shape[0]-1):      #channel iter      
        if(j==0):                    
            condition_base = input<=bin_edge[j]
            bin_data = np.where(condition_base,j,0)
        elif(j!=bin_edge_shape[0]-1):
            condition_base = np.logical_and(bin_edge[j-1]<input, input<=bin_edge[j])
            bin_data_temp = np.where(condition_base,j,0)
            bin_data +=bin_data_temp
        else:                                 #last condition
            condition_base = bin_edge[j-1]<input                    
            bin_data_temp = np.where(condition_base,j,0)
            bin_data +=bin_data_temp
    
    # print(input)
    # print(np.max(input))
    # print(np.min(input))
    # print(bin_data)
    # print(np.max(bin_data))
    # print(np.min(bin_data))

    # print(input.shape)
    # print(np.nonzero(input))
    # print(len(np.nonzero(input)[0]))
    # print(len(input))
    # print("@@@@@@@@@@")    
    # input = input/np.max(input)*255
    
    # input_copy = input/80*255        
    # input_copy=input_copy.squeeze()
    # input_copy = input_copy.astype('uint8')
    # gt_img = Image.fromarray(input_copy)
    # gt_img.save('input00.png')
    # sys.exit()
    return bin_data


# def calculate(gt, pred):
def calculate(kd_gt, pred_base, gt, pred):
    # print("calculate@!@@@@")
    # print(pred_base.shape)
    # sys.exit()
    if gt.shape[0] == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    # print(kd_gt[:][101:].shape)
    # print(gt[:][101:].shape)
    # print(pred_base[:][101:].shape)
    # sys.exit()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)

    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    if np.isnan(silog):
        silog = 0
        
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    # bins = np.linspace(1e-3, 80, 256)
    # kd_class = bin_feature_fn(kd, bins)
    # pred_class = bin_feature_fn(pred_raw, bins)

    ####################
    

    # if kd_gt is not None:
    # if(pred_base.shape[1]!=480):        
    #     bins = np.linspace(1e-3, 80, 256)
    #     kd_class = bin_feature_fn(kd_gt, bins)
    #     pred_class = bin_feature_fn(pred_base, bins)
    # else:
    #     bins = np.linspace(1e-3, 10, 33)
    #     kd_class = bin_feature_fn(kd_gt, bins)
    #     pred_class = bin_feature_fn(pred_base, bins)

    # _,W,H = kd_gt.shape
    # kd_pred = kd_class - pred_class
    # acc_1 = ((W*H)-len(np.nonzero(kd_pred)[0]))/(W*H)

    # kd_pred = np.where(abs(kd_pred)<=5,0,kd_pred)
    # acc_5 = ((W*H)-len(np.nonzero(kd_pred)[0]))/(W*H)
    
    #########
    if(np.max(kd_gt)>10000):
        kd_gt = kd_gt/65535*80

    # print(pred_base.shape)
    # sys.exit()
    if(pred_base.shape[1]!=480):
        rmse_T = (kd_gt[:, 110: ,:] - pred_base[:, 110: ,:]) ** 2
        rmse_T = np.sqrt(rmse_T.mean())

        thresh_T = np.maximum((kd_gt[:, 110: ,:] / pred_base[:, 110: ,:]), (pred_base[:, 110: ,:] / kd_gt[:, 110: ,:]))
        a1_T = (thresh_T < 1.25).mean()
        a2_T = (thresh_T < 1.25 ** 2).mean()
        a3_T = (thresh_T < 1.25 ** 3).mean()

        abs_rel_T = np.mean(np.abs(kd_gt[:, 110: ,:] - pred_base[:, 110: ,:]) / kd_gt[:, 110: ,:])
        sq_rel_T = np.mean(((kd_gt[:, 110: ,:] - pred_base[:, 110: ,:]) ** 2) / kd_gt[:, 110: ,:])
        rmse_log_T = (np.log(kd_gt[:, 110: ,:]) - np.log(pred_base[:, 110: ,:])) ** 2
        rmse_log_T = np.sqrt(rmse_log_T.mean())
    else:
        rmse_T = (kd_gt - pred_base) ** 2
        rmse_T = np.sqrt(rmse_T.mean())
        
        thresh_T = np.maximum((kd_gt / pred_base), (pred_base / kd_gt))
        a1_T = (thresh_T < 1.25).mean()
        a2_T = (thresh_T < 1.25 ** 2).mean()
        a3_T = (thresh_T < 1.25 ** 3).mean()

        abs_rel_T = np.mean(np.abs(kd_gt - pred_base) / kd_gt)
        sq_rel_T = np.mean(((kd_gt - pred_base) ** 2) / kd_gt)
        rmse_log_T = (np.log(kd_gt) - np.log(pred_base)) ** 2
        rmse_log_T = np.sqrt(rmse_log_T.mean())

    # print("hihi")
    # print(np.max(kd_gt))
    # print(np.min(kd_gt))

    # kd_gt_copy = kd_gt/80*255        
    # kd_gt_copy=kd_gt_copy.squeeze()
    # kd_gt_copy = kd_gt_copy.astype('uint8')
    # kd_img = Image.fromarray(kd_gt_copy)
    # kd_img.save('kd_gt_test.png')
    # sys.exit()

    # else:
    #     acc_1=None
    #     acc_5=None
    #     rmse_T=None

    # print(kd_gt.shape)
    # print(type(kd_gt))
    # print(pred_base.shape)
    # print(type(pred_base))
    # print(kd_gt[:, 110: ,:].shape)
    # sys.exit()

    # kd_gt = kd_gt[:][101:]
    # pred_base = pred_base[:][101:]
    # print(kd_gt[:][101:].shape)
    # print(pred_base[:][101:].shape)
    # sys.exit()

    # print("hihi")
    # kd_gt_copy = kd_gt/80*255        
    # kd_gt_copy=kd_gt_copy.squeeze()
    # kd_gt_copy = kd_gt_copy.astype('uint8')
    # kd_img = Image.fromarray(kd_gt_copy)
    # kd_img.save('kd_gt_test.png')

    # kd_gt_copy = pred_base/80*255        
    # kd_gt_copy=kd_gt_copy.squeeze()
    # kd_gt_copy = kd_gt_copy.astype('uint8')
    # kd_img = Image.fromarray(kd_gt_copy)
    # kd_img.save('pred_base_test.png')

    # kd_pred = np.where(kd_pred!=0,1,0)

    # kd_gt_copy = kd_pred*255        
    # kd_gt_copy=kd_gt_copy.squeeze()
    # kd_gt_copy = kd_gt_copy.astype('uint8')
    # kd_img = Image.fromarray(kd_gt_copy)
    # kd_img.save('kd_pred_test.png')

    # kd_gt_copy = kd_gt/80*255        
    # kd_gt_copy=kd_gt_copy.squeeze()
    # kd_gt_copy = kd_gt_copy.astype('uint8')
    # kd_img = Image.fromarray(kd_gt_copy)
    # kd_img.save('kd_gt_test.png')

    # kd_gt_copy = pred_base/80*255        
    # kd_gt_copy=kd_gt_copy.squeeze()
    # kd_gt_copy = kd_gt_copy.astype('uint8')
    # kd_img = Image.fromarray(kd_gt_copy)
    # kd_img.save('pred_base_test.png')

    # # kd_pred = np.where(kd_pred!=0,1,0)

    # diff = abs(kd_gt-pred_base)
    # kd_gt_copy = colorize(diff, vmin=1e-3, vmax=80)

    # # kd_gt_copy = diff/40*255        
    # kd_gt_copy=kd_gt_copy.squeeze()
    # kd_gt_copy = kd_gt_copy.astype('uint8')
    # kd_img = Image.fromarray(kd_gt_copy)
    # kd_img.save('diff.png')
    # sys.exit()

    # sys.exit()
    
    # print(kd_pred)
    # print(np.max(kd_pred))
    # print(np.min(kd_pred))
    
    # print(acc)
    # print(len(np.nonzero(kd_pred)[0]))
    # print(W*H)
    # sys.exit()
    
    ####################



    # return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel
    # return a1, a2, a3, abs_rel, rmse, log_10, rmse_T, silog, sq_rel, rmse_log
    return a1, a2, a3, abs_rel, rmse, log_10, rmse_T, silog, sq_rel, rmse_log, a1_T,a2_T,a3_T, abs_rel_T, sq_rel_T, rmse_log_T


    # return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel, acc_5, acc_1, rmse_T


def metrics(kd, pred_raw, gt, pred, min_depth=1e-3, max_depth=80):
# def metrics(gt, pred, min_depth=1e-3, max_depth=80):
    # print(pred_raw.shape)
    # print("hihihihihi")
    # print(np.max(kd))
    # print(np.max(pred_raw))
    # print("@@@@@@@@@@@@")

    mask_1 = gt > min_depth
    mask_2 = gt < max_depth
    mask = np.logical_and(mask_1, mask_2)
    gt = gt[mask]
    pred = pred[mask]
    
    # a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = calculate(gt, pred)
    # a1, a2, a3, abs_rel, rmse, log_10, rmse_T, silog, sq_rel, rmse_log = calculate(kd,pred_raw, gt, pred)
    a1, a2, a3, abs_rel, rmse, log_10, rmse_T, silog, sq_rel, rmse_log, a1_T,a2_T,a3_T, abs_rel_T, sq_rel_T, rmse_log_T = calculate(kd,pred_raw, gt, pred)


    # a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel, acc_5, acc_1, rmse_T = calculate(kd,pred_raw, gt, pred)



    # return a1, a2, a3, abs_rel, rmse, log_10, rmse_T, silog, sq_rel, rmse_log
    return a1, a2, a3, abs_rel, rmse, log_10, rmse_T, silog, sq_rel, rmse_log, a1_T,a2_T,a3_T, abs_rel_T, sq_rel_T, rmse_log_T
    # return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel, acc_5, acc_1, rmse_T



def eval_metrics(kd_gt, gt, pred, min_depth=1e-3, max_depth=80):
    # print("hihi")
    mask_1 = gt > min_depth
    mask_2 = gt < max_depth
    mask = np.logical_and(mask_1, mask_2)

    gt = gt[mask]
    pred = pred[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()    


    if(kd_gt.shape[1]!=480):
        rmse_T = (kd_gt[:, 110: ,:] - pred[:, 110: ,:]) ** 2
        rmse_T = np.sqrt(rmse_T.mean())

        thresh_T = np.maximum((kd_gt[:, 110: ,:] / pred[:, 110: ,:]), (pred[:, 110: ,:] / kd_gt[:, 110: ,:]))
        a1_T = (thresh_T < 1.25).mean()
        a2_T = (thresh_T < 1.25 ** 2).mean()
        a3_T = (thresh_T < 1.25 ** 3).mean()

        abs_rel_T = np.mean(np.abs(kd_gt[:, 110: ,:] - pred[:, 110: ,:]) / kd_gt[:, 110: ,:])
        sq_rel_T = np.mean(((kd_gt[:, 110: ,:] - pred[:, 110: ,:]) ** 2) / kd_gt[:, 110: ,:])
        rmse_log_T = (np.log(kd_gt[:, 110: ,:]) - np.log(pred[:, 110: ,:])) ** 2
        rmse_log_T = np.sqrt(rmse_log_T.mean())
    else:
        rmse_T = (kd_gt - pred) ** 2
        rmse_T = np.sqrt(rmse_T.mean())
        
        thresh_T = np.maximum((kd_gt / pred), (pred / kd_gt))
        a1_T = (thresh_T < 1.25).mean()
        a2_T = (thresh_T < 1.25 ** 2).mean()
        a3_T = (thresh_T < 1.25 ** 3).mean()

        abs_rel_T = np.mean(np.abs(kd_gt - pred) / kd_gt)
        sq_rel_T = np.mean(((kd_gt - pred) ** 2) / kd_gt)
        rmse_log_T = (np.log(kd_gt) - np.log(pred)) ** 2
        rmse_log_T = np.sqrt(rmse_log_T.mean())

    
    #     _,W,H = kd_gt.shape
    #     kd_pred = kd_class - pred_class
    #     acc_1 = ((W*H)-len(np.nonzero(kd_pred)[0]))/(W*H)

    #     kd_pred = np.where(abs(kd_pred)<=5,0,kd_pred)
    #     acc_5 = ((W*H)-len(np.nonzero(kd_pred)[0]))/(W*H)

        
    # else:
    #     acc_1=None
    #     acc_5=None
    #     rmse_T=None

    # return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
    #             silog=silog, sq_rel=sq_rel, rmse_T=rmse_T)
    
    # return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_T=rmse_T,
    #             silog=silog, sq_rel=sq_rel, rmse_log=rmse_log)
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_T=rmse_T, silog=silog, sq_rel=sq_rel, rmse_log=rmse_log, a1_T=a1_T,a2_T=a2_T,a3_T=a3_T, abs_rel_T=abs_rel_T, sq_rel_T=sq_rel_T, rmse_log_T=rmse_log_T)


def pre_eval_to_metrics(pre_eval_results):
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    ret_metrics = OrderedDict({})

    ret_metrics['a1'] = np.nanmean(pre_eval_results[0])
    ret_metrics['a2'] = np.nanmean(pre_eval_results[1])
    ret_metrics['a3'] = np.nanmean(pre_eval_results[2])
    ret_metrics['abs_rel'] = np.nanmean(pre_eval_results[3])
    ret_metrics['rmse'] = np.nanmean(pre_eval_results[4])
    ret_metrics['log_10'] = np.nanmean(pre_eval_results[5])
    ret_metrics['rmse_log'] = np.nanmean(pre_eval_results[9])
    ret_metrics['silog'] = np.nanmean(pre_eval_results[7])
    ret_metrics['sq_rel'] = np.nanmean(pre_eval_results[8])
    ret_metrics['rmse_T'] = np.nanmean(pre_eval_results[6])
    
    ret_metrics['a1_T'] = np.nanmean(pre_eval_results[10])
    ret_metrics['a2_T'] = np.nanmean(pre_eval_results[11])
    ret_metrics['a3_T'] = np.nanmean(pre_eval_results[12])
    ret_metrics['abs_rel_T'] = np.nanmean(pre_eval_results[13])
    ret_metrics['sq_rel_T'] = np.nanmean(pre_eval_results[14])
    ret_metrics['rmse_log_T'] = np.nanmean(pre_eval_results[15])



    ret_metrics = {
        metric: value
        for metric, value in ret_metrics.items()
    }
    return ret_metrics

