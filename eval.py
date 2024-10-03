import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils_dir.utils as utils
from utils_dir.eval_utils import define_metrics, binarized_negatives_thresholds, append_metrics_binarized_pred, compute_metrics
import easydict
import numpy as np
import argparse
from models.model import EZVSL, SLAVC, FNAC
from models.model_lvs import AVENet
from models.model_ssltie import AVENet_ssltie
from datasets.datasets import get_test_dataset, get_test_dataset_fnac
from datasets.datasets_lvs import GetAudioVideoDataset
from datasets.datasets_ssltie import GetAudioVideoDataset_ssltie
import cv2
from tqdm import tqdm
import utils_dir.utils_lvs as utils_lvs
from utils_dir.opts_ssltie import SSLTIE_args
import matplotlib.pyplot as plt
from PIL import Image

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='', help='path to save trained model weights')
    parser.add_argument('--pth_name', type=str, default='', help='pth name')
    parser.add_argument('--data_dir', type=str, default='', help='path to data directory')

    # Dataset
    parser.add_argument('--testset', default='vggss', type=str, help='testset (flickr or vggss)')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of data')
    parser.add_argument('--test_gt_path', default='', type=str)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')

    # Model
    parser.add_argument('--tau', default=0.03, type=float, help='tau')
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha')
    
    # Metrics
    parser.add_argument('--threshold', default=0.5, type=float, help='Threshold for pIA metric computation.')

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--multiprocessing_distributed', action='store_true')
    
    return parser.parse_args()

def save_metrics_to_file(filepath, metrics_eval):
    
    metrics = {
        "Max_Heatmap": metrics_eval['max_heatmap'],
        "Min_Heatmap": metrics_eval['min_heatmap'],
        "Median_Max_Heatmap": metrics_eval['max_median_heatmap'],
        "Median_Min_Heatmap": metrics_eval['min_median_heatmap'],
        "Max_Heatmap_Silence": metrics_eval['max_heatmap_silence'],
        "Min_Heatmap_Silence": metrics_eval['min_heatmap_silence'],
        "Median_Max_Heatmap_Silence": metrics_eval['max_median_heatmap_silence'],
        "Median_Min_Heatmap_Silence": metrics_eval['min_median_heatmap_silence'],
        "Max_Heatmap_Noise": metrics_eval['max_heatmap_noise'],
        "Min_Heatmap_Noise": metrics_eval['min_heatmap_noise'],
        "Median_Max_Heatmap_Noise": metrics_eval['max_median_heatmap_noise'],
        "Median_Min_Heatmap_Noise": metrics_eval['min_median_heatmap_noise'],
        "Max_Heatmap_Offscreen": metrics_eval['max_heatmap_offscreen'],
        "Min_Heatmap_Offscreen": metrics_eval['min_heatmap_offscreen'],
        "Median_Max_Heatmap_Offscreen": metrics_eval['max_median_heatmap_offscreen'],
        "Median_Min_Heatmap_Offscreen": metrics_eval['min_median_heatmap_offscreen'],
        "CIoU_OGL": metrics_eval['ciou_ogl'],
        "CIoU_05_OGL": metrics_eval['ciou_05_mean_ogl'],
        "CIoU_adap_OGL": metrics_eval['ciou_adap_ogl'],
        "CIoU_max_OGL": metrics_eval['ciou_max_mean_ogl'],
        "CIoU_max10_OGL": metrics_eval['ciou_max10_mean_ogl'],
        "CIoU_maxq3_OGL": metrics_eval['ciou_maxq3_mean_ogl'],
        "AUC_OGL": metrics_eval['auc_ogl'],
        "AUC_05_OGL": metrics_eval['auc_05_ogl'],
        "AUC_adap_OGL": metrics_eval['auc_adap_ogl'],
        "AUC_max_OGL": metrics_eval['auc_max_ogl'],
        "AUC_max10_OGL": metrics_eval['auc_max10_ogl'],
        "AUC_maxq3_OGL": metrics_eval['auc_maxq3_ogl'],
        "pIA_Silence_OGL": metrics_eval['pia_metric_silence_ogl'],
        "pIA_Silence_ThrCIoU_OGL": metrics_eval['pia_metric_silence_ciou_ogl'],
        "pIA_Silence_ThrCIoU_Adap_OGL": metrics_eval['pia_metric_silence_ciouadap_ogl'],
        "pIA_Silence_Max_OGL": metrics_eval['pia_metric_silence_max_ogl'],
        "pIA_Silence_Max10_OGL": metrics_eval['pia_metric_silence_max10_ogl'],
        "pIA_Silence_MaxQ3_OGL": metrics_eval['pia_metric_silence_maxq3_ogl'],
        "AOC_Silence_OGL": metrics_eval['auc_silence_ogl'],
        "AOC_Silence_ThrCIoU_OGL": metrics_eval['auc_silence_ciou_ogl'],
        "AOC_Silence_ThrCIoU_Adap_OGL": metrics_eval['auc_silence_ciouadap_ogl'],
        "AOC_Silence_Max_OGL": metrics_eval['auc_silence_max_ogl'],
        "AOC_Silence_Max10_OGL": metrics_eval['auc_silence_max10_ogl'],
        "AOC_Silence_MaxQ3_OGL": metrics_eval['auc_silence_maxq3_ogl'],
        "pIA_Noise_OG_OGLL_OGL": metrics_eval['pia_metric_noise_ogl'],
        "pIA_Noise_ThrCIoU_OGL": metrics_eval['pia_metric_noise_ciou_ogl'],
        "pIA_Noise_ThrCIoU_Adap_OGL": metrics_eval['pia_metric_noise_ciouadap_ogl'],
        "pIA_Noise_Max_OGL": metrics_eval['pia_metric_noise_max_ogl'],
        "pIA_Noise_Max10_OGL": metrics_eval['pia_metric_noise_max10_ogl'],
        "pIA_Noise_MaxQ3_OGL": metrics_eval['pia_metric_noise_maxq3_ogl'],
        "AOC_Noise_OGL": metrics_eval['auc_noise_ogl'],
        "AOC_Noise_ThrCIoU_OGL": metrics_eval['auc_noise_ciou_ogl'],
        "AOC_Noise_ThrCIoU_Adap_OGL": metrics_eval['auc_noise_ciouadap_ogl'],
        "AOC_Noise_Max_OGL": metrics_eval['auc_noise_max_ogl'],
        "AOC_Noise_Max10_OGL": metrics_eval['auc_noise_max10_ogl'],
        "AOC_Noise_MaxQ3_OGL": metrics_eval['auc_noise_maxq3_ogl'],
        "pIA_OffScreen_OGL": metrics_eval['pia_metric_offscreen_ogl'],
        "pIA_OffScreen_ThrCIoU_OGL": metrics_eval['pia_metric_offscreen_ciou_ogl'],
        "pIA_OffScreen_ThrCIoU_Adap_OGL": metrics_eval['pia_metric_offscreen_ciouadap_ogl'],
        "pIA_OffScreen_Max_OGL": metrics_eval['pia_metric_offscreen_max_ogl'],
        "pIA_OffScreen_Max10_OGL": metrics_eval['pia_metric_offscreen_max10_ogl'],
        "pIA_OffScreen_MaxQ3_OGL": metrics_eval['pia_metric_offscreen_maxq3_ogl'],
        "AOC_Offscreen_OGL": metrics_eval['auc_offscreen_ogl'],
        "AOC_Offscreen_ThrCIoU_OGL": metrics_eval['auc_offscreen_ciou_ogl'],
        "AOC_Offscreen_ThrCIoU_Adap_OGL": metrics_eval['auc_offscreen_ciouadap_ogl'],
        "AOC_OffScreen_Max_OGL": metrics_eval['auc_offscreen_max_ogl'],
        "AOC_OffScreen_Max10_OGL": metrics_eval['auc_offscreen_max10_ogl'],
        "AOC_OffScreen_MaxQ3_OGL": metrics_eval['auc_offscreen_maxq3_ogl'],
        "IoU_Pos_Silence_OGL": metrics_eval['iou_pos_silence_ogl_metric'],
        "IoU_Pos_Silence_Adap_OGL": metrics_eval['iou_pos_silence_adap_ogl_metric'],
        "IoU_Pos_Noise_OGL": metrics_eval['iou_pos_noise_ogl_metric'],
        "IoU_Pos_Noise_Adap_OGL": metrics_eval['iou_pos_noise_adap_ogl_metric'],
        "IoU_Pos_Offscreen_OGL": metrics_eval['iou_pos_offscreen_ogl_metric'],
        "IoU_Pos_Offscreen_Adap_OGL": metrics_eval['iou_pos_offscreen_adap_ogl_metric'],
        "IoU_Silence_Noise_OGL": metrics_eval['iou_silence_noise_ogl_metric'],
        "IoU_Silence_Noise_Adap_OGL": metrics_eval['iou_silence_noise_adap_ogl_metric'],
        "IoU_Silence_Offscreen_OGL": metrics_eval['iou_silence_offscreen_ogl_metric'],
        "IoU_Silence_Offscreen_Adap_OGL": metrics_eval['iou_silence_offscreen_adap_ogl_metric'],
        "IoU_Noise_Offscreen_OGL": metrics_eval['iou_noise_offscreen_ogl_metric'],
        "IoU_Noise_Offscreen_Adap_OGL": metrics_eval['iou_noise_offscreen_adap_ogl_metric'],
        "CIoU": metrics_eval['ciou'],
        "CIoU_05": metrics_eval['ciou_05_mean_wogl'],
        "CIoU_adap": metrics_eval['ciou_adap'],
        "CIoU_max": metrics_eval['ciou_max_mean_wogl'],
        "CIoU_max10": metrics_eval['ciou_max10_mean_wogl'],
        "CIoU_maxq3": metrics_eval['ciou_maxq3_mean_wogl'],
        "F_Pos_Neg": metrics_eval['f_pos_neg'],
        "F_Pos_Silence": metrics_eval['f_ciou_pos_silence'],
        "F_Pos_Noise": metrics_eval['f_ciou_pos_noise'],
        "F_Pos_OffScreen": metrics_eval['f_ciou_pos_offscreen'],
        "F_AUC_AOC_Pos_Neg": metrics_eval['f_auc_aoc_pos_neg'],
        "F_AUC_AOC_Pos_Silence": metrics_eval['f_auc_aoc_pos_silence'],
        "F_AUC_AOC_Pos_Noise": metrics_eval['f_auc_aoc_pos_noise'],
        "F_AUC_AOC_Pos_OffScreen": metrics_eval['f_auc_aoc_pos_offscreen'],
        "AUC": metrics_eval['auc'],
        "AUC_05": metrics_eval['auc_05_wogl'],
        "AUC_adap": metrics_eval['auc_adap'],
        "AUC_max": metrics_eval['auc_max_wogl'],
        "AUC_max10": metrics_eval['auc_max10_wogl'],
        "AUC_maxq3": metrics_eval['auc_maxq3_wogl'],
        "pIA_Silence": metrics_eval['pia_metric_silence_wogl'],
        "pIA_Silence_ThrCIoU": metrics_eval['pia_metric_silence_ciou_wogl'],
        "pIA_Silence_ThrCIoU_Adap": metrics_eval['pia_metric_silence_ciouadap_wogl'],
        "pIA_Silence_Max": metrics_eval['pia_metric_silence_max_wogl'],
        "pIA_Silence_Max10": metrics_eval['pia_metric_silence_max10_wogl'],
        "pIA_Silence_MaxQ3": metrics_eval['pia_metric_silence_maxq3_wogl'],
        "AOC_Silence": metrics_eval['auc_silence_wogl'],
        "AOC_Silence_ThrCIoU": metrics_eval['auc_silence_ciou_wogl'],
        "AOC_Silence_ThrCIoU_Adap": metrics_eval['auc_silence_ciouadap_wogl'],
        "AOC_Silence_Max": metrics_eval['auc_silence_max_wogl'],
        "AOC_Silence_Max10": metrics_eval['auc_silence_max10_wogl'],
        "AOC_Silence_MaxQ3": metrics_eval['auc_silence_maxq3_wogl'],
        "pIA_Noise": metrics_eval['pia_metric_noise_wogl'],
        "pIA_Noise_ThrCIoU": metrics_eval['pia_metric_noise_ciou_wogl'],
        "pIA_Noise_ThrCIoU_Adap": metrics_eval['pia_metric_noise_ciouadap_wogl'],
        "pIA_Noise_Max": metrics_eval['pia_metric_noise_max_wogl'],
        "pIA_Noise_Max10": metrics_eval['pia_metric_noise_max10_wogl'],
        "pIA_Noise_MaxQ3": metrics_eval['pia_metric_noise_maxq3_wogl'],
        "AOC_Noise": metrics_eval['auc_noise_wogl'],
        "AOC_Noise_ThrCIoU": metrics_eval['auc_noise_ciou_wogl'],
        "AOC_Noise_ThrCIoU_Adap": metrics_eval['auc_noise_ciouadap_wogl'],
        "AOC_Noise_Max": metrics_eval['auc_noise_max_wogl'],
        "AOC_Noise_Max10": metrics_eval['auc_noise_max10_wogl'],
        "AOC_Noise_MaxQ3": metrics_eval['auc_noise_maxq3_wogl'],
        "pIA_OffScreen": metrics_eval['pia_metric_offscreen_wogl'],
        "pIA_OffScreen_ThrCIoU": metrics_eval['pia_metric_offscreen_ciou_wogl'],
        "pIA_OffScreem_ThrCIoU_Adap": metrics_eval['pia_metric_offscreen_ciouadap_wogl'],
        "pIA_OffScreen_Max": metrics_eval['pia_metric_offscreen_max_wogl'],
        "pIA_OffScreen_Max10": metrics_eval['pia_metric_offscreen_max10_wogl'],
        "pIA_OffScreen_MaxQ3": metrics_eval['pia_metric_offscreen_maxq3_wogl'],
        "AOC_Offscreen": metrics_eval['auc_offscreen_wogl'],
        "AOC_Offscreen_ThrCIoU": metrics_eval['auc_offscreen_ciou_wogl'],
        "AOC_Offscreen_ThrCIoU_Adap": metrics_eval['auc_offscreen_ciouadap_wogl'],
        "AOC_OffScreen_Max": metrics_eval['auc_offscreen_max_wogl'],
        "AOC_OffScreen_Max10": metrics_eval['auc_offscreen_max10_wogl'],
        "AOC_OffScreen_MaxQ3": metrics_eval['auc_offscreen_maxq3_wogl'],
        "IoU_Pos_Silence_WOGL": metrics_eval['iou_pos_silence_wogl_metric'],
        "IoU_Pos_Silence_Max_WOGL": metrics_eval['iou_pos_silence_max_wogl_metric'],
        "IoU_Pos_Silence_Max_10_WOGL": metrics_eval['iou_pos_silence_max10_wogl_metric'],
        "IoU_Pos_Silence_Max_Q3_WOGL": metrics_eval['iou_pos_silence_maxq3_wogl_metric'],
        "IoU_Pos_Silence_Adap_WOGL": metrics_eval['iou_pos_silence_adap_wogl_metric'],
        "IoU_Pos_Noise_WOGL": metrics_eval['iou_pos_noise_wogl_metric'],
        "IoU_Pos_Noise_Max_WOGL": metrics_eval['iou_pos_noise_max_wogl_metric'],
        "IoU_Pos_Noise_Max_10_WOGL": metrics_eval['iou_pos_noise_max10_wogl_metric'],
        "IoU_Pos_Noise_Max_Q3_WOGL": metrics_eval['iou_pos_noise_maxq3_wogl_metric'],
        "IoU_Pos_Noise_Adap_WOGL": metrics_eval['iou_pos_noise_adap_wogl_metric'],
        "IoU_Pos_Offscreen_WOGL": metrics_eval['iou_pos_offscreen_wogl_metric'],
        "IoU_Pos_Offscreen_Max_WOGL": metrics_eval['iou_pos_offscreen_max_wogl_metric'],
        "IoU_Pos_Offscreen_Max_10_WOGL": metrics_eval['iou_pos_offscreen_max10_wogl_metric'],
        "IoU_Pos_Offscreen_Max_Q3_WOGL": metrics_eval['iou_pos_offscreen_maxq3_wogl_metric'],
        "IoU_Pos_Offscreen_Adap_WOGL": metrics_eval['iou_pos_offscreen_adap_wogl_metric'],
        "IoU_Silence_Noise_WOGL": metrics_eval['iou_silence_noise_wogl_metric'],
        "IoU_Silence_Noise_Max_WOGL": metrics_eval['iou_silence_noise_max_wogl_metric'],
        "IoU_Silence_Noise_Max_10_WOGL": metrics_eval['iou_silence_noise_max10_wogl_metric'],
        "IoU_Silence_Noise_Max_Q3_WOGL": metrics_eval['iou_silence_noise_maxq3_wogl_metric'],
        "IoU_Silence_Noise_Adap_WOGL": metrics_eval['iou_silence_noise_adap_wogl_metric'],
        "IoU_Silence_Offscreen_WOGL": metrics_eval['iou_silence_offscreen_wogl_metric'],
        "IoU_Silence_Offscreen_Max_WOGL": metrics_eval['iou_silence_offscreen_max_wogl_metric'],
        "IoU_Silence_Offscreen_Max_10_WOGL": metrics_eval['iou_silence_offscreen_max10_wogl_metric'],
        "IoU_Silence_Offscreen_Max_Q3_WOGL": metrics_eval['iou_silence_offscreen_maxq3_wogl_metric'],
        "IoU_Silence_Offscreen_Adap_WOGL": metrics_eval['iou_silence_offscreen_adap_wogl_metric'],
        "IoU_Noise_Offscreen_WOGL": metrics_eval['iou_noise_offscreen_wogl_metric'],
        "IoU_Noise_Offscreen_Max_WOGL": metrics_eval['iou_noise_offscreen_max_wogl_metric'],
        "IoU_Noise_Offscreen_Max_10_WOGL": metrics_eval['iou_noise_offscreen_max10_wogl_metric'],
        "IoU_Noise_Offscreen_Max_Q3_WOGL": metrics_eval['iou_noise_offscreen_maxq3_wogl_metric'],
        "IoU_Noise_Offscreen_Adap_WOGL": metrics_eval['iou_noise_offscreen_adap_wogl_metric'],
        "iiou_adap": metrics_eval['iiou_adap'],
        "IAUC_adap": metrics_eval['iauc_adap'],
    }
    
    with open(filepath, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.2f}\n")
            
def save_max_min_numpy(args, model_name, metrics):
    max_min_dir = 'metrics/max_min'
    os.makedirs(max_min_dir, exist_ok=True)
    
    max_pos_out_path = os.path.join(max_min_dir, f"{model_name}_{args.testset}_max_pos.npy")
    min_pos_out_path = os.path.join(max_min_dir, f"{model_name}_{args.testset}_min_pos.npy")

    max_silence_out_path = os.path.join(max_min_dir, f"{model_name}_{args.testset}_max_silence.npy")
    min_silence_out_path = os.path.join(max_min_dir, f"{model_name}_{args.testset}_min_silence.npy")

    max_noise_out_path = os.path.join(max_min_dir, f"{model_name}_{args.testset}_max_noise.npy")
    min_noise_out_path = os.path.join(max_min_dir, f"{model_name}_{args.testset}_min_noise.npy")

    max_offscreen_out_path = os.path.join(max_min_dir, f"{model_name}_{args.testset}_max_offscreen.npy")
    min_offscreen_out_path = os.path.join(max_min_dir, f"{model_name}_{args.testset}_min_offscreen.npy")

    np.save(max_pos_out_path, metrics['max_heatmap_list'])
    np.save(min_pos_out_path, metrics['min_heatmap_list'])
    np.save(max_silence_out_path, metrics['max_heatmap_silence_list'])
    np.save(min_silence_out_path, metrics['min_heatmap_silence_list'])
    np.save(max_noise_out_path, metrics['max_heatmap_noise_list'])
    np.save(min_noise_out_path, metrics['min_heatmap_noise_list'])
    np.save(max_offscreen_out_path, metrics['max_heatmap_offscreen_list'])
    np.save(min_offscreen_out_path, metrics['min_heatmap_offscreen_list'])

def main(args):
    # Fixing random seed for reproducibility
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Adding thresholds
    args.ezvsl_th_max = 0.6041
    args.ezvsl_th_max10 = 0.6645
    args.ezvsl_th_maxq3 = 0.6382
    # args.ezvsl_th_maxq3 = 0.6607

    args.slavc_th_max = 0.5201
    args.slavc_th_max10 = 0.5721
    args.slavc_th_maxq3 = 0.5450
    # args.slavc_th_maxq3 = 0.5513

    args.fnac_th_max = 0.5311
    args.fnac_th_max10 = 0.5842
    args.fnac_th_maxq3 = 0.5538
    # args.fnac_th_maxq3 = 0.5565

    args.lvs_th_max = 0.5527
    args.lvs_th_max10 = 0.6080
    args.lvs_th_maxq3 = 0.6077
    # args.lvs_th_maxq3 = 0.6724

    args.ssltie_th_max = 0.2641
    args.ssltie_th_max10 = 0.2906
    args.ssltie_th_maxq3 = 0.3338
    # args.ssltie_th_maxq3 = 0.3825

    args.sslalign_th_max = 0.2660
    args.sslalign_th_max10 = 0.2926
    args.sslalign_th_maxq3 = 0.3175
    # args.sslalign_th_maxq3 = 0.3388

    # Models
    if 'ssltie' in args.pth_name:
        ssltie_args=SSLTIE_args()
        ssltie_args=ssltie_args.ssltie_args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        audio_visual_model = AVENet_ssltie(ssltie_args)
        audio_visual_model = audio_visual_model.cuda()
        audio_visual_model = nn.DataParallel(audio_visual_model)
        ssltie_args.test_gt_path = args.test_gt_path
        ssltie_args.data_dir = args.data_dir
        ssltie_args.testset = args.testset
        ssltie_args.dataset_mode = args.testset
        ssltie_args.pth_name = args.pth_name
        ssltie_args.threshold = args.threshold
        ssltie_args.ssltie_th_max = args.ssltie_th_max,
        ssltie_args.ssltie_th_max10 = args.ssltie_th_max10,
        ssltie_args.ssltie_th_maxq3 = args.ssltie_th_maxq3,
        
    elif 'lvs' in args.pth_name or 'sslalign' in args.pth_name:
        lvs_args = easydict.EasyDict({
        'data_dir' : args.data_dir,
        "image_size": 224,
        "batch_size" : 64,
        "n_threads" : 10,
        "epsilon" : 0.65,
        "epsilon2" : 0.4,
        'tri_map' : True,
        'Neg' : True,
        'random_threshold' : 1,
        'soft_ep' : 1,
        'test_gt_path' : args.test_gt_path,
        'testset' : args.testset,
        'pth_name' : args.pth_name,
        'threshold': args.threshold,
        'lvs_th_max': args.lvs_th_max,
        'lvs_th_max10': args.lvs_th_max10,
        'lvs_th_maxq3': args.lvs_th_maxq3,
        'sslalign_th_max': args.sslalign_th_max,
        'sslalign_th_max10': args.sslalign_th_max10,
        'sslalign_th_maxq3': args.sslalign_th_maxq3,
        })

        audio_visual_model= AVENet(lvs_args) 
    elif 'ezvsl' in args.pth_name or 'margin' in args.pth_name:
        audio_visual_model = EZVSL(0.03,512)
    elif 'slavc' in args.pth_name:
        audio_visual_model = SLAVC(0.03, 512, 0, 0, 0.9, 0.9, False, None)
    elif 'fnac' in args.pth_name:
        audio_visual_model = FNAC(0.03,512,0,0)
    else:
        audio_visual_model = EZVSL(0.03,512)

    object_saliency_model = resnet18(pretrained=True)
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        audio_visual_model.cuda(args.gpu)
        object_saliency_model.cuda(args.gpu)
    if args.multiprocessing_distributed:
        audio_visual_model = torch.nn.parallel.DistributedDataParallel(audio_visual_model, device_ids=[args.gpu])
        object_saliency_model = torch.nn.parallel.DistributedDataParallel(object_saliency_model, device_ids=[args.gpu])

    # Load weights
    if 'ssltie' in args.pth_name:
        ckp_fn = os.path.join(args.pth_name)
        checkpoint = torch.load(ckp_fn, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        
        model_without_dp = audio_visual_model.module
        model_without_dp.load_state_dict(state_dict)
        
    elif 'lvs' in args.pth_name:
        ckp_fn = os.path.join(args.pth_name)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        audio_visual_model = nn.DataParallel(audio_visual_model)
        audio_visual_model = audio_visual_model.cuda()
        checkpoint = torch.load(ckp_fn)
        model_dict = audio_visual_model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']            
        model_dict.update(pretrained_dict)
        audio_visual_model.load_state_dict(model_dict)
        audio_visual_model.to(device)
    
    elif 'sslalign' in args.pth_name:
        ckp_fn = os.path.join(args.pth_name)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        audio_visual_model = nn.DataParallel(audio_visual_model)
        audio_visual_model = audio_visual_model.cuda()
        checkpoint = torch.load(ckp_fn)
        model_dict = audio_visual_model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        pretrained_dict = {'module.'+x:pretrained_dict[x] for x in pretrained_dict}
        model_dict.update(pretrained_dict)
        audio_visual_model.load_state_dict(model_dict,strict=False)
        audio_visual_model.to(device)

    else:
        ckp_fn = os.path.join(args.pth_name)
        if os.path.exists(ckp_fn):
            ckp = torch.load(ckp_fn, map_location='cpu')
            audio_visual_model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        else:
            print(f"Checkpoint not found: {ckp_fn}")
    print(ckp_fn,' loaded')
    args.files_list = []
    
    if 'lvs' in args.pth_name or 'sslalign' in args.pth_name:
        eval_lvs(audio_visual_model, object_saliency_model, lvs_args)
    elif 'ssltie' in args.pth_name:
        eval_ssltie(audio_visual_model, object_saliency_model, ssltie_args)
    else:
        eval(audio_visual_model, object_saliency_model, args)

@torch.no_grad()
def eval_lvs(model, object_saliency_model, args):
    random_seeds = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 9999]
    
    model.train(False)
    object_saliency_model.train(False)
    
    evaluator = utils.Evaluator_iiou()
    evaluator_ogl = utils.Evaluator_iiou()
    
    metrics_eval = define_metrics()

    for random_seed in tqdm(random_seeds):
        random.seed(random_seed)
        testdataset = GetAudioVideoDataset(args,  mode='test')
        args.image_path = testdataset.image_path
        testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
        
        for step, batch in enumerate(tqdm(testdataloader)):
            image, spec, spec_noise, spec_silence, spec_offscreen, bboxes, name = batch
            
            spec = Variable(spec).cuda()
            spec_noise = Variable(spec_noise).cuda()
            spec_silence = Variable(spec_silence).cuda()
            
            image = Variable(image).cuda()
            
            heatmap,_,Pos,Neg = model(image.float(),spec.float(),args)
            heatmap_noise,_,Pos,Neg = model(image.float(),spec_noise.float(),args)
            heatmap_silence,_,Pos,Neg = model(image.float(),spec_silence.float(),args)
            heatmap_offscreen,_,Pos,Neg = model(image.float(),spec_offscreen.float(),args)

            for htmp, htmp_noise, htmp_silence, htmp_offscreen in zip(heatmap, heatmap_noise, heatmap_silence, heatmap_offscreen):

                metrics_eval['max_heatmap_list'].append(torch.max(htmp).item())
                metrics_eval['max_heatmap_silence_list'].append(torch.max(htmp_silence).item())
                metrics_eval['max_heatmap_noise_list'].append(torch.max(htmp_noise).item())
                metrics_eval['max_heatmap_offscreen_list'].append(torch.max(htmp_offscreen).item())

                metrics_eval['min_heatmap_list'].append(torch.min(htmp).item())
                metrics_eval['min_heatmap_silence_list'].append(torch.min(htmp_silence).item())
                metrics_eval['min_heatmap_noise_list'].append(torch.min(htmp_noise).item())
                metrics_eval['min_heatmap_offscreen_list'].append(torch.min(htmp_offscreen).item())
            
            heatmap_obj_ = object_saliency_model(image)
            
            heatmap_forextend = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=True)
            heatmap_forextend = heatmap_forextend.data.cpu().numpy()
            
            if args.testset == 'vggss':
                heatmap_arr =  heatmap.data.cpu().numpy()
                heatmap_obj = heatmap_obj_.data.cpu().numpy()

                for i in range(spec.shape[0]):
                    heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                    gt_map = bboxes['gt_map'][i].data.cpu().numpy()
                    pred = heatmap_now
                    threshold_ciou_wogl = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
                    evaluator.cal_CIOU(pred,gt_map,threshold_ciou_wogl)
                    ciou_05_val, _, _ = evaluator.cal_CIOU(pred,gt_map,0.5)
                    if np.isnan(ciou_05_val):
                        metrics_eval['ciou_05_wogl'].append(0)
                    else:
                        metrics_eval['ciou_05_wogl'].append(ciou_05_val)

                    if 'lvs' in args.pth_name:
                        ciou_max_neg_val, _, _ = evaluator.cal_CIOU(pred,gt_map,args.lvs_th_max)
                        ciou_max10_neg_val, _, _ = evaluator.cal_CIOU(pred,gt_map,args.lvs_th_max10)
                        ciou_maxq3_neg_val, _, _ = evaluator.cal_CIOU(pred,gt_map,args.lvs_th_maxq3)
                    elif 'sslalign' in args.pth_name:
                        ciou_max_neg_val, _, _ = evaluator.cal_CIOU(pred,gt_map,args.sslalign_th_max)
                        ciou_max10_neg_val, _, _ = evaluator.cal_CIOU(pred,gt_map,args.sslalign_th_max10)
                        ciou_maxq3_neg_val, _, _ = evaluator.cal_CIOU(pred,gt_map,args.sslalign_th_maxq3)
                    
                    if np.isnan(ciou_max_neg_val):
                        metrics_eval['ciou_max_wogl'].append(0)
                    else:
                        metrics_eval['ciou_max_wogl'].append(ciou_max_neg_val)
                    
                    if np.isnan(ciou_max10_neg_val):
                        metrics_eval['ciou_max10_wogl'].append(0)
                    else:
                        metrics_eval['ciou_max10_wogl'].append(ciou_max10_neg_val)
                    
                    if np.isnan(ciou_maxq3_neg_val):
                        metrics_eval['ciou_maxq3_wogl'].append(0)
                    else:
                        metrics_eval['ciou_maxq3_wogl'].append(ciou_maxq3_neg_val)

                    gt_nums = (gt_map!=0).sum()
                    if int(gt_nums) == 0:
                        gt_nums = int(pred.shape[0] * pred.shape[1])//2
                    threshold_ciouadap_wogl = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1])-int(gt_nums)] # adap
                    evaluator.cal_CIOU_adap(pred,gt_map,threshold_ciouadap_wogl)

                    short_name = '_'.join(name[i].split('/')[-1].replace('.jpg','').split('_')[:-1])
                    if short_name in evaluator.iou.keys():
                        evaluator.iou_adap[short_name].append(evaluator.ciou_adap[-1])
                        evaluator.iou[short_name].append(evaluator.ciou[-1])
                    else:
                        evaluator.iou_adap[short_name] = [evaluator.ciou_adap[-1]]
                        evaluator.iou[short_name] = [evaluator.ciou[-1]]
                
                    heatmap_obj_now = cv2.resize(heatmap_obj[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

                    pred_obj = heatmap_obj_now
                    pred_av_obj = (pred * 0.4 + pred_obj * (1 - 0.4))
                    threshold_ciou_ogl = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] / 2)]
                    evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,threshold_ciou_ogl)
                    ciou_05_ogl_val, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,0.5)
                    metrics_eval['ciou_05_ogl'].append(ciou_05_ogl_val)

                    if 'lvs' in args.pth_name:
                        ciou_max_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.lvs_th_max)
                        ciou_max10_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.lvs_th_max10)
                        ciou_maxq3_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.lvs_th_maxq3)
                    elif 'sslalign' in args.pth_name:
                        ciou_max_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.sslalign_th_max)
                        ciou_max10_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.sslalign_th_max10)
                        ciou_maxq3_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.sslalign_th_maxq3)
                    
                    metrics_eval['ciou_max_ogl'].append(ciou_max_neg_val_ogl)
                    metrics_eval['ciou_max10_ogl'].append(ciou_max10_neg_val_ogl)
                    metrics_eval['ciou_maxq3_ogl'].append(ciou_maxq3_neg_val_ogl)

                    threshold_ciouadap_ogl = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1])-int(gt_nums)]
                    evaluator_ogl.cal_CIOU_adap(pred_av_obj,gt_map,threshold_ciouadap_ogl)
                    
                    binarized_negatives = binarized_negatives_thresholds(i, args, pred, pred_av_obj, heatmap_silence, heatmap_noise, heatmap_offscreen, pred_obj, threshold_ciou_wogl, threshold_ciou_ogl, threshold_ciouadap_wogl, threshold_ciouadap_ogl)
                    metrics_eval = append_metrics_binarized_pred(metrics_eval, binarized_negatives)
            
            else:
                heatmap = nn.functional.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=True)
                heatmap_obj = nn.functional.interpolate(heatmap_obj_, size=(224, 224), mode='bilinear', align_corners=True)
                
                for i in range(spec.shape[0]):
                    heatmap_now = heatmap[i:i+1].cpu().numpy()
                    gt_map = bboxes['gt_map'][i].data.cpu().numpy()
                    pred = heatmap_now[0][0]
                    threshold_ciou_wogl = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
                    evaluator.cal_CIOU(pred,gt_map,threshold_ciou_wogl)
                    ciou_05_val, _, _ = evaluator.cal_CIOU(pred,gt_map,0.5)
                    if np.isnan(ciou_05_val):
                        metrics_eval['ciou_05_wogl'].append(0)
                    else:
                        metrics_eval['ciou_05_wogl'].append(ciou_05_val)

                    if 'lvs' in args.pth_name:
                        ciou_max_neg_val_wogl, _, _ = evaluator.cal_CIOU(pred,gt_map,args.lvs_th_max)
                        ciou_max10_neg_val_wogl, _, _ = evaluator.cal_CIOU(pred,gt_map,args.lvs_th_max10)
                        ciou_maxq3_neg_val_wogl, _, _ = evaluator.cal_CIOU(pred,gt_map,args.lvs_th_maxq3)
                    elif 'sslalign' in args.pth_name:
                        ciou_max_neg_val_wogl, _, _ = evaluator.cal_CIOU(pred,gt_map,args.sslalign_th_max)
                        ciou_max10_neg_val_wogl, _, _ = evaluator.cal_CIOU(pred,gt_map,args.sslalign_th_max10)
                        ciou_maxq3_neg_val_wogl, _, _ = evaluator.cal_CIOU(pred,gt_map,args.sslalign_th_maxq3)
                    
                    if np.isnan(ciou_max_neg_val_wogl):
                        metrics_eval['ciou_max_wogl'].append(0)
                    else:
                        metrics_eval['ciou_max_wogl'].append(ciou_max_neg_val_wogl)
                    
                    if np.isnan(ciou_max10_neg_val_wogl):
                        metrics_eval['ciou_max10_wogl'].append(0)
                    else:
                        metrics_eval['ciou_max10_wogl'].append(ciou_max10_neg_val_wogl)
                    
                    if np.isnan(ciou_maxq3_neg_val_wogl):
                        metrics_eval['ciou_maxq3_wogl'].append(0)
                    else:
                        metrics_eval['ciou_maxq3_wogl'].append(ciou_maxq3_neg_val_wogl)

                    gt_nums = (gt_map!=0).sum()
                    if int(gt_nums) == 0:
                        gt_nums = int(pred.shape[0] * pred.shape[1])//2
                    threshold_ciouadap_wogl = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1])-int(gt_nums)] # adap
                    evaluator.cal_CIOU_adap(pred,gt_map,threshold_ciouadap_wogl)

                    short_name = '_'.join(name[i].split('/')[-1].replace('.jpg','').split('_')[:-1])
                    if short_name in evaluator.iou.keys():
                        evaluator.iou_adap[short_name].append(evaluator.ciou_adap[-1])
                        evaluator.iou[short_name].append(evaluator.ciou[-1])
                    else:
                        evaluator.iou_adap[short_name] = [evaluator.ciou_adap[-1]]
                        evaluator.iou[short_name] = [evaluator.ciou[-1]]
                
                    heatmap_obj_now = heatmap_obj[i:i+1].cpu().numpy()
                    pred_obj = heatmap_obj_now[0][0]
                    pred_av_obj = (pred * 0.4 + pred_obj * (1 - 0.4))
                    
                    threshold_ciou_ogl = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] / 2)]
                    evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,threshold_ciou_ogl)
                    ciou_05_ogl_val, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,0.5)
                    metrics_eval['ciou_05_ogl'].append(ciou_05_ogl_val)

                    if 'lvs' in args.pth_name:
                        ciou_max_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.lvs_th_max)
                        ciou_max10_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.lvs_th_max10)
                        ciou_maxq3_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.lvs_th_maxq3)
                    elif 'sslalign' in args.pth_name:
                        ciou_max_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.sslalign_th_max)
                        ciou_max10_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.sslalign_th_max10)
                        ciou_maxq3_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.sslalign_th_maxq3)
                    
                    metrics_eval['ciou_max_ogl'].append(ciou_max_neg_val_ogl)
                    metrics_eval['ciou_max10_ogl'].append(ciou_max10_neg_val_ogl)
                    metrics_eval['ciou_maxq3_ogl'].append(ciou_maxq3_neg_val_ogl)

                    threshold_ciouadap_ogl = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1])-int(gt_nums)]
                    evaluator_ogl.cal_CIOU_adap(pred_av_obj,gt_map,threshold_ciouadap_ogl)
                    
                    binarized_negatives = binarized_negatives_thresholds(i, args, pred, pred_av_obj, heatmap_silence, heatmap_noise, heatmap_offscreen, pred_obj, threshold_ciou_wogl, threshold_ciou_ogl, threshold_ciouadap_wogl, threshold_ciouadap_ogl)
                    metrics_eval = append_metrics_binarized_pred(metrics_eval, binarized_negatives)
        
    model_name = args.pth_name.split("/")[-1].split(".")[0]
    
    if model_name == "ours_sup_previs":
        model_name = "sslalign"
    elif model_name == "lvs_vggss":
        model_name = "lvs"

    save_max_min_numpy(args, model_name, metrics_eval)
    
    ciou = evaluator.finalize_cIoU()        
    auc,auc_adap = evaluator.finalize_AUC()    
    iauc_adap = evaluator.finalize_IAUC_adap()
    
    ciou,ciou_adap,_,iiou_adap = ciou
    
    metrics_eval['ciou'] = ciou * 100
    metrics_eval['ciou_adap'] = ciou_adap * 100
    metrics_eval['iiou_adap'] = iiou_adap * 100
    metrics_eval['auc'] = auc * 100
    metrics_eval['auc_adap'] = auc_adap * 100
    metrics_eval['iauc_adap'] = iauc_adap * 100
    
    metrics_eval['ciou_ogl'] = evaluator_ogl.finalize_cIoU()        
    metrics_eval['auc_ogl'], metrics_eval['auc_adap_ogl'] = evaluator_ogl.finalize_AUC()
    metrics_eval['ciou_ogl'], metrics_eval['ciou_adap_ogl'],_,_ = metrics_eval['ciou_ogl']
    metrics_eval['ciou_ogl'] = metrics_eval['ciou_ogl']*100
    metrics_eval['ciou_adap_ogl'] = metrics_eval['ciou_adap_ogl']*100
    metrics_eval['auc_ogl'] = metrics_eval['auc_ogl']*100
    metrics_eval['auc_adap_ogl'] = metrics_eval['auc_adap_ogl']*100
    
    metrics_eval = compute_metrics(metrics_eval)

    metrics_out_path = os.path.join("metrics", f"{model_name}_{args.testset}.txt")
    print(f"Metrics stored in {metrics_out_path}!")
    save_metrics_to_file(metrics_out_path, metrics_eval)

@torch.no_grad()
def eval_ssltie(model, object_saliency_model, args):
    
    random_seeds = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 9999]
            
    model.train(False)
    object_saliency_model.train(False)
    
    evaluator = utils.Evaluator_iiou()
    evaluator_ogl = utils.Evaluator_iiou()
    # evaluator_adap = utils.Evaluator_iiou()
    
    metrics_eval = define_metrics()
    
    for random_seed in tqdm(random_seeds):
        random.seed(random_seed)
        testdataset = GetAudioVideoDataset_ssltie(args, mode='test')
        args.image_path = testdataset.image_path
        testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
        
        for step, (image, spec, spec_noise, spec_silence, spec_offscreen, bboxes, name) in enumerate(tqdm(testdataloader)):
            spec = Variable(spec).cuda()
            image = Variable(image).cuda()
            
            heatmap, out, Pos, Neg, out_ref = model(image.float(), spec.float(), args, mode='val')
            heatmap_noise, out, Pos, Neg, out_ref = model(image.float(), spec_noise.float(), args, mode='val')
            heatmap_silence, out, Pos, Neg, out_ref = model(image.float(), spec_silence.float(), args, mode='val')
            heatmap_offscreen, out, Pos, Neg, out_ref = model(image.float(), spec_offscreen.float(), args, mode='val')
            
            for htmp, htmp_noise, htmp_silence, htmp_offscreen in zip(heatmap, heatmap_noise, heatmap_silence, heatmap_offscreen):

                metrics_eval['max_heatmap_list'].append(torch.max(htmp).item())
                metrics_eval['max_heatmap_silence_list'].append(torch.max(htmp_silence).item())
                metrics_eval['max_heatmap_noise_list'].append(torch.max(htmp_noise).item())
                metrics_eval['max_heatmap_offscreen_list'].append(torch.max(htmp_offscreen).item())

                metrics_eval['min_heatmap_list'].append(torch.min(htmp).item())
                metrics_eval['min_heatmap_silence_list'].append(torch.min(htmp_silence).item())
                metrics_eval['min_heatmap_noise_list'].append(torch.min(htmp_noise).item())
                metrics_eval['min_heatmap_offscreen_list'].append(torch.min(htmp_offscreen).item())

            heatmap_arr =  heatmap.data.cpu().numpy()
            
            heatmap_obj = object_saliency_model(image)
            heatmap_obj = heatmap_obj.data.cpu().numpy()
            for i in range(spec.shape[0]):
                heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                gt_map = bboxes['gt_map'][i].data.cpu().numpy()#testset_gt(args,name[i])
                pred = heatmap_now
                threshold_ciou_wogl = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]

                evaluator.cal_CIOU(pred,gt_map,threshold_ciou_wogl)
                ciou_05_val, _, _ = evaluator.cal_CIOU(pred,gt_map,0.5)
                if np.isnan(ciou_05_val):
                    metrics_eval['ciou_05_wogl'].append(0)
                else:
                    metrics_eval['ciou_05_wogl'].append(ciou_05_val)

                ciou_max_neg_val_wogl, _, _ = evaluator.cal_CIOU(pred,gt_map,args.ssltie_th_max)
                ciou_max10_neg_val_wogl, _, _ = evaluator.cal_CIOU(pred,gt_map,args.ssltie_th_max10)
                ciou_maxq3_neg_val_wogl, _, _ = evaluator.cal_CIOU(pred,gt_map,args.ssltie_th_maxq3)
                
                if np.isnan(ciou_max_neg_val_wogl):
                    metrics_eval['ciou_max_wogl'].append(0)
                else:
                    metrics_eval['ciou_max_wogl'].append(ciou_max_neg_val_wogl)
                
                if np.isnan(ciou_max10_neg_val_wogl):
                    metrics_eval['ciou_max10_wogl'].append(0)
                else:
                    metrics_eval['ciou_max10_wogl'].append(ciou_max10_neg_val_wogl)
                
                if np.isnan(ciou_maxq3_neg_val_wogl):
                    metrics_eval['ciou_maxq3_wogl'].append(0)
                else:
                    metrics_eval['ciou_maxq3_wogl'].append(ciou_maxq3_neg_val_wogl)
                
                gt_nums = (gt_map!=0).sum()
                if int(gt_nums) == 0:
                    gt_nums = int(pred.shape[0] * pred.shape[1])//2
                threshold_ciouadap_wogl = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1])-int(gt_nums)] # adap
 
                evaluator.cal_CIOU_adap(pred,gt_map,threshold_ciouadap_wogl)
                short_name = '_'.join(name[i].split('/')[-1].replace('.jpg','').split('_')[:-1])
                if short_name in evaluator.iou.keys():
                    evaluator.iou_adap[short_name].append(evaluator.ciou_adap[-1])
                    evaluator.iou[short_name].append(evaluator.ciou[-1])
                else:
                    evaluator.iou_adap[short_name] = [evaluator.ciou_adap[-1]]
                    evaluator.iou[short_name] = [evaluator.ciou[-1]]
                
                heatmap_obj_now = cv2.resize(heatmap_obj[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                
                pred_obj = heatmap_obj_now
                pred_av_obj = (pred * 0.4 + pred_obj * (1 - 0.4))
                threshold_ciou_ogl = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] / 2)]
                evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,threshold_ciou_ogl)
                ciou_05_ogl_val, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,0.5)
                metrics_eval['ciou_05_ogl'].append(ciou_05_ogl_val)

                ciou_max_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.ssltie_th_max)
                ciou_max10_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.ssltie_th_max10)
                ciou_maxq3_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.ssltie_th_maxq3)
                
                metrics_eval['ciou_max_ogl'].append(ciou_max_neg_val_ogl)
                metrics_eval['ciou_max10_ogl'].append(ciou_max10_neg_val_ogl)
                metrics_eval['ciou_maxq3_ogl'].append(ciou_maxq3_neg_val_ogl)
                
                threshold_ciouadap_ogl = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1])-int(gt_nums)]
                evaluator_ogl.cal_CIOU_adap(pred_av_obj,gt_map,threshold_ciouadap_ogl)
                
                binarized_negatives = binarized_negatives_thresholds(i, args, pred, pred_av_obj, heatmap_silence, heatmap_noise, heatmap_offscreen, pred_obj, threshold_ciou_wogl, threshold_ciou_ogl, threshold_ciouadap_wogl, threshold_ciouadap_ogl)
                metrics_eval = append_metrics_binarized_pred(metrics_eval, binarized_negatives)

    model_name = args.pth_name.split("/")[-1].split(".")[0]
    
    save_max_min_numpy(args, model_name, metrics_eval)

    ciou = evaluator.finalize_cIoU()        
    auc,auc_adap = evaluator.finalize_AUC()   
    iauc_adap = evaluator.finalize_IAUC_adap()
    
    ciou,ciou_adap,iiou,iiou_adap = ciou
    
    metrics_eval['ciou'] = ciou * 100
    metrics_eval['ciou_adap'] = ciou_adap * 100
    metrics_eval['iiou_adap'] = iiou_adap * 100
    metrics_eval['auc'] = auc * 100
    metrics_eval['auc_adap'] = auc_adap * 100
    metrics_eval['iauc_adap'] = iauc_adap * 100
    
    metrics_eval['ciou_ogl'] = evaluator_ogl.finalize_cIoU()        
    metrics_eval['auc_ogl'], metrics_eval['auc_adap_ogl'] = evaluator_ogl.finalize_AUC()
    metrics_eval['ciou_ogl'], metrics_eval['ciou_adap_ogl'],_,_ = metrics_eval['ciou_ogl']
    metrics_eval['ciou_ogl'] = metrics_eval['ciou_ogl']*100
    metrics_eval['ciou_adap_ogl'] = metrics_eval['ciou_adap_ogl']*100
    metrics_eval['auc_ogl'] = metrics_eval['auc_ogl']*100
    metrics_eval['auc_adap_ogl'] = metrics_eval['auc_adap_ogl']*100
    
    metrics_eval = compute_metrics(metrics_eval)

    metrics_out_path = os.path.join("metrics", f"{model_name}_{args.testset}.txt")
    print(f"Metrics stored in {metrics_out_path}!")
    save_metrics_to_file(metrics_out_path, metrics_eval)

@torch.no_grad()
def eval(audio_visual_model, object_saliency_model, args):
    
    random_seeds = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 9999]
        
    audio_visual_model.train(False)
    object_saliency_model.train(False)

    evaluator_av = utils.Evaluator_iiou()
    evaluator_ogl = utils.Evaluator_iiou()

    metrics_eval = define_metrics()
    
    for random_seed in tqdm(random_seeds):
        if 'fnac' in args.pth_name:
            random.seed(random_seed)
            testdataset = get_test_dataset_fnac(args)
            args.image_path = testdataset.image_path
        else:
            random.seed(random_seed)
            testdataset = get_test_dataset(args)
            args.image_path = testdataset.image_path
    
        testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers=8)
        
        for step, (image, spec, spec_noise, spec_silence, spec_offscreen, bboxes, name) in enumerate(tqdm(testdataloader)):
            if args.gpu is not None:
                spec = spec.cuda(args.gpu, non_blocking=True)
                spec_noise = spec_noise.cuda(args.gpu, non_blocking=True)
                spec_silence = spec_silence.cuda(args.gpu, non_blocking=True)
                spec_offscreen = spec_offscreen.cuda(args.gpu, non_blocking=True)
                
                image = image.cuda(args.gpu, non_blocking=True)

            heatmap_av = audio_visual_model(image.float(), spec.float())[1].unsqueeze(1)
            heatmap_noise = audio_visual_model(image.float(),spec_noise.float())[1].unsqueeze(1)
            heatmap_silence = audio_visual_model(image.float(),spec_silence.float())[1].unsqueeze(1)
            heatmap_offscreen = audio_visual_model(image.float(),spec_offscreen.float())[1].unsqueeze(1)

            for htmp, htmp_noise, htmp_silence, htmp_offscreen in zip(heatmap_av, heatmap_noise, heatmap_silence, heatmap_offscreen):

                metrics_eval['max_heatmap_list'].append(torch.max(htmp).item())
                metrics_eval['max_heatmap_silence_list'].append(torch.max(htmp_silence).item())
                metrics_eval['max_heatmap_noise_list'].append(torch.max(htmp_noise).item())
                metrics_eval['max_heatmap_offscreen_list'].append(torch.max(htmp_offscreen).item())

                metrics_eval['min_heatmap_list'].append(torch.min(htmp).item())
                metrics_eval['min_heatmap_silence_list'].append(torch.min(htmp_silence).item())
                metrics_eval['min_heatmap_noise_list'].append(torch.min(htmp_noise).item())
                metrics_eval['min_heatmap_offscreen_list'].append(torch.min(htmp_offscreen).item())

            heatmap_av_ = F.interpolate(heatmap_av, size=(224, 224), mode='bilinear', align_corners=True)
            heatmap_av = heatmap_av_.data.cpu().numpy()

            img_feat = object_saliency_model(image)

            heatmap_obj = F.interpolate(img_feat, size=(224, 224), mode='bilinear', align_corners=True)
            heatmap_obj = heatmap_obj.data.cpu().numpy()

            for i in range(spec.shape[0]):
                pred_av = (heatmap_av[i, 0])
                pred_obj = (heatmap_obj[i, 0])
                pred_av_obj = (pred_av * args.alpha + pred_obj * (1 - args.alpha))

                gt_map = bboxes['gt_map'][i].data.cpu().numpy()

                threshold_ciou_wogl = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1] * 0.5)]
                evaluator_av.cal_CIOU(pred_av, gt_map, threshold_ciou_wogl)
                ciou_05_val, _, _ = evaluator_av.cal_CIOU(pred_av,gt_map,0.5)
                if np.isnan(ciou_05_val):
                    metrics_eval['ciou_05_wogl'].append(0)
                else:
                    metrics_eval['ciou_05_wogl'].append(ciou_05_val)

                if 'ezvsl' in args.pth_name:
                    ciou_max_neg_val_wogl, _, _ = evaluator_av.cal_CIOU(pred_av,gt_map,args.ezvsl_th_max)
                    ciou_max10_neg_val_wogl, _, _ = evaluator_av.cal_CIOU(pred_av,gt_map,args.ezvsl_th_max10)
                    ciou_maxq3_neg_val_wogl, _, _ = evaluator_av.cal_CIOU(pred_av,gt_map,args.ezvsl_th_maxq3)
                elif 'slavc' in args.pth_name:
                    ciou_max_neg_val_wogl, _, _ = evaluator_av.cal_CIOU(pred_av,gt_map,args.slavc_th_max)
                    ciou_max10_neg_val_wogl, _, _ = evaluator_av.cal_CIOU(pred_av,gt_map,args.slavc_th_max10)
                    ciou_maxq3_neg_val_wogl, _, _ = evaluator_av.cal_CIOU(pred_av,gt_map,args.slavc_th_maxq3)
                elif 'fnac' in args.pth_name:
                    ciou_max_neg_val_wogl, _, _ = evaluator_av.cal_CIOU(pred_av,gt_map,args.fnac_th_max)
                    ciou_max10_neg_val_wogl, _, _ = evaluator_av.cal_CIOU(pred_av,gt_map,args.fnac_th_max10)
                    ciou_maxq3_neg_val_wogl, _, _ = evaluator_av.cal_CIOU(pred_av,gt_map,args.fnac_th_maxq3)
                
                if np.isnan(ciou_max_neg_val_wogl):
                    metrics_eval['ciou_max_wogl'].append(0)
                else:
                    metrics_eval['ciou_max_wogl'].append(ciou_max_neg_val_wogl)
                
                if np.isnan(ciou_max10_neg_val_wogl):
                    metrics_eval['ciou_max10_wogl'].append(0)
                else:
                    metrics_eval['ciou_max10_wogl'].append(ciou_max10_neg_val_wogl)
                
                if np.isnan(ciou_maxq3_neg_val_wogl):
                    metrics_eval['ciou_maxq3_wogl'].append(0)
                else:
                    metrics_eval['ciou_maxq3_wogl'].append(ciou_maxq3_neg_val_wogl)
            
                gt_nums = (gt_map!=0).sum()
                if int(gt_nums) == 0:
                    gt_nums = int(pred_av.shape[0] * pred_av.shape[1])//2
                threshold_ciouadap_wogl = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1])-int(gt_nums)] # adap
                evaluator_av.cal_CIOU_adap(pred_av, gt_map, threshold_ciouadap_wogl)

                threshold_ciou_ogl = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] * 0.5)]
                evaluator_ogl.cal_CIOU(pred_av_obj, gt_map, threshold_ciou_ogl)
                ciou_05_ogl_val, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,0.5)
                metrics_eval['ciou_05_ogl'].append(ciou_05_ogl_val)

                if 'ezvsl' in args.pth_name:
                    ciou_max_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.ezvsl_th_max)
                    ciou_max10_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.ezvsl_th_max10)
                    ciou_maxq3_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.ezvsl_th_maxq3)
                elif 'slavc' in args.pth_name:
                    ciou_max_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.slavc_th_max)
                    ciou_max10_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.slavc_th_max10)
                    ciou_maxq3_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.slavc_th_maxq3)
                elif 'fnac' in args.pth_name:
                    ciou_max_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.fnac_th_max)
                    ciou_max10_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.fnac_th_max10)
                    ciou_maxq3_neg_val_ogl, _, _ = evaluator_ogl.cal_CIOU(pred_av_obj,gt_map,args.fnac_th_maxq3)
                
                metrics_eval['ciou_max_ogl'].append(ciou_max_neg_val_ogl)
                metrics_eval['ciou_max10_ogl'].append(ciou_max10_neg_val_ogl)
                metrics_eval['ciou_maxq3_ogl'].append(ciou_maxq3_neg_val_ogl)
                
                threshold_ciouadap_ogl = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1])-int(gt_nums)]
                evaluator_ogl.cal_CIOU_adap(pred_av_obj,gt_map,threshold_ciouadap_ogl)
                
                binarized_negatives = binarized_negatives_thresholds(i, args, pred_av, pred_av_obj, heatmap_silence, heatmap_noise, heatmap_offscreen, pred_obj, threshold_ciou_wogl, threshold_ciou_ogl, threshold_ciouadap_wogl, threshold_ciouadap_ogl)
                metrics_eval = append_metrics_binarized_pred(metrics_eval, binarized_negatives)

                short_name = '_'.join(name[i].split('/')[-1].replace('.jpg','').split('_')[:-1])
                if short_name in evaluator_av.iou.keys():
                    evaluator_av.iou_adap[short_name].append(evaluator_av.ciou_adap[-1])
                    evaluator_av.iou[short_name].append(evaluator_av.ciou[-1])
                else:
                    evaluator_av.iou_adap[short_name] = [evaluator_av.ciou_adap[-1]]
                    evaluator_av.iou[short_name] = [evaluator_av.ciou[-1]]

    model_name = args.pth_name.split("/")[-1].split(".")[0]
    
    save_max_min_numpy(args, model_name, metrics_eval)

    ciou = evaluator_av.finalize_cIoU()        
    auc,auc_adap = evaluator_av.finalize_AUC()
    iauc = evaluator_av.finalize_IAUC()       
    iauc_adap = evaluator_av.finalize_IAUC_adap()
    
    ciou,ciou_adap,iiou,iiou_adap = ciou
    
    metrics_eval['ciou'] = ciou * 100
    metrics_eval['ciou_adap'] = ciou_adap * 100
    metrics_eval['iiou'] = iiou * 100
    metrics_eval['iiou_adap'] = iiou_adap * 100
    metrics_eval['auc'] = auc * 100
    metrics_eval['iauc'] = iauc * 100
    metrics_eval['auc_adap'] = auc_adap * 100
    metrics_eval['iauc_adap'] = iauc_adap * 100
    
    metrics_eval['ciou_ogl'] = evaluator_ogl.finalize_cIoU()        
    metrics_eval['auc_ogl'], metrics_eval['auc_adap_ogl'] = evaluator_ogl.finalize_AUC()
    metrics_eval['ciou_ogl'], metrics_eval['ciou_adap_ogl'],_,_ = metrics_eval['ciou_ogl']
    metrics_eval['ciou_ogl'] = metrics_eval['ciou_ogl']*100
    metrics_eval['ciou_adap_ogl'] = metrics_eval['ciou_adap_ogl']*100
    metrics_eval['auc_ogl'] = metrics_eval['auc_ogl']*100
    metrics_eval['auc_adap_ogl'] = metrics_eval['auc_adap_ogl']*100
    
    metrics_eval = compute_metrics(metrics_eval)

    metrics_out_path = os.path.join("metrics", f"{model_name}_{args.testset}.txt")
    print(f"Metrics stored in {metrics_out_path}!")
    save_metrics_to_file(metrics_out_path, metrics_eval)

class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

if __name__ == "__main__":
    main(get_arguments())
