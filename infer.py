import os
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import utils_dir.utils as utils
import easydict
from sklearn import metrics
import numpy as np
import argparse
from models.model import EZVSL, SLAVC, FNAC
from models.model_lvs import AVENet
from models.model_ssltie import AVENet_ssltie
from datasets.datasets import get_test_dataset, inverse_normalize, get_test_dataset_fnac, get_test_dataset_rcgrad
from datasets.datasets_lvs import GetAudioVideoDataset
from datasets.datasets_ssltie import GetAudioVideoDataset_ssltie
import cv2
from tqdm import tqdm
from utils_dir.opts_ssltie import SSLTIE_args
import matplotlib.pyplot as plt
from PIL import Image

# Fixing random seed for reproducibility
random.seed(42)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='', help='path to save trained model weights')
    parser.add_argument('--pth_name', type=str, default='checkpoints/slavc.pth', help='pth name')
    parser.add_argument('--save_visualizations', action='store_true', help='Set to store all VSL visualizations (saved in viz directory within experiment folder)')

    # Dataset
    parser.add_argument('--testset', default='is3', type=str, help='testset (flickr or vggss)')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of data')
    parser.add_argument('--test_gt_path', default='/media/v/IS3Dataset/IS3_annotations.json', type=str)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')

    # Model
    parser.add_argument('--tau', default=0.03, type=float, help='tau')
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha')
    
    # Metrics
    parser.add_argument('--threshold', default=0.5460, type=float, help='Threshold for pIA metric computation.')

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--multiprocessing_distributed', action='store_true')
    
    return parser.parse_args()

def overlay_with_blue_tint(image, heatmap, alpha=0.88, blue_tint_intensity=0.45):
    
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    if isinstance(heatmap, Image.Image):
        heatmap_np = np.array(heatmap)
    else:
        heatmap_np = heatmap
    
    if np.max(heatmap_np) == 0: 

        blue_tint = np.zeros_like(image_np)
        blue_tint[:, :, 2] = 200 
        tinted_overlay = (1 - blue_tint_intensity) * image_np + blue_tint_intensity * blue_tint
        result_image_np = (1 - alpha) * image_np + alpha * tinted_overlay
    else:
        result_image_np = utils.overlay(image_np, heatmap_np) 

    if not isinstance(result_image_np, np.ndarray):
        result_image_np = np.array(result_image_np)

    result_image_np = result_image_np.astype(np.uint8)
    result_image = Image.fromarray(result_image_np)

    return result_image

def generate_img(original_image, heatmap, heatmap_thresholded, out_path, out_path_thresholded, min_val, max_val):
    
    result_image = overlay_with_blue_tint(original_image, heatmap_thresholded)
    plt.imshow(result_image)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title(f"Min: {min_val:.2f} - Max: {max_val:.2f}", fontsize=25)
    plt.savefig(out_path_thresholded, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    plt.imshow(utils.overlay(original_image, heatmap))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title(f"Min: {min_val:.2f} - Max: {max_val:.2f}", fontsize=25)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def heatmaps_negatives(args, i, heatmap_silence, heatmap_noise, heatmap_offscreen):
    heatmap_negatives = {}
    
    heatmap_arr_noise =  heatmap_noise.data.cpu().numpy()
    heatmap_now_noise = cv2.resize(heatmap_arr_noise[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap_now_arr_noise = heatmap_now_noise.copy()
    heatmap_now_arr_noise_thresholded = heatmap_now_noise.copy()
    heatmap_now_noise_thresholded = np.where(heatmap_now_arr_noise_thresholded > args.threshold, heatmap_now_arr_noise_thresholded, 0)
    heatmap_now_noise_thresholded = utils.normalize_thresholded_img(heatmap_now_noise_thresholded)
    heatmap_now_noise = utils.normalize_img(heatmap_now_noise)
    
    heatmap_arr_silence =  heatmap_silence.data.cpu().numpy()
    heatmap_now_silence = cv2.resize(heatmap_arr_silence[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap_now_arr_silence = heatmap_now_silence.copy()
    heatmap_now_arr_silence_thresholded = heatmap_now_silence.copy()
    heatmap_now_silence_thresholded = np.where(heatmap_now_arr_silence_thresholded > args.threshold, heatmap_now_arr_silence_thresholded, 0)
    heatmap_now_silence_thresholded = utils.normalize_thresholded_img(heatmap_now_silence_thresholded)
    heatmap_now_silence = utils.normalize_img(heatmap_now_silence)
    
    heatmap_arr_offscreen =  heatmap_offscreen.data.cpu().numpy()
    heatmap_now_offscreen = cv2.resize(heatmap_arr_offscreen[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap_now_arr_offscreen = heatmap_now_offscreen.copy()
    heatmap_now_arr_offscreen_thresholded = heatmap_now_offscreen.copy()
    heatmap_now_offscreen_thresholded = np.where(heatmap_now_arr_offscreen_thresholded > args.threshold, heatmap_now_arr_offscreen_thresholded, 0)
    heatmap_now_offscreen_thresholded = utils.normalize_thresholded_img(heatmap_now_offscreen_thresholded)
    heatmap_now_offscreen = utils.normalize_img(heatmap_now_offscreen)
    
    heatmap_negatives['heatmap_now_silence'] = heatmap_now_silence
    heatmap_negatives['heatmap_now_silence_thresholded'] = heatmap_now_silence_thresholded
    heatmap_negatives['heatmap_now_arr_silence'] = heatmap_now_arr_silence
    
    heatmap_negatives['heatmap_now_noise'] = heatmap_now_noise
    heatmap_negatives['heatmap_now_noise_thresholded'] = heatmap_now_noise_thresholded
    heatmap_negatives['heatmap_now_arr_noise'] = heatmap_now_arr_noise
    
    heatmap_negatives['heatmap_now_offscreen'] = heatmap_now_offscreen
    heatmap_negatives['heatmap_now_offscreen_thresholded'] = heatmap_now_offscreen_thresholded
    heatmap_negatives['heatmap_now_arr_offscreen'] = heatmap_now_arr_offscreen
    
    return heatmap_negatives

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Models
    if 'ssltie' in args.pth_name:
        ssltie_args=SSLTIE_args()
        ssltie_args=ssltie_args.ssltie_args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        audio_visual_model = AVENet_ssltie(ssltie_args)
        audio_visual_model = audio_visual_model.cuda()
        audio_visual_model = nn.DataParallel(audio_visual_model)
        ssltie_args.test_gt_path = args.test_gt_path
        ssltie_args.testset = args.testset
        ssltie_args.dataset_mode = args.testset
        ssltie_args.pth_name = args.pth_name
        ssltie_args.threshold = args.threshold
        
    elif 'lvs' in args.pth_name or 'sslalign' in args.pth_name:
        lvs_args = easydict.EasyDict({
        "data_path" : '',
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
        'threshold': args.threshold
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
    
    elif 'htf' in args.pth_name:
        print('HTF evaluation')
        ckp_fn = 'htf no load'
    else:
        ckp_fn = os.path.join(args.pth_name)
        if os.path.exists(ckp_fn):
            ckp = torch.load(ckp_fn, map_location='cpu')
            audio_visual_model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        else:
            print(f"Checkpoint not found: {ckp_fn}")
    print(ckp_fn,' loaded')

    args.files_list = ["male_ukulele_9253_male"]
    
    # Dataloader
    if 'ssltie' in args.pth_name:
        testdataset = GetAudioVideoDataset_ssltie(ssltie_args, mode='test', file_list = args.files_list)
        ssltie_args.image_path = testdataset.image_path
    if 'margin' in args.pth_name or 'ezvsl' in args.pth_name or 'slavc' in args.pth_name:
        testdataset = get_test_dataset(args)
        args.image_path = testdataset.image_path
    if 'rcgrad' in args.pth_name:
        testdataset = get_test_dataset_rcgrad(args)
        args.image_path = testdataset.image_path
    if 'fnac' in args.pth_name:
        testdataset = get_test_dataset_fnac(args)
        args.image_path = testdataset.image_path
    if 'lvs' in args.pth_name or 'sslalign' in args.pth_name:
        testdataset = GetAudioVideoDataset(lvs_args,  mode='test', file_list = args.files_list)
        lvs_args.image_path = testdataset.image_path
    if 'htf' in args.pth_name:
        testdataset = get_test_dataset(args)
        args.image_path = testdataset.image_path
    
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)
    
    print("Loaded dataloader.")
    
    if 'lvs' in args.pth_name or 'sslalign' in args.pth_name:
        infer_lvs(testdataloader, audio_visual_model, object_saliency_model, lvs_args)
    elif 'ssltie' in args.pth_name:
        infer_ssltie(testdataloader, audio_visual_model, object_saliency_model, ssltie_args)
    else:
        infer(testdataloader, audio_visual_model, object_saliency_model, args)
    
@torch.no_grad()
def infer_lvs(testdataloader, model, object_saliency_model, args):
    model.train(False)
    object_saliency_model.train(False)
    
    for step, batch in enumerate(tqdm(testdataloader)):
        image, spec, spec_noise, spec_silence, spec_offscreen, bboxes, name = batch
        
        spec = Variable(spec).cuda()
        spec_silence = Variable(spec_silence).cuda()
        spec_noise = Variable(spec_noise).cuda()
        
        image = Variable(image).cuda()
        
        heatmap,_,Pos,Neg = model(image.float(),spec.float(),args)
        heatmap_silence,_,Pos,Neg = model(image.float(),spec_silence.float(),args)
        heatmap_noise,_,Pos,Neg = model(image.float(),spec_noise.float(),args)
        heatmap_offscreen,_,Pos,Neg = model(image.float(),spec_offscreen.float(),args)
        
        heatmap = nn.functional.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=True)
        
        for i in range(spec.shape[0]):
            heatmap_now = heatmap[i:i+1].cpu().numpy()
            heatmap_now_thresholded = heatmap[i:i+1].cpu().numpy().copy()[0][0]

            heatmap_arr = heatmap_now.copy()
            heatmap_now = utils.normalize_img(heatmap_now[0][0])
        
            heatmap_now_thresholded = np.where(heatmap_now_thresholded > args.threshold, heatmap_now_thresholded, 0)
            heatmap_now_thresholded = utils.normalize_thresholded_img(heatmap_now_thresholded)

            heatmap_negatives = heatmaps_negatives(args, i, heatmap_silence, heatmap_noise, heatmap_offscreen)
            
            out_dir = os.path.join("inferences", args.testset, args.pth_name.split("/")[-1].split(".")[0])
            os.makedirs(out_dir, exist_ok=True)
            fname_root = name[i].split(".")[0]
            out_heatmap_name = os.path.join(out_dir, f"{fname_root}.jpg")
            out_silence_name = os.path.join(out_dir, f"{fname_root}_silence.jpg")
            out_noise_name = os.path.join(out_dir, f"{fname_root}_noise.jpg")
            out_offscreen_name = os.path.join(out_dir, f"{fname_root}_offscreen.jpg")

            out_heatmap_name_thresholded = os.path.join(out_dir, f"{fname_root}_thresholded.jpg")
            out_silence_name_thresholded = os.path.join(out_dir, f"{fname_root}_silence_thresholded.jpg")
            out_noise_name_thresholded = os.path.join(out_dir, f"{fname_root}_noise_thresholded.jpg")
            out_offscreen_name_thresholded = os.path.join(out_dir, f"{fname_root}_offscreen_thresholded.jpg")

            frame_ori = Image.open(os.path.join(args.image_path,f"{name[i]}"))
            frame_ori = frame_ori.resize((224,224))

            generate_img(frame_ori, heatmap_now, heatmap_now_thresholded, out_heatmap_name, out_heatmap_name_thresholded, np.min(heatmap_arr[0, 0]), np.max(heatmap_arr[0, 0]))
            generate_img(frame_ori, heatmap_negatives['heatmap_now_silence'], heatmap_negatives['heatmap_now_silence_thresholded'], out_silence_name, out_silence_name_thresholded, np.min(heatmap_negatives['heatmap_now_arr_silence']), np.max(heatmap_negatives['heatmap_now_arr_silence']))
            generate_img(frame_ori, heatmap_negatives['heatmap_now_noise'], heatmap_negatives['heatmap_now_noise_thresholded'], out_noise_name, out_noise_name_thresholded, np.min(heatmap_negatives['heatmap_now_arr_noise']), np.max(heatmap_negatives['heatmap_now_arr_noise']))
            generate_img(frame_ori, heatmap_negatives['heatmap_now_offscreen'], heatmap_negatives['heatmap_now_offscreen_thresholded'], out_offscreen_name, out_offscreen_name_thresholded, np.min(heatmap_negatives['heatmap_now_arr_offscreen']), np.max(heatmap_negatives['heatmap_now_arr_offscreen']))

@torch.no_grad()
def infer_ssltie(testdataloader, model, object_saliency_model, args):
    model.train(False)
    object_saliency_model.train(False)
    
    for step, (image, spec, spec_noise, spec_silence, spec_offscreen, bboxes, name) in enumerate(tqdm(testdataloader)):
        spec = Variable(spec).cuda()
        image = Variable(image).cuda()
        
        heatmap, out, Pos, Neg, out_ref = model(image.float(), spec.float(), args, mode='val')
        heatmap_silence, out, Pos, Neg, out_ref = model(image.float(), spec_silence.float(), args, mode='val')
        heatmap_noise, out, Pos, Neg, out_ref = model(image.float(), spec_noise.float(), args, mode='val')
        heatmap_offscreen, out, Pos, Neg, out_ref = model(image.float(), spec_offscreen.float(), args, mode='val')
        
        heatmap_arr =  heatmap.data.cpu().numpy()
        
        for i in range(spec.shape[0]):
            heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            heatmap_now_thresholded = heatmap_now.copy()
            heatmap_now = utils.normalize_img(heatmap_now)
            
            heatmap_now_thresholded = np.where(heatmap_now_thresholded > args.threshold, heatmap_now_thresholded, 0)
            heatmap_now_thresholded = utils.normalize_thresholded_img(heatmap_now_thresholded)
            
            heatmap_negatives = heatmaps_negatives(args, i, heatmap_silence, heatmap_noise, heatmap_offscreen)
            
            out_dir = os.path.join("inferences", args.testset, args.pth_name.split("/")[-1].split(".")[0])
            os.makedirs(out_dir, exist_ok=True)
            fname_root = name[i].split(".")[0]
            out_heatmap_name = os.path.join(out_dir, f"{fname_root}.jpg")
            out_silence_name = os.path.join(out_dir, f"{fname_root}_silence.jpg")
            out_noise_name = os.path.join(out_dir, f"{fname_root}_noise.jpg")
            out_offscreen_name = os.path.join(out_dir, f"{fname_root}_offscreen.jpg")

            out_heatmap_name_thresholded = os.path.join(out_dir, f"{fname_root}_thresholded.jpg")
            out_silence_name_thresholded = os.path.join(out_dir, f"{fname_root}_silence_thresholded.jpg")
            out_noise_name_thresholded = os.path.join(out_dir, f"{fname_root}_noise_thresholded.jpg")
            out_offscreen_name_thresholded = os.path.join(out_dir, f"{fname_root}_offscreen_thresholded.jpg")

            frame_ori = Image.open(os.path.join(args.image_path,f"{name[i]}"))
            frame_ori = frame_ori.resize((224,224))

            generate_img(frame_ori, heatmap_now, heatmap_now_thresholded, out_heatmap_name, out_heatmap_name_thresholded, np.min(heatmap_arr[0, 0]), np.max(heatmap_arr[0, 0]))
            generate_img(frame_ori, heatmap_negatives['heatmap_now_silence'], heatmap_negatives['heatmap_now_silence_thresholded'], out_silence_name, out_silence_name_thresholded, np.min(heatmap_negatives['heatmap_now_arr_silence']), np.max(heatmap_negatives['heatmap_now_arr_silence']))
            generate_img(frame_ori, heatmap_negatives['heatmap_now_noise'], heatmap_negatives['heatmap_now_noise_thresholded'], out_noise_name, out_noise_name_thresholded, np.min(heatmap_negatives['heatmap_now_arr_noise']), np.max(heatmap_negatives['heatmap_now_arr_noise']))
            generate_img(frame_ori, heatmap_negatives['heatmap_now_offscreen'], heatmap_negatives['heatmap_now_offscreen_thresholded'], out_offscreen_name, out_offscreen_name_thresholded, np.min(heatmap_negatives['heatmap_now_arr_offscreen']), np.max(heatmap_negatives['heatmap_now_arr_offscreen']))

@torch.no_grad()
def infer(testdataloader, audio_visual_model, object_saliency_model, args):
    print(args.threshold)
    audio_visual_model.train(False)
    object_saliency_model.train(False)
    
    for step, (image, spec, spec_noise, spec_silence, spec_offscreen, bboxes, name) in enumerate(tqdm(testdataloader)):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            spec_silence = spec_silence.cuda(args.gpu, non_blocking=True)
            spec_noise = spec_noise.cuda(args.gpu, non_blocking=True)
            spec_offscreen = spec_offscreen.cuda(args.gpu, non_blocking=True)
            
            image = image.cuda(args.gpu, non_blocking=True)

        heatmap_av = audio_visual_model(image.float(), spec.float())[1].unsqueeze(1)
        heatmap_silence = audio_visual_model(image.float(),spec_silence.float())[1].unsqueeze(1)
        heatmap_noise = audio_visual_model(image.float(),spec_noise.float())[1].unsqueeze(1)
        heatmap_offscreen = audio_visual_model(image.float(),spec_offscreen.float())[1].unsqueeze(1)
                
        heatmap_av_ = F.interpolate(heatmap_av, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_av = heatmap_av_.data.cpu().numpy()
        heatmap_av_thresholded = heatmap_av_.data.cpu().numpy()

        for i in range(spec.shape[0]):
            heatmap_av_thresholded_now = heatmap_av_thresholded[i, 0]
            heatmap_now_thresholded = np.where(heatmap_av_thresholded_now > args.threshold, heatmap_av_thresholded_now, 0)
            heatmap_now_thresholded = utils.normalize_thresholded_img(heatmap_now_thresholded)

            pred_av = utils.normalize_img(heatmap_av[i, 0])
            
            heatmap_negatives = heatmaps_negatives(args, i, heatmap_silence, heatmap_noise, heatmap_offscreen)
        
            out_dir = os.path.join("inferences", args.testset, args.pth_name.split("/")[-1].split(".")[0])
            os.makedirs(out_dir, exist_ok=True)
            fname_root = name[i].split(".")[0]
            out_heatmap_name = os.path.join(out_dir, f"{fname_root}.jpg")
            out_silence_name = os.path.join(out_dir, f"{fname_root}_silence.jpg")
            out_noise_name = os.path.join(out_dir, f"{fname_root}_noise.jpg")
            out_offscreen_name = os.path.join(out_dir, f"{fname_root}_offscreen.jpg")

            out_heatmap_name_thresholded = os.path.join(out_dir, f"{fname_root}_thresholded.jpg")
            out_silence_name_thresholded = os.path.join(out_dir, f"{fname_root}_silence_thresholded.jpg")
            out_noise_name_thresholded = os.path.join(out_dir, f"{fname_root}_noise_thresholded.jpg")
            out_offscreen_name_thresholded = os.path.join(out_dir, f"{fname_root}_offscreen_thresholded.jpg")

            frame_ori = Image.open(os.path.join(args.image_path,f"{name[i]}.jpg"))
            frame_ori = frame_ori.resize((224,224))

            generate_img(frame_ori, pred_av, heatmap_now_thresholded, out_heatmap_name, out_heatmap_name_thresholded, np.min(heatmap_av[i, 0]), np.max(heatmap_av[i, 0]))
            generate_img(frame_ori, heatmap_negatives['heatmap_now_silence'], heatmap_negatives['heatmap_now_silence_thresholded'], out_silence_name, out_silence_name_thresholded, np.min(heatmap_negatives['heatmap_now_arr_silence']), np.max(heatmap_negatives['heatmap_now_arr_silence']))
            generate_img(frame_ori, heatmap_negatives['heatmap_now_noise'], heatmap_negatives['heatmap_now_noise_thresholded'], out_noise_name, out_noise_name_thresholded, np.min(heatmap_negatives['heatmap_now_arr_noise']), np.max(heatmap_negatives['heatmap_now_arr_noise']))
            generate_img(frame_ori, heatmap_negatives['heatmap_now_offscreen'], heatmap_negatives['heatmap_now_offscreen_thresholded'], out_offscreen_name, out_offscreen_name_thresholded, np.min(heatmap_negatives['heatmap_now_arr_offscreen']), np.max(heatmap_negatives['heatmap_now_arr_offscreen']))

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
