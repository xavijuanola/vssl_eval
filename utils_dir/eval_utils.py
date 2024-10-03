from collections import defaultdict

import torch
from torchvision import transforms
import clip
import cv2
import numpy as np
import wav2clip
from PIL import Image
from sklearn import metrics
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    if isinstance(image, torch.Tensor):
        if image.ndimension() == 4:  # Handling a batch of images
            images = []
            for img in image:
                print(img.shape)
                img_pil = transforms.ToPILImage()(img)  # Convert each 3D image tensor to a PIL Image
                images.append(img_pil.convert("RGB"))
            return images  # Return the list of converted images
        elif image.ndimension() == 3:  # Single image tensor
            image = transforms.ToPILImage()(image)
        else:
            raise ValueError(f"Unsupported tensor dimensionality: {image.ndimension()}")
    return image.convert("RGB")



def _transform(n_px):
    return Compose(
        [
            Resize((n_px, n_px), interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


preprocess = _transform(224)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def combine_heatmap_img(img, pred):
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    img = preprocess(img).permute(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    vis = show_cam_on_image(img, pred)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def cal_CIOU(infer, gtmap, thres=0.5):
    infer_map = np.zeros((224, 224))
    infer_map[infer >= thres] = 1
    ciou = np.sum(infer_map * gtmap) / (
        np.sum(gtmap) + np.sum(infer_map * (gtmap == 0))
    )
    return (
        ciou,
        np.sum(infer_map * gtmap),
        (np.sum(gtmap) + np.sum(infer_map * (gtmap == 0))),
    )


def clean_pred(pred):
    pred = normalize_img(pred)
    threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
    pred[pred < threshold] = 0
    return pred


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


def process_heatmap(heatmap):
    heatmap_arr = heatmap.data.cpu().numpy()
    heatmap_now = cv2.resize(
        heatmap_arr[0, 0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR
    )
    heatmap_now = normalize_img(-heatmap_now)
    pred = 1 - heatmap_now
    return pred


def compute_metrics(preds, thres=0.5, at=0.5):
    metrics = {}
    ious = [cal_CIOU(pred, gt_map, thres=thres)[0] for _, pred, gt_map in preds]

    results = []
    for i in range(21):
        result = np.sum(np.array(ious) >= 0.05 * i)
        result = result / len(ious)
        results.append(result)
    x = [0.05 * i for i in range(21)]

    metrics["auc"] = metrics.auc(x, results)
    metrics["cIoU"] = np.sum(np.array(ious) >= at) / len(ious)
    return metrics


def extract_audio_embeddings(audio, model, device="cpu"):
    return wav2clip.embed_audio(audio, model)


def extract_text_embeddings(x, model, device="cpu"):
    text = clip.tokenize(x).to(device)
    text_features = model.encode_text(text)
    return text_features

def compute_AUC_pIA(metric):
        cious = [np.sum(np.array(metric) <= 0.05*i) / len(metric)
                 for i in range(21)]
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

def compute_AUC(metric):
        cious = [np.sum(np.array(metric) >= 0.05*i) / len(metric)
                 for i in range(21)]
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

def compute_IOU_maps(binarized_map1, binarized_map2):
  if binarized_map2.sum() == 0 and binarized_map1.sum() > 0:
    iou = 0.0

  elif binarized_map1.sum() == 0 and binarized_map2.sum() == 0:
    iou = 1.0

  else: #binarized_map1.sum() > 0 and binarized_map2.sum() > 0:
    intersection = np.logical_and(binarized_map1, binarized_map2).sum()
    union = np.logical_or(binarized_map1, binarized_map2).sum()

    iou = intersection / union
  return iou

def compute_f(metric1, metric2):
    return (2*(metric1 * metric2) / (metric1 + metric2))

def compute_pIA(binarized_mask):
    highlighted_area = np.sum(binarized_mask)
    total_area = binarized_mask.size

    return (highlighted_area / total_area)

def define_metrics():
    
    metrics = {}
    
    metrics['max_heatmap_list'] = []
    metrics['max_heatmap_silence_list'] = []
    metrics['max_heatmap_noise_list'] = []
    metrics['max_heatmap_offscreen_list'] = []

    metrics['min_heatmap_list'] = []
    metrics['min_heatmap_silence_list'] = []
    metrics['min_heatmap_noise_list'] = []
    metrics['min_heatmap_offscreen_list'] = []

    metrics['ciou_05_wogl'] = []
    metrics['ciou_max_wogl'] = []
    metrics['ciou_max10_wogl'] = []
    metrics['ciou_maxq3_wogl'] = []

    metrics['ciou_05_ogl'] = []
    metrics['ciou_max_ogl'] = []
    metrics['ciou_max10_ogl'] = []
    metrics['ciou_maxq3_ogl'] = []

    metrics['iou_pos_silence_wogl'] = []
    metrics['iou_pos_silence_adap_wogl'] = []
    metrics['iou_pos_noise_wogl'] = []
    metrics['iou_pos_noise_adap_wogl'] = []
    metrics['iou_pos_offscreen_wogl'] = []
    metrics['iou_pos_offscreen_adap_wogl'] = []
    metrics['iou_silence_noise_wogl'] = []
    metrics['iou_silence_noise_adap_wogl'] = []
    metrics['iou_silence_offscreen_wogl'] = []
    metrics['iou_silence_offscreen_adap_wogl'] = []
    metrics['iou_noise_offscreen_wogl'] = []
    metrics['iou_noise_offscreen_adap_wogl'] = []

    metrics['iou_pos_silence_max_wogl'] = []
    metrics['iou_pos_noise_max_wogl'] = []
    metrics['iou_pos_offscreen_max_wogl'] = []
    metrics['iou_silence_noise_max_wogl'] = []
    metrics['iou_silence_offscreen_max_wogl'] = []
    metrics['iou_noise_offscreen_max_wogl'] = []

    metrics['iou_pos_silence_max10_wogl'] = []
    metrics['iou_pos_noise_max10_wogl'] = []
    metrics['iou_pos_offscreen_max10_wogl'] = []
    metrics['iou_silence_noise_max10_wogl'] = []
    metrics['iou_silence_offscreen_max10_wogl'] = []
    metrics['iou_noise_offscreen_max10_wogl'] = []

    metrics['iou_pos_silence_maxq3_wogl'] = []
    metrics['iou_pos_noise_maxq3_wogl'] = []
    metrics['iou_pos_offscreen_maxq3_wogl'] = []
    metrics['iou_silence_noise_maxq3_wogl'] = []
    metrics['iou_silence_offscreen_maxq3_wogl'] = []
    metrics['iou_noise_offscreen_maxq3_wogl'] = []

    metrics['iou_pos_silence_ogl'] = []
    metrics['iou_pos_silence_adap_ogl'] = []
    metrics['iou_pos_noise_ogl'] = []
    metrics['iou_pos_noise_adap_ogl'] = []
    metrics['iou_pos_offscreen_ogl'] = []
    metrics['iou_pos_offscreen_adap_ogl'] = []
    metrics['iou_silence_noise_ogl'] = []
    metrics['iou_silence_noise_adap_ogl'] = []
    metrics['iou_silence_offscreen_ogl'] = []
    metrics['iou_silence_offscreen_adap_ogl'] = []
    metrics['iou_noise_offscreen_ogl'] = []
    metrics['iou_noise_offscreen_adap_ogl'] = []

    metrics['pia_batch_noise_wogl'] = []
    metrics['pia_batch_noise_ciou_wogl'] = []
    metrics['pia_batch_noise_ciouadap_wogl'] = []
    metrics['pia_batch_noise_max_wogl'] = []
    metrics['pia_batch_noise_max10_wogl'] = []
    metrics['pia_batch_noise_maxq3_wogl'] = []
    
    metrics['pia_batch_noise_ogl'] = []
    metrics['pia_batch_noise_ciou_ogl'] = []
    metrics['pia_batch_noise_ciouadap_ogl'] = []
    metrics['pia_batch_noise_max_ogl'] = []
    metrics['pia_batch_noise_max10_ogl'] = []
    metrics['pia_batch_noise_maxq3_ogl'] = []
    
    metrics['pia_batch_silence_wogl'] = []
    metrics['pia_batch_silence_ciou_wogl'] = []
    metrics['pia_batch_silence_ciouadap_wogl'] = []
    metrics['pia_batch_silence_max_wogl'] = []
    metrics['pia_batch_silence_max10_wogl'] = []
    metrics['pia_batch_silence_maxq3_wogl'] = []
    
    metrics['pia_batch_silence_ogl'] = []
    metrics['pia_batch_silence_ciou_ogl'] = []
    metrics['pia_batch_silence_ciouadap_ogl'] = []
    metrics['pia_batch_silence_max_ogl'] = []
    metrics['pia_batch_silence_max10_ogl'] = []
    metrics['pia_batch_silence_maxq3_ogl'] = []
    
    metrics['pia_batch_offscreen_wogl'] = []
    metrics['pia_batch_offscreen_ciou_wogl'] = []
    metrics['pia_batch_offscreen_ciouadap_wogl'] = []
    metrics['pia_batch_offscreen_max_wogl'] = []
    metrics['pia_batch_offscreen_max10_wogl'] = []
    metrics['pia_batch_offscreen_maxq3_wogl'] = []
    
    metrics['pia_batch_offscreen_ogl'] = []
    metrics['pia_batch_offscreen_ciou_ogl'] = []
    metrics['pia_batch_offscreen_ciouadap_ogl'] = []
    metrics['pia_batch_offscreen_max_ogl'] = []
    metrics['pia_batch_offscreen_max10_ogl'] = []
    metrics['pia_batch_offscreen_maxq3_ogl'] = []
    
    return metrics

def binarized_negatives_thresholds(i, args, pred, pred_av_obj, heatmap_silence, heatmap_noise, heatmap_offscreen, pred_obj, threshold_ciou_wogl, threshold_ciou_ogl, threshold_ciouadap_wogl, threshold_ciouadap_ogl):
    binarized_negatives = {}
    
    heatmap_arr_silence =  heatmap_silence.data.cpu().numpy()
    heatmap_now_silence = cv2.resize(heatmap_arr_silence[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    pred_silence = heatmap_now_silence
    binarized_negatives['binarized_pred_silence_wogl'] = (pred_silence > args.threshold).astype(int)
    binarized_negatives['binarized_pred_silence_thrciou_wogl'] = (pred_silence > threshold_ciou_wogl).astype(int)
    binarized_negatives['binarized_pred_silence_thrciouadap_wogl'] = (pred_silence > threshold_ciouadap_wogl).astype(int)
    
    pred_av_obj_silence = (pred_silence * 0.4 + pred_obj * (1 - 0.4))
    binarized_negatives['binarized_pred_silence_ogl'] = (pred_av_obj_silence > args.threshold).astype(int)
    binarized_negatives['binarized_pred_silence_thrciou_ogl'] = (pred_av_obj_silence > threshold_ciou_ogl).astype(int)
    binarized_negatives['binarized_pred_silence_thrciouadap_ogl'] = (pred_av_obj_silence > threshold_ciouadap_ogl).astype(int)
    
    heatmap_arr_noise =  heatmap_noise.data.cpu().numpy()
    heatmap_now_noise = cv2.resize(heatmap_arr_noise[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    pred_noise = heatmap_now_noise
    binarized_negatives['binarized_pred_noise_wogl'] = (pred_noise > args.threshold).astype(int)
    binarized_negatives['binarized_pred_noise_thrciou_wogl'] = (pred_noise > threshold_ciou_wogl).astype(int)
    binarized_negatives['binarized_pred_noise_thrciouadap_wogl'] = (pred_noise > threshold_ciouadap_wogl).astype(int)
    
    pred_av_obj_noise = (pred_noise * 0.4 + pred_obj * (1 - 0.4))
    binarized_negatives['binarized_pred_noise_ogl'] = (pred_av_obj_noise > args.threshold).astype(int)
    binarized_negatives['binarized_pred_noise_thrciou_ogl'] = (pred_av_obj_noise > threshold_ciou_ogl).astype(int)
    binarized_negatives['binarized_pred_noise_thrciouadap_ogl'] = (pred_av_obj_noise > threshold_ciouadap_ogl).astype(int)
    
    heatmap_arr_offscreen =  heatmap_offscreen.data.cpu().numpy()
    heatmap_now_offscreen = cv2.resize(heatmap_arr_offscreen[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    pred_offscreen = heatmap_now_offscreen
    binarized_negatives['binarized_pred_offscreen_wogl'] = (pred_offscreen > args.threshold).astype(int)
    binarized_negatives['binarized_pred_offscreen_thrciou_wogl'] = (pred_offscreen > threshold_ciou_wogl).astype(int)
    binarized_negatives['binarized_pred_offscreen_thrciouadap_wogl'] = (pred_offscreen > threshold_ciouadap_wogl).astype(int)
    
    pred_av_obj_offscreen = (pred_offscreen * 0.4 + pred_obj * (1 - 0.4))
    binarized_negatives['binarized_pred_offscreen_ogl'] = (pred_av_obj_offscreen > args.threshold).astype(int)
    binarized_negatives['binarized_pred_offscreen_thrciou_ogl'] = (pred_av_obj_offscreen > threshold_ciou_ogl).astype(int)
    binarized_negatives['binarized_pred_offscreen_thrciouadap_ogl'] = (pred_av_obj_offscreen > threshold_ciouadap_ogl).astype(int)
    
    if 'lvs' in args.pth_name:
        binarized_negatives['binarized_pred_max_wogl'] = (pred > args.lvs_th_max).astype(int)
        binarized_negatives['binarized_pred_max10_wogl'] = (pred > args.lvs_th_max10).astype(int)
        binarized_negatives['binarized_pred_maxq3_wogl'] = (pred > args.lvs_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_noise_max_wogl'] = (pred_noise > args.lvs_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_wogl'] = (pred_noise > args.lvs_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_wogl'] = (pred_noise > args.lvs_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_noise_max_ogl'] = (pred_av_obj_noise > args.lvs_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_ogl'] = (pred_av_obj_noise > args.lvs_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_ogl'] = (pred_av_obj_noise > args.lvs_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_silence_max_wogl'] = (pred_silence > args.lvs_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_wogl'] = (pred_silence > args.lvs_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_wogl'] = (pred_silence > args.lvs_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_silence_max_ogl'] = (pred_av_obj_silence > args.lvs_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_ogl'] = (pred_av_obj_silence > args.lvs_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_ogl'] = (pred_av_obj_silence > args.lvs_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_offscreen_max_wogl'] = (pred_offscreen > args.lvs_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_wogl'] = (pred_offscreen > args.lvs_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_wogl'] = (pred_offscreen > args.lvs_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_offscreen_max_ogl'] = (pred_av_obj_offscreen > args.lvs_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_ogl'] = (pred_av_obj_offscreen > args.lvs_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_ogl'] = (pred_av_obj_offscreen > args.lvs_th_maxq3).astype(int)

    elif 'sslalign' in args.pth_name:
        binarized_negatives['binarized_pred_max_wogl'] = (pred > args.sslalign_th_max).astype(int)
        binarized_negatives['binarized_pred_max10_wogl'] = (pred > args.sslalign_th_max10).astype(int)
        binarized_negatives['binarized_pred_maxq3_wogl'] = (pred > args.sslalign_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_noise_max_wogl'] = (pred_noise > args.sslalign_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_wogl'] = (pred_noise > args.sslalign_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_wogl'] = (pred_noise > args.sslalign_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_noise_max_ogl'] = (pred_av_obj_noise > args.sslalign_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_ogl'] = (pred_av_obj_noise > args.sslalign_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_ogl'] = (pred_av_obj_noise > args.sslalign_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_silence_max_wogl'] = (pred_silence > args.sslalign_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_wogl'] = (pred_silence > args.sslalign_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_wogl'] = (pred_silence > args.sslalign_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_silence_max_ogl'] = (pred_av_obj_silence > args.sslalign_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_ogl'] = (pred_av_obj_silence > args.sslalign_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_ogl'] = (pred_av_obj_silence > args.sslalign_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_offscreen_max_wogl'] = (pred_offscreen > args.sslalign_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_wogl'] = (pred_offscreen > args.sslalign_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_wogl'] = (pred_offscreen > args.sslalign_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_offscreen_max_ogl'] = (pred_av_obj_offscreen > args.sslalign_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_ogl'] = (pred_av_obj_offscreen > args.sslalign_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_ogl'] = (pred_av_obj_offscreen > args.sslalign_th_maxq3).astype(int)
    
    elif 'ssltie' in args.pth_name:
        binarized_negatives['binarized_pred_max_wogl'] = (pred > args.ssltie_th_max).astype(int)
        binarized_negatives['binarized_pred_max10_wogl'] = (pred > args.ssltie_th_max10).astype(int)
        binarized_negatives['binarized_pred_maxq3_wogl'] = (pred > args.ssltie_th_maxq3).astype(int)
                
        binarized_negatives['binarized_pred_noise_max_wogl'] = (pred_noise > args.ssltie_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_wogl'] = (pred_noise > args.ssltie_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_wogl'] = (pred_noise > args.ssltie_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_noise_max_ogl'] = (pred_av_obj_noise > args.ssltie_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_ogl'] = (pred_av_obj_noise > args.ssltie_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_ogl'] = (pred_av_obj_noise > args.ssltie_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_silence_max_wogl'] = (pred_silence > args.ssltie_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_wogl'] = (pred_silence > args.ssltie_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_wogl'] = (pred_silence > args.ssltie_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_silence_max_ogl'] = (pred_av_obj_silence > args.ssltie_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_ogl'] = (pred_av_obj_silence > args.ssltie_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_ogl'] = (pred_av_obj_silence > args.ssltie_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_offscreen_max_wogl'] = (pred_offscreen > args.ssltie_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_wogl'] = (pred_offscreen > args.ssltie_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_wogl'] = (pred_offscreen > args.ssltie_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_offscreen_max_ogl'] = (pred_av_obj_offscreen > args.ssltie_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_ogl'] = (pred_av_obj_offscreen > args.ssltie_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_ogl'] = (pred_av_obj_offscreen > args.ssltie_th_maxq3).astype(int)
    
    elif 'ezvsl' in args.pth_name:
        binarized_negatives['binarized_pred_max_wogl'] = (pred > args.ezvsl_th_max).astype(int)
        binarized_negatives['binarized_pred_max10_wogl'] = (pred > args.ezvsl_th_max10).astype(int)
        binarized_negatives['binarized_pred_maxq3_wogl'] = (pred > args.ezvsl_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_noise_max_wogl'] = (pred_noise > args.ezvsl_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_wogl'] = (pred_noise > args.ezvsl_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_wogl'] = (pred_noise > args.ezvsl_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_noise_max_ogl'] = (pred_av_obj_noise > args.ezvsl_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_ogl'] = (pred_av_obj_noise > args.ezvsl_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_ogl'] = (pred_av_obj_noise > args.ezvsl_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_silence_max_wogl'] = (pred_silence > args.ezvsl_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_wogl'] = (pred_silence > args.ezvsl_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_wogl'] = (pred_silence > args.ezvsl_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_silence_max_ogl'] = (pred_av_obj_silence > args.ezvsl_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_ogl'] = (pred_av_obj_silence > args.ezvsl_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_ogl'] = (pred_av_obj_silence > args.ezvsl_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_offscreen_max_wogl'] = (pred_offscreen > args.ezvsl_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_wogl'] = (pred_offscreen > args.ezvsl_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_wogl'] = (pred_offscreen > args.ezvsl_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_offscreen_max_ogl'] = (pred_av_obj_offscreen > args.ezvsl_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_ogl'] = (pred_av_obj_offscreen > args.ezvsl_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_ogl'] = (pred_av_obj_offscreen > args.ezvsl_th_maxq3).astype(int)

    elif 'slavc' in args.pth_name:
        binarized_negatives['binarized_pred_max_wogl'] = (pred > args.slavc_th_max).astype(int)
        binarized_negatives['binarized_pred_max10_wogl'] = (pred > args.slavc_th_max10).astype(int)
        binarized_negatives['binarized_pred_maxq3_wogl'] = (pred > args.slavc_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_noise_max_wogl'] = (pred_noise > args.slavc_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_wogl'] = (pred_noise > args.slavc_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_wogl'] = (pred_noise > args.slavc_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_noise_max_ogl'] = (pred_av_obj_noise > args.slavc_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_ogl'] = (pred_av_obj_noise > args.slavc_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_ogl'] = (pred_av_obj_noise > args.slavc_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_silence_max_wogl'] = (pred_silence > args.slavc_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_wogl'] = (pred_silence > args.slavc_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_wogl'] = (pred_silence > args.slavc_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_silence_max_ogl'] = (pred_av_obj_silence > args.slavc_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_ogl'] = (pred_av_obj_silence > args.slavc_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_ogl'] = (pred_av_obj_silence > args.slavc_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_offscreen_max_wogl'] = (pred_offscreen > args.slavc_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_wogl'] = (pred_offscreen > args.slavc_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_wogl'] = (pred_offscreen > args.slavc_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_offscreen_max_ogl'] = (pred_av_obj_offscreen > args.slavc_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_ogl'] = (pred_av_obj_offscreen > args.slavc_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_ogl'] = (pred_av_obj_offscreen > args.slavc_th_maxq3).astype(int)
    
    elif 'fnac' in args.pth_name:
        binarized_negatives['binarized_pred_max_wogl'] = (pred > args.fnac_th_max).astype(int)
        binarized_negatives['binarized_pred_max10_wogl'] = (pred > args.fnac_th_max10).astype(int)
        binarized_negatives['binarized_pred_maxq3_wogl'] = (pred > args.fnac_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_noise_max_wogl'] = (pred_noise > args.fnac_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_wogl'] = (pred_noise > args.fnac_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_wogl'] = (pred_noise > args.fnac_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_noise_max_ogl'] = (pred_av_obj_noise > args.fnac_th_max).astype(int)
        binarized_negatives['binarized_pred_noise_max10_ogl'] = (pred_av_obj_noise > args.fnac_th_max10).astype(int)
        binarized_negatives['binarized_pred_noise_maxq3_ogl'] = (pred_av_obj_noise > args.fnac_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_silence_max_wogl'] = (pred_silence > args.fnac_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_wogl'] = (pred_silence > args.fnac_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_wogl'] = (pred_silence > args.fnac_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_silence_max_ogl'] = (pred_av_obj_silence > args.fnac_th_max).astype(int)
        binarized_negatives['binarized_pred_silence_max10_ogl'] = (pred_av_obj_silence > args.fnac_th_max10).astype(int)
        binarized_negatives['binarized_pred_silence_maxq3_ogl'] = (pred_av_obj_silence > args.fnac_th_maxq3).astype(int)

        binarized_negatives['binarized_pred_offscreen_max_wogl'] = (pred_offscreen > args.fnac_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_wogl'] = (pred_offscreen > args.fnac_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_wogl'] = (pred_offscreen > args.fnac_th_maxq3).astype(int)
        binarized_negatives['binarized_pred_offscreen_max_ogl'] = (pred_av_obj_offscreen > args.fnac_th_max).astype(int)
        binarized_negatives['binarized_pred_offscreen_max10_ogl'] = (pred_av_obj_offscreen > args.fnac_th_max10).astype(int)
        binarized_negatives['binarized_pred_offscreen_maxq3_ogl'] = (pred_av_obj_offscreen > args.fnac_th_maxq3).astype(int)
                
    binarized_negatives['binarized_pred_wogl'] = (pred > args.threshold).astype(int)
    binarized_negatives['binarized_pred_adap_wogl'] = (pred > threshold_ciouadap_wogl).astype(int)

    binarized_negatives['binarized_pred_ogl'] = (pred_av_obj > args.threshold).astype(int)
    binarized_negatives['binarized_pred_adap_ogl'] = (pred_av_obj > threshold_ciouadap_ogl).astype(int)

    return binarized_negatives

def append_metrics_binarized_pred(metrics_eval, binarized_negatives):
    metrics_eval['iou_pos_silence_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_wogl'], binarized_negatives['binarized_pred_silence_wogl']))
    metrics_eval['iou_pos_silence_adap_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_adap_wogl'], binarized_negatives['binarized_pred_silence_thrciouadap_wogl']))
    metrics_eval['iou_pos_noise_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_wogl'], binarized_negatives['binarized_pred_noise_wogl']))
    metrics_eval['iou_pos_noise_adap_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_adap_wogl'], binarized_negatives['binarized_pred_noise_thrciouadap_wogl']))
    metrics_eval['iou_pos_offscreen_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_wogl'], binarized_negatives['binarized_pred_offscreen_wogl']))
    metrics_eval['iou_pos_offscreen_adap_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_adap_wogl'], binarized_negatives['binarized_pred_offscreen_thrciouadap_wogl']))
    metrics_eval['iou_silence_noise_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_wogl'], binarized_negatives['binarized_pred_noise_wogl']))
    metrics_eval['iou_silence_noise_adap_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_thrciouadap_wogl'], binarized_negatives['binarized_pred_noise_thrciouadap_wogl']))
    metrics_eval['iou_silence_offscreen_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_wogl'], binarized_negatives['binarized_pred_offscreen_wogl']))
    metrics_eval['iou_silence_offscreen_adap_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_thrciouadap_wogl'], binarized_negatives['binarized_pred_offscreen_thrciouadap_wogl']))
    metrics_eval['iou_noise_offscreen_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_noise_wogl'], binarized_negatives['binarized_pred_offscreen_wogl']))
    metrics_eval['iou_noise_offscreen_adap_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_noise_thrciouadap_wogl'], binarized_negatives['binarized_pred_offscreen_thrciouadap_wogl']))

    metrics_eval['iou_pos_silence_max_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_max_wogl'], binarized_negatives['binarized_pred_silence_max_wogl']))
    metrics_eval['iou_pos_noise_max_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_max_wogl'], binarized_negatives['binarized_pred_noise_max_wogl']))
    metrics_eval['iou_pos_offscreen_max_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_max_wogl'], binarized_negatives['binarized_pred_offscreen_max_wogl']))
    metrics_eval['iou_silence_noise_max_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_max_wogl'], binarized_negatives['binarized_pred_noise_max_wogl']))
    metrics_eval['iou_silence_offscreen_max_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_max_wogl'], binarized_negatives['binarized_pred_offscreen_max_wogl']))
    metrics_eval['iou_noise_offscreen_max_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_noise_max_wogl'], binarized_negatives['binarized_pred_offscreen_max_wogl']))
    
    metrics_eval['iou_pos_silence_max10_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_max10_wogl'], binarized_negatives['binarized_pred_silence_max10_wogl']))
    metrics_eval['iou_pos_noise_max10_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_max10_wogl'], binarized_negatives['binarized_pred_noise_max10_wogl']))
    metrics_eval['iou_pos_offscreen_max10_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_max10_wogl'], binarized_negatives['binarized_pred_offscreen_max10_wogl']))
    metrics_eval['iou_silence_noise_max10_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_max10_wogl'], binarized_negatives['binarized_pred_noise_max10_wogl']))
    metrics_eval['iou_silence_offscreen_max10_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_max10_wogl'], binarized_negatives['binarized_pred_offscreen_max10_wogl']))
    metrics_eval['iou_noise_offscreen_max10_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_noise_max10_wogl'], binarized_negatives['binarized_pred_offscreen_max10_wogl']))

    metrics_eval['iou_pos_silence_maxq3_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_maxq3_wogl'], binarized_negatives['binarized_pred_silence_maxq3_wogl']))
    metrics_eval['iou_pos_noise_maxq3_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_maxq3_wogl'], binarized_negatives['binarized_pred_noise_maxq3_wogl']))
    metrics_eval['iou_pos_offscreen_maxq3_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_maxq3_wogl'], binarized_negatives['binarized_pred_offscreen_maxq3_wogl']))
    metrics_eval['iou_silence_noise_maxq3_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_maxq3_wogl'], binarized_negatives['binarized_pred_noise_maxq3_wogl']))
    metrics_eval['iou_silence_offscreen_maxq3_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_maxq3_wogl'], binarized_negatives['binarized_pred_offscreen_maxq3_wogl']))
    metrics_eval['iou_noise_offscreen_maxq3_wogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_noise_maxq3_wogl'], binarized_negatives['binarized_pred_offscreen_maxq3_wogl']))

    metrics_eval['iou_pos_silence_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_ogl'], binarized_negatives['binarized_pred_silence_ogl']))
    metrics_eval['iou_pos_silence_adap_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_adap_ogl'], binarized_negatives['binarized_pred_silence_thrciouadap_ogl']))
    metrics_eval['iou_pos_noise_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_ogl'], binarized_negatives['binarized_pred_noise_ogl']))
    metrics_eval['iou_pos_noise_adap_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_adap_ogl'], binarized_negatives['binarized_pred_noise_thrciouadap_ogl']))
    metrics_eval['iou_pos_offscreen_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_ogl'], binarized_negatives['binarized_pred_offscreen_ogl']))
    metrics_eval['iou_pos_offscreen_adap_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_adap_ogl'], binarized_negatives['binarized_pred_offscreen_thrciouadap_ogl']))
    metrics_eval['iou_silence_noise_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_ogl'], binarized_negatives['binarized_pred_noise_ogl']))
    metrics_eval['iou_silence_noise_adap_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_thrciouadap_ogl'], binarized_negatives['binarized_pred_noise_thrciouadap_ogl']))
    metrics_eval['iou_silence_offscreen_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_ogl'], binarized_negatives['binarized_pred_offscreen_ogl']))
    metrics_eval['iou_silence_offscreen_adap_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_silence_thrciouadap_ogl'], binarized_negatives['binarized_pred_offscreen_thrciouadap_ogl']))
    metrics_eval['iou_noise_offscreen_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_noise_ogl'], binarized_negatives['binarized_pred_offscreen_ogl']))
    metrics_eval['iou_noise_offscreen_adap_ogl'].append(compute_IOU_maps(binarized_negatives['binarized_pred_noise_thrciouadap_ogl'], binarized_negatives['binarized_pred_offscreen_thrciouadap_ogl']))

    metrics_eval['pia_batch_noise_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_wogl']))
    metrics_eval['pia_batch_noise_max_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_max_wogl']))
    metrics_eval['pia_batch_noise_max10_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_max10_wogl']))
    metrics_eval['pia_batch_noise_maxq3_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_maxq3_wogl']))
    metrics_eval['pia_batch_noise_ciou_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_thrciou_wogl']))
    metrics_eval['pia_batch_noise_ciouadap_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_thrciouadap_wogl']))
    
    metrics_eval['pia_batch_noise_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_ogl']))
    metrics_eval['pia_batch_noise_max_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_max_ogl']))
    metrics_eval['pia_batch_noise_max10_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_max10_ogl']))
    metrics_eval['pia_batch_noise_maxq3_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_maxq3_ogl']))
    metrics_eval['pia_batch_noise_ciou_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_thrciou_ogl']))
    metrics_eval['pia_batch_noise_ciouadap_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_noise_thrciouadap_ogl']))
    
    metrics_eval['pia_batch_silence_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_wogl']))
    metrics_eval['pia_batch_silence_max_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_max_wogl']))
    metrics_eval['pia_batch_silence_max10_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_max10_wogl']))
    metrics_eval['pia_batch_silence_maxq3_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_maxq3_wogl']))
    metrics_eval['pia_batch_silence_ciou_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_thrciou_wogl']))
    metrics_eval['pia_batch_silence_ciouadap_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_thrciouadap_wogl']))
    
    metrics_eval['pia_batch_silence_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_ogl']))
    metrics_eval['pia_batch_silence_max_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_max_ogl']))
    metrics_eval['pia_batch_silence_max10_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_max10_ogl']))
    metrics_eval['pia_batch_silence_maxq3_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_maxq3_ogl']))
    metrics_eval['pia_batch_silence_ciou_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_thrciou_ogl']))
    metrics_eval['pia_batch_silence_ciouadap_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_silence_thrciouadap_ogl']))
    
    metrics_eval['pia_batch_offscreen_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_wogl']))
    metrics_eval['pia_batch_offscreen_max_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_max_wogl']))
    metrics_eval['pia_batch_offscreen_max10_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_max10_wogl']))
    metrics_eval['pia_batch_offscreen_maxq3_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_maxq3_wogl']))
    metrics_eval['pia_batch_offscreen_ciou_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_thrciou_wogl']))
    metrics_eval['pia_batch_offscreen_ciouadap_wogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_thrciouadap_wogl']))
    
    metrics_eval['pia_batch_offscreen_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_ogl']))
    metrics_eval['pia_batch_offscreen_max_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_max_ogl']))
    metrics_eval['pia_batch_offscreen_max10_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_max10_ogl']))
    metrics_eval['pia_batch_offscreen_maxq3_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_maxq3_ogl']))
    metrics_eval['pia_batch_offscreen_ciou_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_thrciou_ogl']))
    metrics_eval['pia_batch_offscreen_ciouadap_ogl'].append(compute_pIA(binarized_negatives['binarized_pred_offscreen_thrciouadap_ogl']))
                    
    return metrics_eval

def compute_metrics(metrics_eval):
    metrics_eval['max_heatmap'] = np.mean(np.array(metrics_eval['max_heatmap_list']))
    metrics_eval['max_heatmap_silence'] = np.mean(np.array(metrics_eval['max_heatmap_silence_list']))
    metrics_eval['max_heatmap_noise'] = np.mean(np.array(metrics_eval['max_heatmap_noise_list']))
    metrics_eval['max_heatmap_offscreen'] = np.mean(np.array(metrics_eval['max_heatmap_offscreen_list']))

    metrics_eval['max_median_heatmap'] = np.median(np.array(metrics_eval['max_heatmap_list']))
    metrics_eval['max_median_heatmap_silence'] = np.median(np.array(metrics_eval['max_heatmap_silence_list']))
    metrics_eval['max_median_heatmap_noise'] = np.median(np.array(metrics_eval['max_heatmap_noise_list']))
    metrics_eval['max_median_heatmap_offscreen'] = np.median(np.array(metrics_eval['max_heatmap_offscreen_list']))

    metrics_eval['min_heatmap'] = np.mean(np.array(metrics_eval['min_heatmap_list']))
    metrics_eval['min_heatmap_silence'] = np.mean(np.array(metrics_eval['min_heatmap_silence_list']))
    metrics_eval['min_heatmap_noise'] = np.mean(np.array(metrics_eval['min_heatmap_noise_list']))
    metrics_eval['min_heatmap_offscreen'] = np.mean(np.array(metrics_eval['min_heatmap_offscreen_list']))

    metrics_eval['min_median_heatmap'] = np.median(np.array(metrics_eval['min_heatmap_list']))
    metrics_eval['min_median_heatmap_silence'] = np.median(np.array(metrics_eval['min_heatmap_silence_list']))
    metrics_eval['min_median_heatmap_noise'] = np.median(np.array(metrics_eval['min_heatmap_noise_list']))
    metrics_eval['min_median_heatmap_offscreen'] = np.median(np.array(metrics_eval['min_heatmap_offscreen_list']))
    
    metrics_eval['ciou_05_mean_wogl'] = np.mean(np.array(metrics_eval['ciou_05_wogl'])) * 100
    metrics_eval['auc_05_wogl'] = compute_AUC(metrics_eval['ciou_05_wogl']) * 100

    metrics_eval['ciou_max_mean_wogl'] = np.mean(np.array(metrics_eval['ciou_max_wogl'])) * 100
    metrics_eval['ciou_max10_mean_wogl'] = np.mean(np.array(metrics_eval['ciou_max10_wogl'])) * 100
    metrics_eval['ciou_maxq3_mean_wogl'] = np.mean(np.array(metrics_eval['ciou_maxq3_wogl'])) * 100
    metrics_eval['auc_max_wogl'] = compute_AUC(metrics_eval['ciou_max_wogl']) * 100
    metrics_eval['auc_max10_wogl'] = compute_AUC(metrics_eval['ciou_max10_wogl']) * 100
    metrics_eval['auc_maxq3_wogl'] = compute_AUC(metrics_eval['ciou_maxq3_wogl']) * 100

    metrics_eval['ciou_05_mean_ogl'] = np.mean(np.array(metrics_eval['ciou_05_ogl'])) * 100
    metrics_eval['auc_05_ogl'] = compute_AUC(metrics_eval['ciou_05_ogl']) * 100

    metrics_eval['ciou_max_mean_ogl'] = np.mean(np.array(metrics_eval['ciou_max_ogl'])) * 100
    metrics_eval['ciou_max10_mean_ogl'] = np.mean(np.array(metrics_eval['ciou_max10_ogl'])) * 100
    metrics_eval['ciou_maxq3_mean_ogl'] = np.mean(np.array(metrics_eval['ciou_maxq3_ogl'])) * 100
    metrics_eval['auc_max_ogl'] = compute_AUC(metrics_eval['ciou_max_ogl']) * 100
    metrics_eval['auc_max10_ogl'] = compute_AUC(metrics_eval['ciou_max10_ogl']) * 100
    metrics_eval['auc_maxq3_ogl'] = compute_AUC(metrics_eval['ciou_maxq3_ogl']) * 100

    metrics_eval['iou_pos_silence_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_silence_wogl'])) * 100
    metrics_eval['iou_pos_silence_adap_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_silence_adap_wogl'])) * 100
    metrics_eval['iou_pos_noise_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_noise_wogl'])) * 100
    metrics_eval['iou_pos_noise_adap_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_noise_adap_wogl'])) * 100
    metrics_eval['iou_pos_offscreen_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_offscreen_wogl'])) * 100
    metrics_eval['iou_pos_offscreen_adap_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_offscreen_adap_wogl'])) * 100
    metrics_eval['iou_silence_noise_wogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_noise_wogl'])) * 100
    metrics_eval['iou_silence_noise_adap_wogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_noise_adap_wogl'])) * 100
    metrics_eval['iou_silence_offscreen_wogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_offscreen_wogl'])) * 100
    metrics_eval['iou_silence_offscreen_adap_wogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_offscreen_adap_wogl'])) * 100
    metrics_eval['iou_noise_offscreen_wogl_metric'] = np.mean(np.array(metrics_eval['iou_noise_offscreen_wogl'])) * 100
    metrics_eval['iou_noise_offscreen_adap_wogl_metric'] = np.mean(np.array(metrics_eval['iou_noise_offscreen_adap_wogl'])) * 100

    metrics_eval['iou_pos_silence_max_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_silence_max_wogl'])) * 100
    metrics_eval['iou_pos_noise_max_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_noise_max_wogl'])) * 100
    metrics_eval['iou_pos_offscreen_max_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_offscreen_max_wogl'])) * 100
    metrics_eval['iou_silence_noise_max_wogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_noise_max_wogl'])) * 100
    metrics_eval['iou_silence_offscreen_max_wogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_offscreen_max_wogl'])) * 100
    metrics_eval['iou_noise_offscreen_max_wogl_metric'] = np.mean(np.array(metrics_eval['iou_noise_offscreen_max_wogl'])) * 100

    metrics_eval['iou_pos_silence_max10_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_silence_max10_wogl'])) * 100
    metrics_eval['iou_pos_noise_max10_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_noise_max10_wogl'])) * 100
    metrics_eval['iou_pos_offscreen_max10_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_offscreen_max10_wogl'])) * 100
    metrics_eval['iou_silence_noise_max10_wogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_noise_max10_wogl'])) * 100
    metrics_eval['iou_silence_offscreen_max10_wogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_offscreen_max10_wogl'])) * 100
    metrics_eval['iou_noise_offscreen_max10_wogl_metric'] = np.mean(np.array(metrics_eval['iou_noise_offscreen_max10_wogl'])) * 100

    metrics_eval['iou_pos_silence_maxq3_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_silence_maxq3_wogl'])) * 100
    metrics_eval['iou_pos_noise_maxq3_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_noise_maxq3_wogl'])) * 100
    metrics_eval['iou_pos_offscreen_maxq3_wogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_offscreen_maxq3_wogl'])) * 100
    metrics_eval['iou_silence_noise_maxq3_wogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_noise_maxq3_wogl'])) * 100
    metrics_eval['iou_silence_offscreen_maxq3_wogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_offscreen_maxq3_wogl'])) * 100
    metrics_eval['iou_noise_offscreen_maxq3_wogl_metric'] = np.mean(np.array(metrics_eval['iou_noise_offscreen_maxq3_wogl'])) * 100

    metrics_eval['iou_pos_silence_ogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_silence_ogl'])) * 100
    metrics_eval['iou_pos_silence_adap_ogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_silence_adap_ogl'])) * 100
    metrics_eval['iou_pos_noise_ogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_noise_ogl'])) * 100
    metrics_eval['iou_pos_noise_adap_ogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_noise_adap_ogl'])) * 100
    metrics_eval['iou_pos_offscreen_ogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_offscreen_ogl'])) * 100
    metrics_eval['iou_pos_offscreen_adap_ogl_metric'] = np.mean(np.array(metrics_eval['iou_pos_offscreen_adap_ogl'])) * 100
    metrics_eval['iou_silence_noise_ogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_noise_ogl'])) * 100
    metrics_eval['iou_silence_noise_adap_ogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_noise_adap_ogl'])) * 100
    metrics_eval['iou_silence_offscreen_ogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_offscreen_ogl'])) * 100
    metrics_eval['iou_silence_offscreen_adap_ogl_metric'] = np.mean(np.array(metrics_eval['iou_silence_offscreen_adap_ogl'])) * 100
    metrics_eval['iou_noise_offscreen_ogl_metric'] = np.mean(np.array(metrics_eval['iou_noise_offscreen_ogl'])) * 100
    metrics_eval['iou_noise_offscreen_adap_ogl_metric'] = np.mean(np.array(metrics_eval['iou_noise_offscreen_adap_ogl'])) * 100
    
    metrics_eval['pia_metric_noise_wogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_wogl'])) * 100
    metrics_eval['pia_metric_noise_max_wogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_max_wogl'])) * 100
    metrics_eval['pia_metric_noise_max10_wogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_max10_wogl'])) * 100
    metrics_eval['pia_metric_noise_maxq3_wogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_maxq3_wogl'])) * 100
    metrics_eval['pia_metric_silence_wogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_wogl'])) * 100
    metrics_eval['pia_metric_silence_max_wogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_max_wogl'])) * 100
    metrics_eval['pia_metric_silence_max10_wogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_max10_wogl'])) * 100
    metrics_eval['pia_metric_silence_maxq3_wogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_maxq3_wogl'])) * 100
    metrics_eval['pia_metric_offscreen_wogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_wogl'])) * 100
    metrics_eval['pia_metric_offscreen_max_wogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_max_wogl'])) * 100
    metrics_eval['pia_metric_offscreen_max10_wogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_max10_wogl'])) * 100
    metrics_eval['pia_metric_offscreen_maxq3_wogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_maxq3_wogl'])) * 100

    metrics_eval['auc_noise_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_wogl']) * 100
    metrics_eval['auc_noise_max_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_max_wogl']) * 100
    metrics_eval['auc_noise_max10_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_max10_wogl']) * 100
    metrics_eval['auc_noise_maxq3_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_maxq3_wogl']) * 100
    metrics_eval['auc_silence_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_wogl']) * 100
    metrics_eval['auc_silence_max_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_max_wogl']) * 100
    metrics_eval['auc_silence_max10_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_max10_wogl']) * 100
    metrics_eval['auc_silence_maxq3_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_maxq3_wogl']) * 100
    metrics_eval['auc_offscreen_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_wogl']) * 100
    metrics_eval['auc_offscreen_max_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_max_wogl']) * 100
    metrics_eval['auc_offscreen_max10_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_max10_wogl']) * 100
    metrics_eval['auc_offscreen_maxq3_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_maxq3_wogl']) * 100
    
    metrics_eval['pia_metric_noise_ciou_wogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_ciou_wogl'])) * 100
    metrics_eval['auc_noise_ciou_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_ciou_wogl']) * 100
    metrics_eval['pia_metric_silence_ciou_wogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_ciou_wogl'])) * 100
    metrics_eval['auc_silence_ciou_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_ciou_wogl']) * 100
    metrics_eval['pia_metric_offscreen_ciou_wogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_ciou_wogl'])) * 100
    metrics_eval['auc_offscreen_ciou_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_ciou_wogl']) * 100
    
    metrics_eval['pia_metric_noise_ciouadap_wogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_ciouadap_wogl'])) * 100
    metrics_eval['auc_noise_ciouadap_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_ciouadap_wogl']) * 100
    metrics_eval['pia_metric_silence_ciouadap_wogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_ciouadap_wogl'])) * 100
    metrics_eval['auc_silence_ciouadap_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_ciouadap_wogl']) * 100
    metrics_eval['pia_metric_offscreen_ciouadap_wogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_ciouadap_wogl'])) * 100
    metrics_eval['auc_offscreen_ciouadap_wogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_ciouadap_wogl']) * 100
    
    metrics_eval['pia_metric_noise_ogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_ogl'])) * 100
    metrics_eval['pia_metric_noise_max_ogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_max_ogl'])) * 100
    metrics_eval['pia_metric_noise_max10_ogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_max10_ogl'])) * 100
    metrics_eval['pia_metric_noise_maxq3_ogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_maxq3_ogl'])) * 100
    metrics_eval['pia_metric_silence_ogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_ogl'])) * 100
    metrics_eval['pia_metric_silence_max_ogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_max_ogl'])) * 100
    metrics_eval['pia_metric_silence_max10_ogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_max10_ogl'])) * 100
    metrics_eval['pia_metric_silence_maxq3_ogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_maxq3_ogl'])) * 100
    metrics_eval['pia_metric_offscreen_ogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_ogl'])) * 100
    metrics_eval['pia_metric_offscreen_max_ogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_max_ogl'])) * 100
    metrics_eval['pia_metric_offscreen_max10_ogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_max10_ogl'])) * 100
    metrics_eval['pia_metric_offscreen_maxq3_ogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_maxq3_ogl'])) * 100

    metrics_eval['auc_noise_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_ogl']) * 100
    metrics_eval['auc_noise_max_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_max_ogl']) * 100
    metrics_eval['auc_noise_max10_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_max10_ogl']) * 100
    metrics_eval['auc_noise_maxq3_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_maxq3_ogl']) * 100
    metrics_eval['auc_silence_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_ogl']) * 100
    metrics_eval['auc_silence_max_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_max_ogl']) * 100
    metrics_eval['auc_silence_max10_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_max10_ogl']) * 100
    metrics_eval['auc_silence_maxq3_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_maxq3_ogl']) * 100
    metrics_eval['auc_offscreen_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_ogl']) * 100
    metrics_eval['auc_offscreen_max_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_max_ogl']) * 100
    metrics_eval['auc_offscreen_max10_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_max10_ogl']) * 100
    metrics_eval['auc_offscreen_maxq3_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_maxq3_ogl']) * 100
    
    metrics_eval['pia_metric_noise_ciou_ogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_ciou_ogl'])) * 100
    metrics_eval['auc_noise_ciou_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_ciou_ogl']) * 100
    metrics_eval['pia_metric_silence_ciou_ogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_ciou_ogl'])) * 100
    metrics_eval['auc_silence_ciou_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_ciou_ogl']) * 100
    metrics_eval['pia_metric_offscreen_ciou_ogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_ciou_ogl'])) * 100
    metrics_eval['auc_offscreen_ciou_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_ciou_ogl']) * 100
    
    metrics_eval['pia_metric_noise_ciouadap_ogl'] = np.mean(np.array(metrics_eval['pia_batch_noise_ciouadap_ogl'])) * 100
    metrics_eval['auc_noise_ciouadap_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_noise_ciouadap_ogl']) * 100
    metrics_eval['pia_metric_silence_ciouadap_ogl'] = np.mean(np.array(metrics_eval['pia_batch_silence_ciouadap_ogl'])) * 100
    metrics_eval['auc_silence_ciouadap_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_silence_ciouadap_ogl']) * 100
    metrics_eval['pia_metric_offscreen_ciouadap_ogl'] = np.mean(np.array(metrics_eval['pia_batch_offscreen_ciouadap_ogl'])) * 100
    metrics_eval['auc_offscreen_ciouadap_ogl'] = compute_AUC_pIA(metrics_eval['pia_batch_offscreen_ciouadap_ogl']) * 100

    metrics_eval['pia_mean_neg'] = np.mean([100 - metrics_eval['pia_metric_silence_maxq3_wogl'], 100 - metrics_eval['pia_metric_noise_maxq3_wogl'], 100 - metrics_eval['pia_metric_offscreen_maxq3_wogl']])

    metrics_eval['f_pos_neg'] = compute_f(metrics_eval['ciou_maxq3_mean_wogl'], metrics_eval['pia_mean_neg'])
    metrics_eval['f_ciou_pos_silence'] = compute_f(metrics_eval['ciou_maxq3_mean_wogl'], (100 - metrics_eval['pia_metric_silence_maxq3_wogl']))
    metrics_eval['f_ciou_pos_noise'] = compute_f(metrics_eval['ciou_maxq3_mean_wogl'], (100 - metrics_eval['pia_metric_noise_maxq3_wogl']))
    metrics_eval['f_ciou_pos_offscreen'] = compute_f(metrics_eval['ciou_maxq3_mean_wogl'], (100 - metrics_eval['pia_metric_offscreen_maxq3_wogl']))

    metrics_eval['aoc_mean_neg'] = np.mean([metrics_eval['auc_silence_maxq3_wogl'], metrics_eval['auc_noise_maxq3_wogl'], metrics_eval['auc_offscreen_maxq3_wogl']])

    metrics_eval['f_auc_aoc_pos_neg'] = compute_f(metrics_eval['auc_maxq3_wogl'], metrics_eval['aoc_mean_neg'])
    metrics_eval['f_auc_aoc_pos_silence'] = compute_f(metrics_eval['auc_maxq3_wogl'], (metrics_eval['auc_silence_maxq3_wogl']))
    metrics_eval['f_auc_aoc_pos_noise'] = compute_f(metrics_eval['auc_maxq3_wogl'], (metrics_eval['auc_noise_maxq3_wogl']))
    metrics_eval['f_auc_aoc_pos_offscreen'] = compute_f(metrics_eval['auc_maxq3_wogl'], (metrics_eval['auc_offscreen_maxq3_wogl']))
    
    return metrics_eval
