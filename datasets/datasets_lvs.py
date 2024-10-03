import os
import fnmatch
import pickle
import copy
import cv2
import json
import torch
import csv
import librosa
import skvideo.io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor, Compose
import pdb
import time
from PIL import Image
import glob
import sys 
import scipy.io.wavfile as wav
from scipy import signal
import random
import soundfile as sf
import xml.etree.ElementTree as ET

# Fixing random seed for reproducibility
# random.seed(42)

def vgg_filename(name):
    return '_'.join([name[:11],str(int(name[12:])*1000),str((int(name[12:])+10)*1000)])


def load_all_bboxes(annotation_dir, format='flickr'):
    gt_bboxes = {}
    if format == 'flickr':
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.')[0]
            gt = ET.parse(f"{annotation_dir}/{filename}").getroot()
            bboxes = []
            for child in gt:
                for childs in child:
                    bbox = []
                    if childs.tag == 'bbox':
                        for index, ch in enumerate(childs):
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text)/256))
                    bboxes.append(bbox)
            gt_bboxes[file] = bboxes

    elif format == 'vggss':
        with open('metadata/vggss_annotations.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in annotation['bbox']]
            filename = annotation['file']
            gt_bboxes[filename] = bboxes

    if format == 'is3':
        with open('metadata/synthetic3240_bbox.json') as fi:
            annotations = json.load(fi)
        for annotation in annotations:
            bboxes = [[int(bbox) for bbox in annotation['gt_box']]]
            gt_bboxes[annotation['image'].split('/')[-1].split('.')[0]] = bboxes
    if format == 'vposs':
        with open('metadata/vpo_ss_bbox.json') as fi:
            annotations = json.load(fi)
        for annotation in annotations:
            bboxes = [[int(bbox) for bbox in annotation['gt_box']]]
            gt_bboxes[annotation['image'].split('/')[-1].split('.')[0]] = bboxes
    if format == 'vpoms':
        with open('metadata/vpo_ms_bbox.json') as fi:
            annotations = json.load(fi)
        for annotation in annotations:
            bboxes = [[int(bbox) for bbox in annotation['gt_box']]]
            gt_bboxes[annotation['image'].split('/')[-1].split('.')[0]] = bboxes
    if format == 'ms3':
        with open('metadata/ms3_box.json') as fi:
            annotations = json.load(fi)
        for annotation in annotations:
            bboxes = [[int(bbox) for bbox in annotation['gt_box']]]
            gt_bboxes['/'.join(annotation['image'].split('/')[-2:])[:-4]] = bboxes
    if format == 's4':
        with open('metadata/s4_box.json') as fi:
            annotations = json.load(fi)
        for annotation in annotations:
            bboxes = [[int(bbox) for bbox in annotation['gt_box']]]
            gt_bboxes['/'.join(annotation['image'].split('/')[-3:])[:-4]] = bboxes
    
    return gt_bboxes



def bbox2gtmap(bboxes, format='flickr'):
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp

    if format == 'flickr':
        # Annotation consensus
        gt_map = gt_map / 2
        gt_map[gt_map > 1] = 1

    else:#if format == 'vggss':
        # Single annotation
        gt_map[gt_map > 0] = 1

    return gt_map



class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None, file_list=[]):
        
        self.audio_path = f'{args.data_dir}/audio/'
        self.image_path = f'{args.data_dir}/frames/'
        
        self.imgSize = args.image_size 
        self.args = args
        
        self.mode = mode
        self.transforms = transforms
        # initialize video transform
        self._init_atransform()
        self._init_transform()
        #  Retrieve list of audio and video files
        self.video_files = []

        data = []
        if args.testset == 'flickr':
            testcsv = 'metadata/flickr_test.csv'
        elif args.testset == 'vggss':
            testcsv = 'metadata/vggss_annotations.json'
            new_classes_json = "metadata/vggss_class_files_dict.json"

            with open(new_classes_json, 'r') as file:
                self.new_classes_data = json.load(file)

        elif args.testset == 'is3':
            testcsv = 'metadata/IS3_annotation.json'
            new_classes_json = "metadata/IS3_class_files_dict.json"

            with open(new_classes_json, 'r') as file:
                self.new_classes_data = json.load(file)

        elif args.testset == 'ms3':
            testcsv = 'metadata/ms3_box.json'
        elif args.testset == 's4':
            testcsv = 'metadata/s4_box.json'
        elif args.testset == 'vposs':
            testcsv = 'metadata/vpo_ss_bbox.json'
        elif args.testset == 'vpoms':
            testcsv = 'metadata/vpo_ms_bbox.json'
        
        self.audio_length = 10
        self.st = 3.5
        self.fi = 6.5
            
        if 'json' in testcsv:
            with open(testcsv) as fi:
                jsonfile = json.load(fi)

            self.all_bboxes = load_all_bboxes(args.test_gt_path, format=args.testset)
            
            if args.testset == 'ms3':
                self.audio_length = 5
                self.st = 1
                self.fi = 4
                
                self.audio_files = [fn['audio'].split('/')[-1] for fn in jsonfile]
                image_files = ['/'.join(fn['image'].split('/')[-2:]) for fn in jsonfile]

                self.video_files = ['/'.join(fn['image'].split('/')[-2:]) for fn in jsonfile]

                self.audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
                self.image_path = '/'.join(jsonfile[0]['image'].split('/')[:-2])
            elif args.testset == 's4':
                self.audio_length = 5
                self.st = 1
                self.fi = 4
                
                self.audio_files = ['/'.join(fn['audio'].split('/')[-2:]) for fn in jsonfile]
                image_files = ['/'.join(fn['image'].split('/')[-3:]) for fn in jsonfile]

                self.video_files = ['/'.join(fn['image'].split('/')[-3:]) for fn in jsonfile]

                self.audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-2])
                self.image_path = '/'.join(jsonfile[0]['image'].split('/')[:-3])
            else:
                if args.testset == "vggss":
                    self.files = [fn['file'].split('/')[-1] for fn in jsonfile]
                    self.audio_files = [fn+".wav" for fn in self.files]
                    self.image_files = [fn+".jpg" for fn in self.files]
                    self.video_files = [fn+".mp4" for fn in self.files]

                    self.audio_path = f"{args.data_dir}/audio"
                    self.image_path = f"{args.data_dir}/frames"
                
                elif args.testset == "is3":
                    self.audio_files = [fn['audio'].split('/')[-1] for fn in jsonfile]
                    image_files = [fn['image'].split('/')[-1] for fn in jsonfile]

                    self.video_files = [fn['image'].split('/')[-1] for fn in jsonfile]

                    self.audio_path = f"{args.data_dir}/audio_wav"
                    self.image_path = f"{args.data_dir}/images"
                    
                else: 
                    self.audio_files = [fn['audio'].split('/')[-1] for fn in jsonfile]
                    image_files = [fn['image'].split('/')[-1] for fn in jsonfile]

                    self.video_files = [fn['image'].split('/')[-1] for fn in jsonfile]

                    self.audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
                    self.image_path = '/'.join(jsonfile[0]['image'].split('/')[:-1])
            
            
        else:
            with open(testcsv) as f:
                csv_reader = csv.reader(f)
                for item in csv_reader:
                    data.append(item[0] + '.jpg')
            self.all_bboxes = load_all_bboxes(args.test_gt_path, format=args.testset)
            if args.testset == 'vggss':
                exists = os.listdir(os.path.join(args.data_dir, 'frames'))
                exists = set(exists)-set(['7XQN9XDnRm4_80000_90000'])
                exists = set([x+'.mp4' for x in exists])
#                 exists = set([x[:12]+str(int(x[12:].split('_')[0])//1000).zfill(6)+'.mp4' for x in exists])
                data = set(data).intersection(exists)
            for item in data:
                self.video_files.append(item )

        self.count = 0

        if len(file_list) > 0:
            self.audio_files = [fn+".wav" for fn in file_list]
            image_files = [fn+".jpg" for fn in file_list]
            self.video_files = [fn+".jpg" for fn in file_list]

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])      ## where we got 85.2 on flickr
            '''Does order of normalization matters?'''
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.imgSize,self.imgSize), transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.Normalize(mean, std)])      ## setting for the tables on the overleaf now

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])
#  

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img
    
    def get_key(self, dct, val):
        list_keys = list(dct.keys())
        for key in list_keys:
            values = dct[key]
            if val in values:
                return key
        
        return None
    
    def find_files_with_prefix(self, directory, prefix):
        matching_files = []
        
        for filename in os.listdir(directory):
            if fnmatch.fnmatch(filename, f"{prefix}*"):
                matching_files.append(filename)
        
        return matching_files

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        # Image
        file = self.video_files[idx]
        file_root = file.split(".")[0]

        # class_file = self.new_classes_data
        new_classes = list(self.new_classes_data.keys())

        if self.args.testset == "is3":

            prefix_file = "_".join(file_root.split("_")[:-1])
            files_newclasses = self.find_files_with_prefix(self.image_path, prefix_file)

            new_classes_list = []
            for file_newclass in files_newclasses:
                new_classes_list.append(self.get_key(self.new_classes_data, file_newclass.split(".")[0]))

            if new_classes_list[0] != new_classes_list[1]:
                new_classes.remove(new_classes_list[0])
                new_classes.remove(new_classes_list[1])
            else:
                new_classes.remove(new_classes_list[0])
        else:
            new_class = self.get_key(self.new_classes_data, file_root)
            if new_class == None:
                print(file_root+".wav")
            new_classes.remove(new_class)
        
        # random.seed(self.random_seed)
        random_class = random.choice(sorted(new_classes))
        random_file = random.choice(self.new_classes_data[random_class])
        # print(f'Offscreen used: {random_file}')
                
        if self.args.testset == 'flickr':
            jpg = file.replace(".mp4", ".jpg")
            # jpg = os.listdir(self.image_path)
            # jpg = [x for x in jpg if x[-3:]=='jpg'][0]
            filename = os.path.join(self.image_path ,jpg)
            audiofilename = self.audio_path + file[:-3]+'wav'
            
        elif self.args.testset == 'vggss':
            audio_file = self.audio_files[idx]
            image_file = self.image_files[idx]
            audio_file_offscreen = os.path.join(self.audio_path, f"{random_file}.wav")
#             filetmp = filetmp[:11]+'_'+str(int(filetmp[12:])*1000)+'_'+str((int(filetmp[12:])+10)*1000)
            filename = os.path.join(self.image_path, image_file)
            audiofilename = os.path.join(self.audio_path, audio_file)
        elif self.args.testset == 'is3' or self.args.testset == 'vposs' or self.args.testset == 'vpoms':
            filename = os.path.join(self.image_path,file)
            audiofilename = os.path.join(self.audio_path,self.audio_files[idx])
            audio_file_offscreen = os.path.join(self.audio_path, f"{random_file}.wav")
        elif self.args.testset == 'ms3' or self.args.testset == 's4':
            filename = os.path.join(self.image_path,file)
            audiofilename = os.path.join(self.audio_path,self.audio_files[idx])
            
        frame = self.img_transform(self._load_frame(filename))
        frame_ori = np.array(self._load_frame(filename))
        # Audio
        # print(f"Audio path: {audiofilename}")
        # print(f"Audio path: {audio_file_offscreen}")


        samples, samplerate = sf.read(audiofilename)
        samples_offscreen, samplerate = sf.read(audio_file_offscreen)
        
        if len(samples.shape) > 1:
            if samples.shape[1] == 2:
                samples = samples[:,0]
        if len(samples_offscreen.shape) > 1:
            if samples_offscreen.shape[1] == 2:
                samples_offscreen = samples_offscreen[:,0]

        # repeat if audio is too short
        if samples.shape[0] < samplerate * self.audio_length:
            n = int(samplerate * self.audio_length / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:samplerate*self.audio_length]
        resamples = resamples[int(16000*self.st):int(16000*self.fi)]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        
        if samples_offscreen.shape[0] < samplerate * self.audio_length:
            n = int(samplerate * self.audio_length / samples_offscreen.shape[0]) + 1
            samples_offscreen = np.tile(samples_offscreen, n)
        resamples_offscreen = samples_offscreen[:samplerate*self.audio_length]
        resamples_offscreen = resamples_offscreen[int(16000*self.st):int(16000*self.fi)]

        resamples_offscreen[resamples_offscreen > 1.] = 1.
        resamples_offscreen[resamples_offscreen < -1.] = -1.
        
        resamples_noise = np.random.normal(0, 1, len(resamples))
        frequencies, times, spectrogram_noise = signal.spectrogram(resamples_noise,samplerate, nperseg=512,noverlap=353)
        spectrogram_noise = np.log(spectrogram_noise+ 1e-7)
        spectrogram_noise = self.aid_transform(spectrogram_noise)

        resamples_silence = np.zeros_like(resamples)
        frequencies, times, spectrogram_silence = signal.spectrogram(resamples_silence,samplerate, nperseg=512,noverlap=353)
        spectrogram_silence = np.log(spectrogram_silence+ 1e-7)
        spectrogram_silence = self.aid_transform(spectrogram_silence)
        
        frequencies, times, spectrogram = signal.spectrogram(resamples,samplerate, nperseg=512,noverlap=353)
        frequencies, times, spectrogram_offscreen = signal.spectrogram(resamples_offscreen,samplerate, nperseg=512,noverlap=353)
        
        spectrogram = np.log(spectrogram+ 1e-7)
        spectrogram_offscreen = np.log(spectrogram_offscreen+ 1e-7)
        
        spectrogram = self.aid_transform(spectrogram)
        spectrogram_offscreen = self.aid_transform(spectrogram_offscreen)
        
        bboxes = {}
        if self.all_bboxes is not None:
            bb = -torch.ones((10, 4)).long()
            tmpbox = self.all_bboxes[file[:-4]]
            bb[:len(tmpbox)] = torch.from_numpy(np.array(tmpbox))
            bboxes['bboxes'] = bb
            
            bboxes['gt_map'] = bbox2gtmap(self.all_bboxes[file[:-4]], self.args.testset)
        
            return frame, spectrogram, spectrogram_noise, spectrogram_silence, spectrogram_offscreen, bboxes, file

class AudioVisualDataset(Dataset):

    def __init__(self, 
                data_dir="", 
                image_size=224, 
                audio_length=3, 
                audio_shift_min = 0.1,
                audio_shift_max = 0.5,
                sample_rate=16000, 
                files=[], 
                mode='train', 
                transforms=None
                ):

        self.data_dir = data_dir
        self.imgSize = image_size

        self.audio_shift_min = audio_shift_min
        self.audio_shift_max = audio_shift_max
        
        self.mode = mode
        self.transforms = transforms
        self.files = files

        # initialize video transform
        self._init_atransform()
        self._init_transform()
        self.transform_224 = Compose([Resize((self.imgSize, self.imgSize), Image.BICUBIC), 
                                      ToTensor()])
        
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.st = 3.5
        self.fi = 6.5
        self.error_data = False

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])      ## where we got 85.2 on flickr
            '''Does order of normalization matters?'''
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.imgSize,self.imgSize), transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.Normalize(mean, std)])      ## setting for the tables on the overleaf now

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.files)  # self.length

    def __getitem__(self, idx):
        f = self.files[idx]
        audio_path = os.path.join(self.data_dir, "train", "audio", f + ".wav")
        video_path = os.path.join(self.data_dir, "train", "video", f + ".mp4")
        class_path = os.path.join(self.data_dir, "train", "classes", f"{f}.txt")
        with open(class_path, "r") as file:
            class_vggsound = file.readlines()[0]

        audio_shift = round(round(random.uniform(self.audio_shift_min, self.audio_shift_max), 2) * self.sample_rate)
        audio_shift = random.choice([audio_shift, -audio_shift])

        # audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        samples, samplerate = sf.read(audio_path)
        if len(samples.shape) > 1:
            if samples.shape[1] == 2:
                samples = samples[:,0]

        video = skvideo.io.vread(video_path)
        video_duration = samples.shape[0] / samplerate
        frame_rate = int(video.shape[0] / video_duration)
        
        num_audio_samples = self.audio_length * samplerate

        if (samples.shape[0] >= num_audio_samples):
            if ((frame_rate * 0.7*self.audio_length) <= (video.shape[0] - (frame_rate * 0.7*self.audio_length))):
                
                audio_center = random.randint(int(samplerate * 0.6 * self.audio_length), samples.shape[0] - int(samplerate * 0.6 * self.audio_length))
                audio_start = audio_center - (num_audio_samples // 2)
                audio_slice = slice(audio_start, audio_start + num_audio_samples)
                
                video_frame = int(audio_center * frame_rate / samplerate)
                audio_index_aug = audio_center + audio_shift
                audio_start_aug = audio_index_aug - (num_audio_samples // 2)
                audio_slice_aug = slice(audio_start_aug, audio_start_aug + num_audio_samples)

                if samples[audio_slice].shape[0] == num_audio_samples and samples[audio_slice_aug].shape[0] == num_audio_samples:
                    audio_orig = samples[audio_slice]
                    frequencies, times, spectrogram_orig = signal.spectrogram(audio_orig, samplerate, nperseg=512, noverlap=353)
                    spectrogram_orig = np.log(spectrogram_orig+ 1e-7)
                    spectrogram_orig = self.aid_transform(spectrogram_orig)

                    audio_aug = samples[audio_slice_aug]
                    frequencies, times, spectrogram_aug = signal.spectrogram(audio_aug, samplerate, nperseg=512, noverlap=353)
                    spectrogram_aug = np.log(spectrogram_aug+ 1e-7)
                    spectrogram_aug = self.aid_transform(spectrogram_aug)

                    frame_orig = self.transform_224(Image.fromarray(video[video_frame]))
                    frame_augm = self.img_transform(Image.fromarray(video[video_frame]))

                    return spectrogram_orig, spectrogram_aug, frame_orig, frame_augm, class_vggsound
                else:
                    self.error_data = True

            else:
                video_frame = video.shape[0] // 2
                audio_index = samples.shape[0] // 2
                audio_start = audio_index - (num_audio_samples // 2)
                audio_slice = slice(audio_start, audio_start + num_audio_samples)

                audio_index_aug = audio_index + audio_shift
                audio_start_aug = audio_index_aug - (num_audio_samples // 2)
                audio_slice_aug = slice(audio_start_aug, audio_start_aug + num_audio_samples)

                if samples[audio_slice].shape[0] == num_audio_samples and samples[audio_slice_aug].shape[0] == num_audio_samples:
                    audio_orig = samples[audio_slice]
                    frequencies, times, spectrogram_orig = signal.spectrogram(audio_orig, samplerate, nperseg=512, noverlap=353)
                    spectrogram_orig = np.log(spectrogram_orig+ 1e-7)
                    spectrogram_orig = self.aid_transform(spectrogram_orig)

                    audio_aug = samples[audio_slice_aug]
                    frequencies, times, spectrogram_aug = signal.spectrogram(audio_aug, samplerate, nperseg=512, noverlap=353)
                    spectrogram_aug = np.log(spectrogram_aug+ 1e-7)
                    spectrogram_aug = self.aid_transform(spectrogram_aug)

                    frame_orig = self.transform_224(Image.fromarray(video[video_frame]))
                    frame_augm = self.img_transform(Image.fromarray(video[video_frame]))

                    return spectrogram_orig, spectrogram_aug, frame_orig, frame_augm, class_vggsound
                else:
                    self.error_data = True
        else:
            raise f"Error, video shorter than {self.audio_length} seconds."
        
        if self.error_data == True:
            video_frame = video.shape[0] // 2
            audio_index = samples.shape[0] // 2
            audio_start = audio_index - (num_audio_samples // 2)
            audio_slice = slice(audio_start, audio_start + num_audio_samples)

            audio_index_aug = audio_index + audio_shift
            audio_start_aug = audio_index_aug - (num_audio_samples // 2)
            audio_slice_aug = slice(audio_start_aug, audio_start_aug + num_audio_samples)

            if samples[audio_slice].shape[0] == num_audio_samples and samples[audio_slice_aug].shape[0] == num_audio_samples:
                audio_orig = samples[audio_slice]
                frequencies, times, spectrogram_orig = signal.spectrogram(audio_orig, samplerate, nperseg=512, noverlap=353)
                spectrogram_orig = np.log(spectrogram_orig+ 1e-7)
                spectrogram_orig = self.aid_transform(spectrogram_orig)

                audio_aug = samples[audio_slice_aug]
                frequencies, times, spectrogram_aug = signal.spectrogram(audio_aug, samplerate, nperseg=512, noverlap=353)
                spectrogram_aug = np.log(spectrogram_aug+ 1e-7)
                spectrogram_aug = self.aid_transform(spectrogram_aug)

                frame_orig = self.transform_224(Image.fromarray(video[video_frame]))
                frame_augm = self.img_transform(Image.fromarray(video[video_frame]))

                self.error_data = False

                return spectrogram_orig, spectrogram_aug, frame_orig, frame_augm, class_vggsound
        
            print(f"File: {f} - {num_audio_samples}\n{samples[audio_slice].shape[0]}\n{samples[audio_slice_aug].shape[0]}\n")
            return None



class AudioVisualDataset2(Dataset):

    def __init__(self, 
                data_dir="", 
                image_size=224, 
                audio_length=3, 
                audio_shift_min = 0.1,
                audio_shift_max = 0.5,
                sample_rate=16000, 
                files=[], 
                mode='train', 
                transforms=None
                ):

        self.data_dir = data_dir
        self.imgSize = image_size

        self.audio_shift_min = audio_shift_min
        self.audio_shift_max = audio_shift_max
        
        self.mode = mode
        self.transforms = transforms
        self.files = files

        # initialize video transform
        self._init_atransform()
        self._init_transform()
        self.transform_224 = Compose([Resize((self.imgSize, self.imgSize), Image.BICUBIC), 
                                      ToTensor()])
        
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.st = 3.5
        self.fi = 6.5
        self.error_data = False

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])      ## where we got 85.2 on flickr
            '''Does order of normalization matters?'''
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.imgSize,self.imgSize), transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.Normalize(mean, std)])      ## setting for the tables on the overleaf now

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.files)  # self.length

    def __getitem__(self, idx):
        f = self.files[idx]
        audio_path = os.path.join(self.data_dir, "train", "audio", f + ".wav")
        video_path = os.path.join(self.data_dir, "train", "video", f + ".mp4")
        class_path = os.path.join(self.data_dir, "train", "classes", f"{f}.txt")
        with open(class_path, "r") as file:
            class_vggsound = file.readlines()[0]

        audio_shift = round(round(random.uniform(self.audio_shift_min, self.audio_shift_max), 2) * self.sample_rate)
        audio_shift = random.choice([audio_shift, -audio_shift])

        # audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        samples, samplerate = sf.read(audio_path)
        if len(samples.shape) > 1:
            if samples.shape[1] == 2:
                samples = samples[:,0]

        video = skvideo.io.vread(video_path)
        video_duration = samples.shape[0] / samplerate
        frame_rate = int(video.shape[0] / video_duration)
        
        num_audio_samples = self.audio_length * samplerate

        if (samples.shape[0] >= num_audio_samples):
            if ((frame_rate * 0.7*self.audio_length) <= (video.shape[0] - (frame_rate * 0.7*self.audio_length))):
                video_frame = random.randint(int((frame_rate * 0.7*self.audio_length) + 1), int((video.shape[0] - (frame_rate * 0.7*self.audio_length)) - 1))

                audio_index = int(video_frame * samplerate / frame_rate)
                audio_start = audio_index - (num_audio_samples // 2)
                audio_slice = slice(audio_start, audio_start + num_audio_samples)

                audio_index_aug = audio_index + audio_shift
                audio_start_aug = audio_index_aug - (num_audio_samples // 2)
                audio_slice_aug = slice(audio_start_aug, audio_start_aug + num_audio_samples)

                if samples[audio_slice].shape[0] == num_audio_samples and samples[audio_slice_aug].shape[0] == num_audio_samples:
                    audio_orig = samples[audio_slice]
                    frequencies, times, spectrogram_orig = signal.spectrogram(audio_orig, samplerate, nperseg=512, noverlap=353)
                    spectrogram_orig = np.log(spectrogram_orig+ 1e-7)
                    spectrogram_orig = self.aid_transform(spectrogram_orig)

                    audio_aug = samples[audio_slice_aug]
                    frequencies, times, spectrogram_aug = signal.spectrogram(audio_aug, samplerate, nperseg=512, noverlap=353)
                    spectrogram_aug = np.log(spectrogram_aug+ 1e-7)
                    spectrogram_aug = self.aid_transform(spectrogram_aug)

                    frame_orig = self.transform_224(Image.fromarray(video[video_frame]))
                    frame_augm = self.img_transform(Image.fromarray(video[video_frame]))

                    return spectrogram_orig, spectrogram_aug, frame_orig, frame_augm, class_vggsound
                else:
                    self.error_data = True

            else:
                video_frame = video.shape[0] // 2
                audio_index = samples.shape[0] // 2
                audio_start = audio_index - (num_audio_samples // 2)
                audio_slice = slice(audio_start, audio_start + num_audio_samples)

                audio_index_aug = audio_index + audio_shift
                audio_start_aug = audio_index_aug - (num_audio_samples // 2)
                audio_slice_aug = slice(audio_start_aug, audio_start_aug + num_audio_samples)

                if samples[audio_slice].shape[0] == num_audio_samples and samples[audio_slice_aug].shape[0] == num_audio_samples:
                    audio_orig = samples[audio_slice]
                    frequencies, times, spectrogram_orig = signal.spectrogram(audio_orig, samplerate, nperseg=512, noverlap=353)
                    spectrogram_orig = np.log(spectrogram_orig+ 1e-7)
                    spectrogram_orig = self.aid_transform(spectrogram_orig)

                    audio_aug = samples[audio_slice_aug]
                    frequencies, times, spectrogram_aug = signal.spectrogram(audio_aug, samplerate, nperseg=512, noverlap=353)
                    spectrogram_aug = np.log(spectrogram_aug+ 1e-7)
                    spectrogram_aug = self.aid_transform(spectrogram_aug)

                    frame_orig = self.transform_224(Image.fromarray(video[video_frame]))
                    frame_augm = self.img_transform(Image.fromarray(video[video_frame]))

                    return spectrogram_orig, spectrogram_aug, frame_orig, frame_augm, class_vggsound
                else:
                    self.error_data = True
        else:
            raise f"Error, video shorter than {self.audio_length} seconds."
        
        if self.error_data == True:
            video_frame = video.shape[0] // 2
            audio_index = samples.shape[0] // 2
            audio_start = audio_index - (num_audio_samples // 2)
            audio_slice = slice(audio_start, audio_start + num_audio_samples)

            audio_index_aug = audio_index + audio_shift
            audio_start_aug = audio_index_aug - (num_audio_samples // 2)
            audio_slice_aug = slice(audio_start_aug, audio_start_aug + num_audio_samples)

            if samples[audio_slice].shape[0] == num_audio_samples and samples[audio_slice_aug].shape[0] == num_audio_samples:
                audio_orig = samples[audio_slice]
                frequencies, times, spectrogram_orig = signal.spectrogram(audio_orig, samplerate, nperseg=512, noverlap=353)
                spectrogram_orig = np.log(spectrogram_orig+ 1e-7)
                spectrogram_orig = self.aid_transform(spectrogram_orig)

                audio_aug = samples[audio_slice_aug]
                frequencies, times, spectrogram_aug = signal.spectrogram(audio_aug, samplerate, nperseg=512, noverlap=353)
                spectrogram_aug = np.log(spectrogram_aug+ 1e-7)
                spectrogram_aug = self.aid_transform(spectrogram_aug)

                frame_orig = self.transform_224(Image.fromarray(video[video_frame]))
                frame_augm = self.img_transform(Image.fromarray(video[video_frame]))

                self.error_data = False

                return spectrogram_orig, spectrogram_aug, frame_orig, frame_augm, class_vggsound
            
            print(f"File: {f} - {num_audio_samples}\n{samples[audio_slice].shape[0]}\n{samples[audio_slice_aug].shape[0]}\n")
            return None


class AudioVisualDataset3(Dataset):

    def __init__(self, 
                data_dir="", 
                image_size=224, 
                audio_length=3, 
                audio_shift_min = 0.1,
                audio_shift_max = 0.5,
                sample_rate=16000, 
                files=[], 
                mode='train', 
                transforms=None
                ):

        self.data_dir = data_dir
        self.imgSize = image_size

        self.audio_shift_min = audio_shift_min
        self.audio_shift_max = audio_shift_max
        
        self.mode = mode
        self.transforms = transforms
        self.files = files

        # initialize video transform
        self._init_atransform()
        self._init_transform()
        self.transform_224 = Compose([Resize((self.imgSize, self.imgSize), Image.BICUBIC), 
                                      ToTensor()])
        
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.st = 3.5
        self.fi = 6.5

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])      ## where we got 85.2 on flickr
            '''Does order of normalization matters?'''
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.imgSize,self.imgSize), transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.Normalize(mean, std)])      ## setting for the tables on the overleaf now

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.files)  # self.length

    def __getitem__(self, idx):
        f = self.files[idx]
        audio_path = os.path.join(self.data_dir, "train", "audio", f + ".wav")
        video_path = os.path.join(self.data_dir, "train", "video", f + ".mp4")
        class_path = os.path.join(self.data_dir, "train", "classes", f"{f}.txt")
        with open(class_path, "r") as file:
            class_vggsound = file.readlines()[0]

        audio_shift = round(round(random.uniform(self.audio_shift_min, self.audio_shift_max), 2) * self.sample_rate)
        audio_shift = random.choice([audio_shift, -audio_shift])

        # audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        samples, samplerate = sf.read(audio_path)
        if len(samples.shape) > 1:
            if samples.shape[1] == 2:
                samples = samples[:,0]

        video = skvideo.io.vread(video_path)
        video_duration = samples.shape[0] / samplerate
        frame_rate = int(video.shape[0] / video_duration)
        
        num_audio_samples = self.audio_length * samplerate

        try:
            if (samples.shape[0] >= num_audio_samples):
            # if (samples.shape[0] >= num_audio_samples) and ((frame_rate * self.audio_length) <= (video.shape[0] - (frame_rate * self.audio_length))):
                video_frame = random.randint((frame_rate * 1.5*self.audio_length) + 1, (video.shape[0] - (frame_rate * 1.5*self.audio_length)) - 1)

                audio_index = int(video_frame * samplerate / frame_rate)
                audio_start = audio_index - (num_audio_samples // 2)
                audio_slice = slice(audio_start, audio_start + num_audio_samples)

                audio_index_aug = audio_index + audio_shift
                audio_start_aug = audio_index_aug - (num_audio_samples // 2)
                audio_slice_aug = slice(audio_start_aug, audio_start_aug + num_audio_samples)

                if samples[audio_slice].shape[0] == num_audio_samples and samples[audio_slice_aug].shape[0] == num_audio_samples:
                    audio_orig = samples[audio_slice]
                    frequencies, times, spectrogram_orig = signal.spectrogram(audio_orig, samplerate, nperseg=512, noverlap=353)
                    spectrogram_orig = np.log(spectrogram_orig+ 1e-7)
                    spectrogram_orig = self.aid_transform(spectrogram_orig)

                    audio_aug = samples[audio_slice_aug]
                    frequencies, times, spectrogram_aug = signal.spectrogram(audio_aug, samplerate, nperseg=512, noverlap=353)
                    spectrogram_aug = np.log(spectrogram_aug+ 1e-7)
                    spectrogram_aug = self.aid_transform(spectrogram_aug)

                    frame_orig = self.transform_224(Image.fromarray(video[video_frame]))
                    frame_augm = self.img_transform(Image.fromarray(video[video_frame]))

                    return spectrogram_orig, spectrogram_aug, frame_orig, frame_augm, class_vggsound
        except:
            if samples.shape[0] >= num_audio_samples:
                video_frame = video.shape[0] // 2

                audio_index = samples.shape[0] // 2
                audio_start = audio_index - (num_audio_samples // 2)
                audio_slice = slice(audio_start, audio_start + num_audio_samples)

                audio_index_aug = audio_index + audio_shift
                audio_start_aug = audio_index_aug - (num_audio_samples // 2)
                audio_slice_aug = slice(audio_start_aug, audio_start_aug + num_audio_samples)
                
                if samples[audio_slice].shape[0] == num_audio_samples and samples[audio_slice_aug].shape[0] == num_audio_samples:
                    audio_orig = samples[audio_slice]
                    frequencies, times, spectrogram_orig = signal.spectrogram(audio_orig, samplerate, nperseg=512, noverlap=353)
                    spectrogram_orig = np.log(spectrogram_orig+ 1e-7)
                    spectrogram_orig = self.aid_transform(spectrogram_orig)

                    audio_aug = samples[audio_slice_aug]
                    frequencies, times, spectrogram_aug = signal.spectrogram(audio_aug, samplerate, nperseg=512, noverlap=353)
                    spectrogram_aug = np.log(spectrogram_aug+ 1e-7)
                    spectrogram_aug = self.aid_transform(spectrogram_aug)

                    frame_orig = self.transform_224(Image.fromarray(video[video_frame]))
                    frame_augm = self.img_transform(Image.fromarray(video[video_frame]))

                    return spectrogram_orig, spectrogram_aug, frame_orig, frame_augm, class_vggsound