import os
import fnmatch
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import xml.etree.ElementTree as ET
from utils_dir.audio_io import load_audio_av, open_audio_av

def load_image(path):
    return Image.open(path).convert('RGB')

def load_spectrogram_fnac(path, path_offscreen, dur=3.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_ss = max(float(audio_dur)/2 - dur/2, 0)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)
    
    audio_ctr_offscreen = open_audio_av(path_offscreen)
    audio_dur_offscreen = audio_ctr_offscreen.streams.audio[0].duration * audio_ctr_offscreen.streams.audio[0].time_base
    audio_ss_offscreen = max(float(audio_dur_offscreen)/2 - dur/2, 0)
    audio_offscreen, samplerate_offscreen = load_audio_av(container=audio_ctr_offscreen, start_time=audio_ss, duration=dur)
    
    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)
    audio_offscreen = np.clip(audio_offscreen, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]
    
    if audio_offscreen.shape[0] < samplerate_offscreen * dur:
        n = int(samplerate_offscreen * dur / audio_offscreen.shape[0]) + 1
        audio_offscreen = np.tile(audio_offscreen, n)
    audio_offscreen = audio_offscreen[:int(samplerate_offscreen * dur)]
    
    audio_noise = np.random.normal(0, 1, len(audio))
    frequencies, times, spectrogram_noise = signal.spectrogram(audio_noise, samplerate, nperseg=512, noverlap=274)
    
    audio_silence = np.zeros_like(audio)
    frequencies, times, spectrogram_silence = signal.spectrogram(audio_silence, samplerate, nperseg=512, noverlap=274)

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
    frequencies, times, spectrogram_offscreen = signal.spectrogram(audio_offscreen, samplerate_offscreen, nperseg=512, noverlap=274)
    
    spectrogram = np.log(spectrogram + 1e-7)
    spectrogram_offscreen = np.log(spectrogram_offscreen + 1e-7)
    
    return spectrogram, spectrogram_noise, spectrogram_silence, spectrogram_offscreen

def load_spectrogram(path, path_offscreen, dur=10.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_ctr_offscreen = open_audio_av(path_offscreen)
    
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_dur_offscreen = audio_ctr_offscreen.streams.audio[0].duration * audio_ctr_offscreen.streams.audio[0].time_base
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=0, duration=dur)  # Get full audio
    audio_offscreen, samplerate_offscreen = load_audio_av(container=audio_ctr_offscreen, start_time=0, duration=dur)  # Get full audio

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)
    audio_offscreen = np.clip(audio_offscreen, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]
    
    if audio_offscreen.shape[0] < samplerate_offscreen * dur:
        n = int(samplerate_offscreen * dur / audio.shape[0]) + 1
        audio_offscreen = np.tile(audio_offscreen, n)
    audio_offscreen = audio_offscreen[:int(samplerate_offscreen * dur)]
    
    audio_noise = np.random.normal(0, 1, len(audio))
    frequencies, times, spectrogram_noise = signal.spectrogram(audio_noise, samplerate, nperseg=512, noverlap=274)
    
    audio_silence = np.zeros_like(audio)
    frequencies, times, spectrogram_silence = signal.spectrogram(audio_silence, samplerate, nperseg=512, noverlap=274)
    
    audio_center = int(float(audio_dur) / 2 * samplerate)
    shift_index = int(samplerate * dur / 2) - audio_center
    audio = np.roll(audio, shift_index)

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
    frequencies, times, spectrogram_offscreen = signal.spectrogram(audio_offscreen, samplerate_offscreen, nperseg=512, noverlap=274)
    
    spectrogram = np.log(spectrogram + 1e-7)
    spectrogram_offscreen = np.log(spectrogram_offscreen + 1e-7)
    
    return spectrogram, spectrogram_noise, spectrogram_silence, spectrogram_offscreen

def load_spectrogram_rcgrad(path, path_offscreen, dur=10.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_ctr_offscreen = open_audio_av(path_offscreen)
    
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_dur_offscreen = audio_ctr_offscreen.streams.audio[0].duration * audio_ctr_offscreen.streams.audio[0].time_base
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=0, duration=dur)  # Get full audio
    audio_offscreen, samplerate_offscreen = load_audio_av(container=audio_ctr_offscreen, start_time=0, duration=dur)  # Get full audio

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)
    audio_offscreen = np.clip(audio_offscreen, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]
    
    if audio_offscreen.shape[0] < samplerate_offscreen * dur:
        n = int(samplerate_offscreen * dur / audio.shape[0]) + 1
        audio_offscreen = np.tile(audio_offscreen, n)
    audio_offscreen = audio_offscreen[:int(samplerate_offscreen * dur)]
    
    audio_noise = np.random.normal(0, 1, len(audio))
    
    audio_silence = np.zeros_like(audio)
    
    audio_center = int(float(audio_dur) / 2 * samplerate)
    shift_index = int(samplerate * dur / 2) - audio_center
    audio = np.roll(audio, shift_index)
    
    return audio, audio_noise, audio_silence, audio_offscreen


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

    else:
        gt_map[gt_map > 0] = 1

    return gt_map

class AudioVisualDataset(Dataset):
    def __init__(self, args, image_files, audio_files, image_path, audio_path, audio_dur=3., image_transform=None, audio_transform=None, all_bboxes=None, bbox_format='flickr', model_name='ezvsl', img_selection=None, dual=False):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path
        self.audio_dur = audio_dur

        self.audio_files = np.array(audio_files)
        self.image_files = np.array(image_files)
        self.all_bboxes = all_bboxes
        
        self.bbox_format = bbox_format

        self.image_transform = image_transform
        self.audio_transform = audio_transform

        self.img_selection = img_selection
        self.model_name = model_name
        self.args = args
        
        if args.testset == "vggss":
            new_classes_json = "metadata/vggss_class_files_dict.json"

            with open(new_classes_json, 'r') as file:
                self.new_classes_data = json.load(file)
        elif args.testset == "is3":
            new_classes_json = "metadata/IS3_class_files_dict.json"

            with open(new_classes_json, 'r') as file:
                self.new_classes_data = json.load(file)
    
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
        
    def getitem(self, idx):
        file = self.image_files[idx]
        file_root = file.split(".")[0]

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
        
        random_class = random.choice(sorted(new_classes))
        random_file = random.choice(self.new_classes_data[random_class])
        file_id = file.split('.')[0]
        if self.args.testset == 'vggss':
            file_id = file.split('/')[0]

        # Image
        img_fn = os.path.join(self.image_path, self.image_files[idx])
        if self.img_selection == 'random':
            item = random.choice(os.listdir(os.path.join(self.image_path, file_id)))
            img_fn = os.path.join(self.image_path, file_id, item)
        if self.img_selection == 'center':
            items = os.listdir(os.path.join(self.image_path, file_id))
            item = items[int(len(items)/2)]
            img_fn = os.path.join(self.image_path, file_id, item)
        if self.img_selection == 'first':
            items = os.listdir(os.path.join(self.image_path, file_id))
            item = items[0]
            img_fn = os.path.join(self.image_path, file_id, item)        

        frame = self.image_transform(load_image(img_fn))

        # Audio
        audio_fn = os.path.join(self.audio_path, self.audio_files[idx])
        audio_fn_offscreen = os.path.join(self.audio_path, f"{random_file}.wav")
        
        if self.model_name == 'fnac':
            spectrogram, spectrogram_noise, spectrogram_silence, spectrogram_offscreen = load_spectrogram_fnac(audio_fn, audio_fn_offscreen)
            spectrogram = self.audio_transform(spectrogram)
            spectrogram_noise = self.audio_transform(spectrogram_noise)
            spectrogram_silence = self.audio_transform(spectrogram_silence)
            spectrogram_offscreen = self.audio_transform(spectrogram_offscreen)
        if self.args.pth_name == 'rcgrad':
            spectrogram, spectrogram_noise, spectrogram_silence, spectrogram_offscreen = load_spectrogram_rcgrad(audio_fn, audio_fn_offscreen)
        else:
            spectrogram, spectrogram_noise, spectrogram_silence, spectrogram_offscreen = load_spectrogram(audio_fn, audio_fn_offscreen, dur = self.audio_dur)
            spectrogram = self.audio_transform(spectrogram)
            spectrogram_noise = self.audio_transform(spectrogram_noise)
            spectrogram_silence = self.audio_transform(spectrogram_silence)
            spectrogram_offscreen = self.audio_transform(spectrogram_offscreen)
            
        bboxes = {}
        if self.all_bboxes is not None:
            bboxes['gt_map'] = bbox2gtmap(self.all_bboxes[file_id.split(".")[0]], self.bbox_format)

        return frame, spectrogram, spectrogram_noise, spectrogram_silence, spectrogram_offscreen, bboxes, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.getitem(idx)

def get_test_dataset(args):
    if args.testset == 'vggss':
        audio_path = f'{args.data_dir}/audio/'
        image_path = f'{args.data_dir}/frames/'
    elif args.testset == 'is3':
        audio_path = f'{args.data_dir}/audio_wav'
        image_path = f'{args.data_dir}/images'
    
    if args.testset == 'vggss':
        testcsv = 'metadata/vggss_annotations.json'
    elif args.testset == 'is3':
        testcsv = 'metadata/IS3_annotation.json'
    elif args.testset == 'vggss_heard':
        testcsv = 'metadata/vggss_heard_test.csv'
    elif args.testset == 'vggss_unheard':
        testcsv = 'metadata/vggss_unheard_test.csv'
    elif args.testset == 'vposs':
        testcsv = 'metadata/vpo_ss_bbox.json'
    elif args.testset == 'vpoms':
        testcsv = 'metadata/vpo_ms_bbox.json'
    elif args.testset == 'ms3':
        testcsv = 'metadata/ms3_box.json'
    elif args.testset == 's4':
        testcsv = 'metadata/s4_box.json'
    else:
        raise NotImplementedError
    bbox_format = {'flickr': 'flickr',
                   'vggss': 'vggss',
                   'vggss_heard': 'vggss',
                   'vggss_unheard': 'vggss',
                   'is3':'is3',
                   'ms3':'ms3',
                   's4':'s4',
                   'vpoms':'vpoms',
                   'vposs':'vposs'
                  }[args.testset]
    audio_length = 10.0

    #  Retrieve list of audio and video files
    if 'json' in testcsv:
        with open(testcsv) as fi:
            jsonfile = json.load(fi)
        
        all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)
        
        if args.testset == 'ms3':
            audio_length = 5.0
            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-2])

            audio_files = [fn['audio'].split('/')[-1][:-4] for fn in jsonfile]
            image_files = ['/'.join(fn['image'].split('/')[-2:]) for fn in jsonfile]
            
        elif args.testset == 's4':
            audio_length = 5.0
            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-2])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-3])

            audio_files = ['/'.join(fn['audio'].split('/')[-2:])[:-4] for fn in jsonfile]
            image_files = ['/'.join(fn['image'].split('/')[-3:]) for fn in jsonfile]
            
        elif args.testset == "vggss":
            files = [fn['file'].split('/')[-1] for fn in jsonfile]
            audio_files = [fn+".wav" for fn in files]
            image_files = [fn+".jpg" for fn in files]

            audio_path = f'{args.data_dir}/audio'
            image_path = f'{args.data_dir}/frames'
        
        elif args.testset == "is3":
            audio_files = [fn['audio'].split('/')[-1] for fn in jsonfile]
            image_files = [fn['image'].split('/')[-1] for fn in jsonfile]

            audio_path = f'{args.data_dir}/audio_wav'
            image_path = f'{args.data_dir}/images'
            
        else: 
            audio_files = [fn['audio'].split('/')[-1] for fn in jsonfile]
            image_files = [fn['image'].split('/')[-1] for fn in jsonfile]

            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-1])
    else:
        testset = set([item[0] for item in csv.reader(open(testcsv))])

        # Intersect with available files
        audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
        image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
        avail_files = audio_files.intersection(image_files)
        testset = testset.intersection(avail_files)

        testset = sorted(list(testset))
        
        if args.testset == 'flickr':
            image_files = [dt+'.jpg' for dt in testset]
            audio_files = [dt for dt in testset]
        elif args.testset == 'vggss':
            image_files = [dt+'/image_050.jpg' for dt in testset]
            audio_files = [dt for dt in testset]
            
    if len(args.files_list) > 0:
        audio_files = [fn+".wav" for fn in args.files_list]
        image_files = [fn+".jpg" for fn in args.files_list]

    # Bounding boxes
    all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)

    # Transforms
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    spec_size = (257, 200)
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.CenterCrop(spec_size),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        args=args,
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=audio_length,
        image_transform=image_transform,
        audio_transform=audio_transform,
        all_bboxes=all_bboxes,
        bbox_format=bbox_format
    )

def get_test_dataset_fnac(args):
    if args.testset == 'vggss':
        audio_path = f'{args.data_dir}/audio/'
        image_path = f'{args.data_dir}/frames/'
    elif args.testset == 'is3':
        audio_path = f'{args.data_dir}/audio_wav'
        image_path = f'{args.data_dir}/images'
    
    if args.testset == 'vggss':
        testcsv = 'metadata/vggss_annotations.json'
    elif args.testset == 'is3':
        testcsv = 'metadata/IS3_annotation.json'
    elif args.testset == 'vggss_heard':
        testcsv = 'metadata/vggss_heard_test.csv'
    elif args.testset == 'vggss_unheard':
        testcsv = 'metadata/vggss_unheard_test.csv'
    elif args.testset == 'vposs':
        testcsv = 'metadata/vpo_ss_bbox.json'
    elif args.testset == 'vpoms':
        testcsv = 'metadata/vpo_ms_bbox.json'
    elif args.testset == 'ms3':
        testcsv = 'metadata/ms3_box.json'
    elif args.testset == 's4':
        testcsv = 'metadata/s4_box.json'
    else:
        raise NotImplementedError
    bbox_format = {'flickr': 'flickr',
                   'vggss': 'vggss',
                   'vggss_heard': 'vggss',
                   'vggss_unheard': 'vggss',
                   'is3':'is3',
                   'ms3':'ms3',
                   's4':'s4',
                   'vpoms':'vpoms',
                   'vposs':'vposs'
                  }[args.testset]
    
    if 'json' in testcsv:
        with open(testcsv) as fi:
            jsonfile = json.load(fi)
        
        all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)
        
        if args.testset == 'ms3':
            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-2])

            audio_files = [fn['audio'].split('/')[-1][:-4] for fn in jsonfile]
            image_files = ['/'.join(fn['image'].split('/')[-2:]) for fn in jsonfile]
            
        elif args.testset == 's4':
            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-2])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-3])

            audio_files = ['/'.join(fn['audio'].split('/')[-2:])[:-4] for fn in jsonfile]
            image_files = ['/'.join(fn['image'].split('/')[-3:]) for fn in jsonfile]
            
        elif args.testset == "vggss":
            files = [fn['file'].split('/')[-1] for fn in jsonfile]
            audio_files = [fn+".wav" for fn in files]
            image_files = [fn+".jpg" for fn in files]

            audio_path = f'{args.data_dir}/audio'
            image_path = f'{args.data_dir}/frames'
        
        elif args.testset == "is3":
            audio_files = [fn['audio'].split('/')[-1] for fn in jsonfile]
            image_files = [fn['image'].split('/')[-1] for fn in jsonfile]

            audio_path = f'{args.data_dir}/audio_wav'
            image_path = f'{args.data_dir}/images'
            
        else: 
            audio_files = [fn['audio'].split('/')[-1] for fn in jsonfile]
            image_files = [fn['image'].split('/')[-1] for fn in jsonfile]

            audio_path = '/'.join(jsonfile[0]['audio'].split('/')[:-1])
            image_path = '/'.join(jsonfile[0]['image'].split('/')[:-1])
    else:
        testset = set([item[0] for item in csv.reader(open(testcsv))])

        # Intersect with available files
        audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
        image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
        avail_files = audio_files.intersection(image_files)
        testset = testset.intersection(avail_files)

        testset = sorted(list(testset))
        
        if args.testset == 'flickr':
            image_files = [dt+'.jpg' for dt in testset]
            audio_files = [dt for dt in testset]
        elif args.testset == 'vggss':
            image_files = [dt+'/image_050.jpg' for dt in testset]
            audio_files = [dt for dt in testset] 

    if len(args.files_list) > 0:
        audio_files = [fn+".wav" for fn in args.files_list]
        image_files = [fn+".jpg" for fn in args.files_list]
    
    # Bounding boxes
    all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)

    # Transforms
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        args=args,
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        all_bboxes=all_bboxes,
        bbox_format=bbox_format,
        model_name='fnac'
    )

def inverse_normalize(tensor):
    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor


