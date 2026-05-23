import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model.resnet50 import resnet50
from model.dino_classifier import dino_classifier
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np
import argparse
from util import data_augment, GetPathsAndLabels
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score

# Set Seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################################################################################################################
class EvaluateDataset(Dataset):
    def __init__(self, root_list, model='cnn', crop_size=224, fake_equal_real=True, data_augment_params=None):
        self.root_list = root_list
        self.real_image_paths = []
        self.real_labels = []
        self.fake_image_paths = []
        self.fake_labels = []
        self.image_paths = []
        self.labels = []
        self.fake_equal_real = fake_equal_real
        self.crop_size = crop_size
        self.evaluate_size = None
        self.max_size = None
        self.data_augment_params = data_augment_params
        
        if model == 'clip':
            mean=[0.48145466, 0.4578275, 0.40821073]
            std=[0.26862954, 0.26130258, 0.27577711]
        else:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]

        # transforme for patch
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std),
                        ])

        def root_warning(root):
            if not os.path.exists(root):
                raise ValueError(f"Root directory {root} does not exist.")
            if not os.path.isdir(root):
                raise ValueError(f"Root directory {root} is not a directory.")
        
        is_realset_loaded = False
        is_fakeset_loaded = False
        # load real images
        for i, root in enumerate(self.root_list):
            if 'real' in root:
                root_warning(root)
                i_image_paths, i_labels = GetPathsAndLabels(root, max_size=self.max_size)
                self.real_image_paths.extend(i_image_paths)
                self.real_labels.extend(i_labels)
                is_realset_loaded = True
        # load fake images
        for i, root in enumerate(self.root_list):
            if 'real' not in root:
                root_warning(root)
                i_image_paths, i_labels = GetPathsAndLabels(root, max_size=self.max_size)
                self.fake_image_paths.extend(i_image_paths)
                self.fake_labels.extend(i_labels)
                is_fakeset_loaded = True
        
        if self.fake_equal_real:
            if is_realset_loaded and is_fakeset_loaded:
                image_size = min(len(self.real_image_paths), len(self.fake_image_paths))
            elif not is_realset_loaded and is_fakeset_loaded:
                image_size = len(self.fake_image_paths)
            elif is_realset_loaded and not is_fakeset_loaded:
                image_size = len(self.real_image_paths)
        
            if is_realset_loaded:
                combined = list(zip(self.real_image_paths, self.real_labels))
                sampled = random.sample(combined, image_size)
                self.real_image_paths, self.real_labels = zip(*sampled)
                self.image_paths.extend(self.real_image_paths)
                self.labels.extend(self.real_labels)
            
            if is_fakeset_loaded:
                combined = list(zip(self.fake_image_paths, self.fake_labels))
                sampled = random.sample(combined, image_size)
                self.fake_image_paths, self.fake_labels = zip(*sampled)
                self.image_paths.extend(self.fake_image_paths)
                self.labels.extend(self.fake_labels)
                
        else:
            self.image_paths.extend(self.real_image_paths)
            self.labels.extend(self.real_labels)
            self.image_paths.extend(self.fake_image_paths)
            self.labels.extend(self.fake_labels)
        
        if self.evaluate_size is not None:
            if len(self.image_paths) > self.evaluate_size:
                combined = list(zip(self.image_paths, self.labels))
                sampled = random.sample(combined, self.evaluate_size)
                self.image_paths, self.labels = zip(*sampled)
        
            
    def __len__(self):
        return len(self.image_paths)
    
    def print_image_size(self):
        return len(self.real_image_paths), len(self.fake_image_paths)

    # def generate_sliding_patches(self, image):
    #     w, h = image.size
    #     patch_size = self.crop_size
    #     patch_stride = self.crop_size
    #     patches = []

    #     # calculate patch positions
    #     x_steps = []
    #     x = 0
    #     while x + patch_size <= w:
    #         x_steps.append(x)
    #         x += patch_stride
    #     # cover edge case
    #     if x_steps and x_steps[-1] + patch_size < w:
    #         x_steps.append(w - patch_size)
        
    #     y_steps = []
    #     y = 0
    #     while y + patch_size <= h:
    #         y_steps.append(y)
    #         y += patch_stride
    #     if y_steps and y_steps[-1] + patch_size < h:
    #         y_steps.append(h - patch_size)
        
    #     # crop image to patches
    #     for x in x_steps:
    #         for y in y_steps:
    #             patch = image.crop((x, y, x + patch_size, y + patch_size))
    #             patches.append(patch)
    #     return patches


    def generate_sliding_patches(self, image):
        """
        OPTIMIZED VERSION
        image: PIL Image (after data_augment)
        return: Tensor [num_patches, 3, crop_size, crop_size]
        """
        img_tensor = self.transform(image)  # [3, H, W]
        C, H, W = img_tensor.shape
        ps = self.crop_size
        stride = self.crop_size

        if H < ps or W < ps:
            return torch.zeros(0, 3, ps, ps)

        # unfold on tensor (much faster than PIL crop)
        patches = img_tensor.unfold(1, ps, stride).unfold(2, ps, stride)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(-1, C, ps, ps)

        return patches
    
    def __getitem__(self, idx):
        # get image_path
        image_path = self.image_paths[idx]
        # get image_tensor
        try:
            image = Image.open(image_path).convert('RGB')
            image = data_augment(image, data_augment_params=self.data_augment_params, mode='fix', crop_size=self.crop_size)
            # patches = self.generate_sliding_patches(image)
            # patch_tensors = [self.transform(patch) for patch in patches]
            # patch_tensors = torch.stack(patch_tensors)  # [num_patches, C, H, W]
            # num_patches = len(patch_tensors)  # patch number for one image

            patch_tensors = self.generate_sliding_patches(image)
            num_patches = patch_tensors.shape[0]

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            patch_tensors = torch.zeros(0, 3, self.crop_size, self.crop_size)
            num_patches = 0
        
        # get label
        label = self.labels[idx]
        return image_path, patch_tensors, label, num_patches

# custom collate_fn
def collate_fn(batch):
    """
    batch: [(image_path, patch_tensors, label, num_patches), ...]
    """
    image_paths = []
    all_patches = []
    labels = []
    num_patches_list = []
    
    for item in batch:
        image_path, patch_tensors, label, num_patches = item
        image_paths.append(image_path)
        if num_patches > 0:
            all_patches.append(patch_tensors)
        labels.append(label)
        num_patches_list.append(num_patches)
    # concat patch tensors to one big tensor
    if all_patches:
        all_patches = torch.cat(all_patches, dim=0)
    else:
        all_patches = torch.zeros(0, 3, batch[0][1].shape[2], batch[0][1].shape[3])
    
    return {
        'image_paths': image_paths,
        'all_patches': all_patches,
        'labels': torch.tensor(labels),
        'num_patches_list': num_patches_list
    }

###########################################################################################################################    
# evaluate
def evaluate(model, dataloader):
    print(f'real images: {dataset.print_image_size()[0]},  fake images: {dataset.print_image_size()[1]}')
    model.eval()
    y_true, y_pred = [], []
    cnt = 0
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_data in dataloader:
            all_patches = batch_data['all_patches'].to(device)  # [total_patches, C, H, W]
            labels = batch_data['labels']  # [batch_size]
            num_patches_list = batch_data['num_patches_list']  # patch number list for images
            
            if all_patches.shape[0] > 0:
                logits = model(all_patches)  # [total_patches, 1]
                logits = logits.squeeze(-1)  # [total_patches]
            else:
                logits = torch.zeros(sum(num_patches_list)).to(device)
            
            # calculate prob for every image
            batch_preds = []
            start_idx = 0
            for num_p in num_patches_list:
                if num_p == 0:
                    avg_logit = torch.tensor(0.0).to(device)
                else:
                    sample_logits = logits[start_idx:start_idx + num_p]
                    avg_logit = sample_logits.mean()
                    start_idx += num_p
                batch_preds.append(avg_logit.sigmoid().cpu().item())
            
            y_pred.extend(batch_preds)
            y_true.extend(labels.cpu().numpy().tolist())
            
            cnt += 1
            print(f'Processed : {cnt}/{total_batches} | Current batch size: {len(labels)}', end='\r')
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) == 0:
        raise ValueError("No valid data for evaluation!")
    
    fake_mask = y_true == 1
    real_mask = y_true == 0
    
    fake_acc = accuracy_score(y_true[fake_mask], y_pred[fake_mask] > 0.5) if np.any(fake_mask) else 0.0
    real_acc = accuracy_score(y_true[real_mask], y_pred[real_mask] > 0.5) if np.any(real_mask) else 0.0
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    avg_logit = np.mean([np.log(p/(1-p)) if 0 < p < 1 else 0 for p in y_pred])
    
    return acc, fake_acc, real_acc, ap, y_true, y_pred, avg_logit


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='parser for evaluation')
    parser.add_argument('--root_list', nargs='+',
                        default=[
                            '/data_center/data2/dataset/detection_dataset/synthbuster/synthbuster/dalle2',
                        ])
    parser.add_argument('--detector_type', type=str, default='cnn', help='cnn->imagenet normalization')
    parser.add_argument('--fake_equal_real', action='store_false')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--crop_size', type=int, default=224, help='Patch crop size')

    parser.add_argument("--blur_prob", default=0.0, type=float)
    parser.add_argument("--blur_value", default=0.0, type=float)
    parser.add_argument("--jpeg_prob", default=0.0, type=float)
    parser.add_argument("--jpeg_value", default=100, type=int)
    parser.add_argument("--noise_prob", default=0.0, type=float)
    parser.add_argument("--noise_value", default=0.0, type=float)
    parser.add_argument("--scale_prob", default=0.0, type=float)
    parser.add_argument("--scale_value", default=1.0, type=float)

    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--ckpt", type=str)

    args = parser.parse_args()

    data_augment_params = {
        # blur
        'blur_prob':args.blur_prob,
        'blur_sig_min':0.0,
        'blur_sig_max':3.0,
        'blur_sig':args.blur_value,
        # jpeg
        'jpeg_prob':args.jpeg_prob,
        'jpeg_quality_min':60,
        'jpeg_quality_max':100, 
        'jpeg_quality':args.jpeg_value,
        # cutout
        'cutout_prob':0.0,
        'cutout_ratio_min': 0.1,
        'cutout_ratio_max': 0.5,
        # noise
        'noise_prob':args.noise_prob,
        'noise_std_min':0.0,
        'noise_std_max':50.0,
        'noise_std':args.noise_value,
        # resize
        'resize_prob':args.scale_prob,
        'resize_scale_min':0.5,
        'resize_scale_max':2.0,
        'resize_scale':args.scale_value,
        # pad mode
        'pad_mode':'pad',
    }
    
    # initial dataset and dataloader
    root_list = args.root_list
    detector_type = args.detector_type
    dataset = EvaluateDataset(
        root_list=root_list, 
        model=detector_type, 
        crop_size=args.crop_size, 
        fake_equal_real=args.fake_equal_real,
        data_augment_params=data_augment_params,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.batch_size/2),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        # persistent_workers=True
    )
    
    # load detector 
    # ResNet 
    detector_name = 'ResNet50'
    detector = resnet50(num_classes=1, classifier_type='linear')
    checkpoint_path = args.ckpt
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    detector.load_state_dict(checkpoint['model_state_dict'],strict=True)

    detector = detector.to(device)
    detector.eval()
    
    acc, fake_acc, real_acc, ap, y_true, y_pred, logit = evaluate(detector, dataloader)
    

    print('\n' + '='*50)
    print(f'detector:{detector_name} | Testset: {root_list}')
    print(f'Batch size: {args.batch_size} | Crop size: {args.crop_size}')
    print(f'Fake Accuracy: {fake_acc:.4f}')
    print(f'Real Accuracy: {real_acc:.4f}')
    print(f'Overall Accuracy: {acc:.4f}')
    print(f'Average Precision: {ap:.4f}')
    print(f'Average Logit: {logit:.4f}')
    print('='*50)

    if args.output_path:
        with open(args.output_path, 'a') as f:
            f.write('\n' + '='*50 + '\n')
            f.write(f'detector:{detector_name} | Testset: {", ".join(root_list)}\n')
            f.write(f'ckpt:{args.ckpt}')
            f.write(f'Crop size: {args.crop_size}\n')
            f.write(f'Fake Accuracy: {fake_acc*100:.2f}\n')
            f.write(f'Real Accuracy: {real_acc*100:.2f}\n')
            f.write(f'Overall Accuracy: {acc*100:.2f}\n')
            f.write(f'Average Precision: {ap*100:.2f}\n')
            f.write('='*50 + '\n')




















