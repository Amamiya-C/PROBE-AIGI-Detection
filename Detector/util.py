import torch 
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
import torch.nn.functional as F
from PIL import Image
import os 
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import random
from io import BytesIO

def GetPathsAndLabels(image_root, max_size=None, filter=None):
    img_suffixes = ('.jpg', '.png', '.jpeg')
    image_paths = [os.path.join(image_root, img) for img in os.listdir(image_root) if img.lower().endswith(img_suffixes)]
    
    # use filter to exclude certain images
    if filter is not None:
        image_paths = [img_path for img_path in image_paths if all(f not in img_path for f in filter)]
    
    if max_size is not None and max_size < len(image_paths):
        image_paths = random.sample(image_paths, max_size)
        
    if 'real' in image_root:
        labels = [0] * len(image_paths)  # Real images labeled as 0
    else:
        labels = [1] * len(image_paths)
    
    return image_paths, labels


def float_int_float_simulate(image_tensor):
    # [0,1] -> [0,255]
    image_tensor = image_tensor * 255.0
    # float -> int -> float
    image_tensor += image_tensor.round().detach() - image_tensor.detach()
    # [0,255] -> [0,1]
    image_tensor = image_tensor / 255.0
    return image_tensor


def imagenet_process(image_tensor, resize_size=(256,256), crop_size=(224, 224), crop_mode='center'):
    image_tensor = float_int_float_simulate(image_tensor)
    
    resized_tensor = F.interpolate(image_tensor, size=resize_size, mode="bilinear", align_corners=False, antialias=True)
    resized_tensor = float_int_float_simulate(resized_tensor)
    
    if crop_mode == 'center':
        top = (resize_size[0] - crop_size[0]) // 2
        left = (resize_size[1] - crop_size[1]) // 2
        cropped_tensor = resized_tensor[:, :, top:top+224, left:left+224]
        cropped_tensor = float_int_float_simulate(cropped_tensor)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)
    normalized_tensor = (cropped_tensor - mean) / std
    
    return normalized_tensor


def process_wo_resize(image_tensor, crop_size=(224, 224), crop_mode='center', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # image_tensor : [B, C, H, W]
    image_tensor = float_int_float_simulate(image_tensor)
    
    if crop_mode == 'center':
        height, width = image_tensor.shape[2], image_tensor.shape[3]
        top = (height - crop_size[0]) // 2
        left = (width - crop_size[1]) // 2
        cropped_tensor = image_tensor[:, :, top:top+224, left:left+224]
    
    
    mean = torch.tensor(mean, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=image_tensor.device).view(1, 3, 1, 1)
    normalized_tensor = (cropped_tensor - mean) / std
    return normalized_tensor


def decode_image(latents, pipe):
    latents = 1 / pipe.vae.config.scaling_factor * latents
    image_tensor = pipe.vae.decode(latents, return_dict=False)[0]
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    return image_tensor


def sample_continuous(region):
    if len(region) == 1:
        return region[0]
    if len(region) == 2:
        region_size = region[1] - region[0]
    return random.random() * region_size + region[0]


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
    

# def jpeg_compress(img, jpeg_quality):
#     out = BytesIO()
#     # img = Image.fromarray(img)
#     img.save(out, format='jpeg', quality=jpeg_quality)
#     img = Image.open(out)
#     # load from memory before ByteIO closes
#     img = np.array(img)
#     out.close()
#     img = Image.fromarray(img)
#     return img

def jpeg_compress(img, jpeg_quality):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    out = BytesIO()
    img.save(out, format='jpeg', quality=jpeg_quality)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img


def pad_image(img, size=224, mode='pad'):
    width, height = img.size
    
    if width>=size and height>=size:
        return img

    if mode == 'repeat':
        repeat_times_width = max(1, (size + width - 1) // width)  
        repeat_times_height = max(1, (size + height - 1) // height) 
        new_width = width * repeat_times_width
        new_height = height * repeat_times_height
        new_img = Image.new('RGB', (new_width, new_height))

        for i in range(repeat_times_height):
            for j in range(repeat_times_width):
                new_img.paste(img, (j * width, i * height))
        return new_img
    if mode == 'pad':
        new_img = Image.new('RGB', (size, size), (0,0,0))
        paste_x = (size - width) // 2
        paste_y = (size - height) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img


def cutout(img, max_ratio_min=0.1, mask_ratio_max=0.5, mask_value=0):
    height, width, _ = img.shape
    # random mask_size
    mask_size = random.randint(int(max_ratio_min*min(height, width)), int(mask_ratio_max*min(height, width)) )
    # random mask_position
    top = random.randint(0, height - mask_size)
    left = random.randint(0, width - mask_size)
    img[top:top + mask_size, left:left + mask_size] = mask_value
    return img


def resize(img, scale):
    width, height = img.size
    new_w = int(round(width*scale))
    new_h = int(round(height*scale))
    resample_mode = Image.Resampling.BILINEAR
    return img.resize((new_w, new_h), resample=resample_mode)


def gaussian_noise(img, std=0):
    noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img
    
    
def data_augment(img, data_augment_params=None, mode='random', crop_size=224):
    
    if data_augment_params is None:
        data_augment_params = {
        # blur
        'blur_prob':0.0,
        'blur_sig_min':0.0,
        'blur_sig_max':3.0,
        'blur_sig':1.0,
        # jpeg
        'jpeg_prob':0.0,
        'jpeg_quality_min':60,
        'jpeg_quality_max':100, 
        'jpeg_quality':95,
        # cutout
        'cutout_prob':0.0,
        'cutout_ratio_min': 0.1,
        'cutout_ratio_max': 0.5,
        # noise
        'noise_prob':0.0,
        'noise_std_min':0.0,
        'noise_std_max':50.0,
        'noise_std':0.0,
        # resize
        'resize_prob':0.0,
        'resize_scale_min':0.5,
        'resize_scale_max':2.0,
        'resize_scale':1.0,
        # pad mode
        'pad_mode':'pad',
    }

    if random.random() < data_augment_params['resize_prob']:
        if mode != 'random':
            scale = data_augment_params['resize_scale']
        else:
            scale = sample_continuous([data_augment_params['resize_scale_min'], data_augment_params['resize_scale_max']])
        img = resize(img, scale)
    
    img = np.array(img)
    
    # random blur
    if random.random() < data_augment_params['blur_prob']:
        if mode != 'random':
            sig = data_augment_params['blur_sig']
        else:
            sig = sample_continuous([data_augment_params['blur_sig_min'], data_augment_params['blur_sig_max']])
        gaussian_blur(img, sig)
        
    # random gaussian noise
    if random.random() < data_augment_params['noise_prob']:
        if mode != 'random':
            std = data_augment_params['noise_std']
        else:
            std = random.randint(data_augment_params['noise_std_min'], data_augment_params['noise_std_max'])
        img = gaussian_noise(img, std)
    
    # random cut-out
    if random.random() < data_augment_params['cutout_prob']:
        img = cutout(img, max_ratio_min=data_augment_params['cutout_ratio_min'], mask_ratio_max=data_augment_params['cutout_ratio_max'])
    
    # random jpeg compress
    if random.random() < data_augment_params['jpeg_prob']:
        if mode != 'random':
            jpeg_quality = data_augment_params['jpeg_quality']
        else:
            jpeg_quality = random.randint(data_augment_params['jpeg_quality_min'], data_augment_params['jpeg_quality_max'])
        img = jpeg_compress(img, jpeg_quality)

    img = Image.fromarray(img) 
    img = pad_image(img, size=crop_size, mode=data_augment_params['pad_mode'])

    return img


# def get_image_per_step(pipe, step, timestep, callback_kwargs):
#     start_ratio = 0
#     end_ratio = 1
    
#     if int(pipe.num_timesteps * end_ratio) >= step >= int(pipe.num_timesteps * start_ratio):
#         # get latents
#         latents = callback_kwargs["latents"].detach()
#         # forward
#         image = decode_image(latents, pipe)
#         # image = image.clamp(0, 255).squeeze(0)
#         # image = image.permute(1, 2, 0)
#         # image_array = image.byte().cpu().numpy()
#         # image_pil = Image.fromarray(image_array)
#         # image_pil.save(f"/data3/czj21164/proj/proposal/data/image_per_step/{step}.png")
#         image = image.detach().cpu()
#         image = transforms.ToPILImage()(image.squeeze())
#         image.save(f"/data3/czj21164/proj/proposal/data/image_per_step/{step}.png")
#     return callback_kwargs