import random
import numpy as np
from monai.transforms import Compose,EnsureChannelFirst,RandSpatialCrop,RandRotate90
from monai.data import ImageDataset,DataLoader
import torch
from nets.unet import res_unet
from itertools import combinations
import nibabel as nib
def map_channels(dataset_channels, total_modalities):
        channel_map = []
        for channel in dataset_channels:
            for index, modality in enumerate(total_modalities):
                if channel == modality:
                    channel_map.append(index)
        return channel_map

def rand_set_channels_to_zero(dataset_modalities: list, batch_img_data: torch.Tensor):
    modalities_remaining=[]
    for i in range (batch_img_data.shape[0]):
        number_of_dropped_modalities = np.random.randint(0,len(dataset_modalities))
        modalities_dropped = random.sample(list(np.arange(len(dataset_modalities))), number_of_dropped_modalities)
        modalities_dropped.sort()
        batch_img_data[i,modalities_dropped,:,:,:] = 0.
        modalities_remaining.append(list(set(np.arange(len(dataset_modalities))) - set(modalities_dropped)))        
    return modalities_remaining, batch_img_data

def create_dataloader(val_size: int, images, segs, workers, train_batch_size: int, total_train_data_size: int, current_train_data_size: int, cropped_input_size:list, limited_data = False, limited_data_size = 10):
    div = total_train_data_size//current_train_data_size
    rem = total_train_data_size%current_train_data_size
    train_images = images[:-val_size]
    train_images = train_images * div + train_images[:rem]
    train_segs = segs[:-val_size]
    train_segs = train_segs * div + train_segs[:rem]
    # image augmentation through spatial cropping to size and by randomly rotating
    train_imtrans = Compose(
        [
            EnsureChannelFirst(strict_check=True),
            RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            # RandCropByPosNegLabel((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),label=train_segs),
            RandRotate90(prob=0.1, spatial_axes=(0, 2)),
        ]
    )
    val_imtrans = Compose([EnsureChannelFirst()])
    val_segtrans = Compose([EnsureChannelFirst()])
    # create a training data loader
    train_ds = ImageDataset(train_images, train_segs, transform=train_imtrans, seg_transform=train_imtrans)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=workers, pin_memory=0)
    # create a validation data loader
    val_ds = ImageDataset(images[-val_size:], segs[-val_size:], transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)
    return train_loader, val_loader


def create_net(model_file_path,model_net_type,model_modalities_trained_on, device,cuda_id):
    if model_net_type == "UNet":    
      model = res_unet(in_channels=model_modalities_trained_on,
                          out_channels=1).to(device)
      model.load_state_dict(torch.load(model_file_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
      model.eval()
    return model

def create_modality_combinations(modalities: list):
    modality_combinations = []
    for i in range(1,len(modalities)+1):   
      modality_combinations = modality_combinations + list(combinations(modalities,i))
    return modality_combinations

def create_UNET_input(val_data, modalities, dataset_name,model_modalities_trained_on,model_channel_map):
    zeros_arr = np.zeros_like(val_data[0])
    zeros_arr[:,modalities,:,:,:] = np.array(val_data[0][:,modalities,:,:,:])
    val_data[0] = torch.from_numpy(zeros_arr)
    input_data = torch.from_numpy(np.zeros((1,model_modalities_trained_on,val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
    input_data[:,model_channel_map[dataset_name],:,:,:] = val_data[0][:,range(0,val_data[0].shape[1]),:,:,:]
    return input_data

def create_UNET_input_quicktest(val_data, modalities, channel_map, model_modalities_trained_on):
    zeros_arr = np.zeros_like(val_data[0])
    zeros_arr[:,modalities,:,:,:] = np.array(val_data[0][:,modalities,:,:,:])
    val_data[0] = torch.from_numpy(zeros_arr)
    input_data = torch.from_numpy(np.zeros((1,model_modalities_trained_on,val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
    input_data[:,channel_map,:,:,:] = val_data[0][:,range(0,val_data[0].shape[1]),:,:,:]
    return input_data

def save_nifti(tensor: torch.Tensor, file_path: str,affine):
    vars_numpy = tensor.cpu().detach().numpy()
    vars_numpy = np.squeeze(vars_numpy)
    # vars_numpy = np.transpose(vars_numpy,(1,2,3,0))    
    new_image = nib.Nifti1Image(vars_numpy,affine=affine.squeeze())       
    nib.save(new_image, file_path)
  