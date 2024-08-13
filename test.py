
import torch
from monai.utils import set_determinism
import utils
from monai.transforms import Compose,EnsureChannelFirst,Activations, AsDiscrete
from glob import glob
import os
from monai.data import ImageDataset,DataLoader,decollate_batch
import config
import argparse
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
import numpy as np


def create_dataset_for_test(dataset):
    val_imtrans = Compose([EnsureChannelFirst()])
    val_segtrans = Compose([EnsureChannelFirst()])
    Database_config=config.Database_config()
    img_path=Database_config.img_path
    seg_path=Database_config.seg_path
    val_size=Database_config.val_size  
    images = sorted(glob(os.path.join(img_path[dataset], "*.*")))
    segs = sorted(glob(os.path.join(seg_path[dataset], "*.*")))
    val_ds = ImageDataset(images[-val_size[dataset]:], segs[-val_size[dataset]:], transform=val_imtrans, seg_transform=val_segtrans,image_only=False)   
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=0)
    return val_loader 

def test(model, val_loader, dataset_name, modalities,model_net_type,model_modalities_trained_on,model_channel_map, device, save_outputs, save_path):
  cropped_input_size = [128,128,128]

  #can add other metrics (here only show dice)
  dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
  with torch.no_grad():

    #initialize
    val_images = None
    val_labels = None
    val_outputs = None
    steps = 0
    dice_metric.reset()
    dice_metrics = []
    segment_pixel_vol = []
    gt_pixel_vol = []
    for val_data in val_loader:      
      roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
      sw_batch_size = 1

      #test using sliding window
      if model_net_type == "UNet":
        val_data[0] = utils.create_UNET_input(val_data, modalities, dataset_name, model_modalities_trained_on,model_channel_map)      
    
      val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
      val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
      post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
      val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]     

      # compute metric for the current iteration
      current_dice = dice_metric(y_pred=val_outputs, y=val_labels)

      if save_outputs:
        #save output with the original affine
        file_save_path = save_path + str(steps)+'_'+str(current_dice) + ".nii.gz"
        utils.save_nifti(val_outputs[0], file_save_path, val_data[3]["affine"])
      pixels_segmented = np.count_nonzero(val_outputs[0])
      gt_segmented = np.count_nonzero(val_labels[0])
      segment_pixel_vol.append(pixels_segmented)
      gt_pixel_vol.append(gt_segmented)
      steps+=1

    metric = dice_metric.aggregate().item()
    print("DICE Metric:")
    print(metric)
    dice_metric.reset()
    return dice_metrics, segment_pixel_vol, gt_pixel_vol

if __name__=="__main__":  

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument("--datasets_to_test", help="dataset for testing", type=str)
    parser.add_argument("--modalities_to_test", help="The modalities for testing (the index of the modalities for that input),using '_' to separate if 0_1_2 for BRATS it would mean test on FLAIR, T1, T1c", type=str)
    parser.add_argument("--test_all_combinations", help="0 or 1 1 if testing on all possible modality combinations", type=int, default='0')
    args = parser.parse_args()
     
    test_all_combinations=bool(args.test_all_combinations)
    cuda_id = "cuda:" + str(args.device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)
    results = {}    
    Test_config=config.Test_config()

    print("*************** TESTING NET " + str(Test_config.model_file_path) + " **************")        

    model = utils.create_net(Test_config.model_file_path,Test_config.model_net_type,Test_config.model_modalities_trained_on, device, cuda_id)

    print("************** TESTING DATASET " + args.datasets_to_test + " ***************")
    dataloader = create_dataset_for_test(args.datasets_to_test)
    if test_all_combinations:
                modalities = utils.create_modality_combinations([int(x) for x in args.modalities_to_test.split("_")])
    else:
                modalities = [[int(x) for x in args.modalities_to_test.split("_")]]

    for combination in modalities:
                print(combination)
                dsc_scores, seg_pix_vols, gt_pix_vols = test(model,
                    dataloader,
                    args.datasets_to_test,
                    combination,
                    Test_config.model_net_type,
                    Test_config.model_modalities_trained_on,
                    Test_config.model_channel_map,
                    device,
                    save_outputs= Test_config.save_segs,
                    save_path=Test_config.save_path,
                    )
 


