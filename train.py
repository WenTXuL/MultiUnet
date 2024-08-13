import torch
from glob import glob
import os
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,ConfusionMatrixMetric,MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism
from monai.losses.dice import DiceLoss
from nets.unet import res_unet
import numpy as np
import utils
#import wandb
import config
import argparse
#from config import Training_config,Database_config

if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy('file_system') 

    #command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument("--datasets", help="datasets for training, using '_' to separate", type=str)
    parser.add_argument("--save_name", help="File name for saving model weights and checkpoints", type=str, default='save')
    parser.add_argument("--randomly_drop", help="0 or 1, 1 if random dropping modalities when training", type=int, default='1')

    args = parser.parse_args()
    randomly_drop = bool(args.randomly_drop)

    #load config
    Training_config = config.Training_config()
    Database_config = config.Database_config()

    cropped_input_size = Training_config.cropped_input_size
    epochs = Training_config.epoch

    # Use wandb for recording
    # run = wandb.init(
    # # Set the project where this run will be logged
    # project="all_in_one",
    # name=save_name,
    # )  

    #print setting for training
    print("lr: ",Training_config.lr)
    print("Workers: ", Training_config.workers)
    print("Batch size: ",Training_config.train_batch_size)
    print("RANDOMLY DROP? ",randomly_drop)

    #set index
    img_index = 0
    label_index = 1    

    # Set the data size and total modalities
    channels= Database_config.channels
    train_size=Database_config.train_size
    total_size=Database_config.total_size
    datasetlist=args.datasets.split("_")
    total_modalities=[]
    total_modalities=set(total_modalities)
    data_size=0
    for dataset in datasetlist:
        total_modalities=total_modalities.union(set(channels[dataset]))
        data_size=max(data_size,train_size[dataset])
    total_modalities = sorted(list(total_modalities))   
    print("data_size",data_size)
    print("Total modalities: ", total_modalities)
    
    # Loop for allocating channel
    channel_map={}
    for dataset in datasetlist:
         channel_map[dataset]=utils.map_channels(channels[dataset], total_modalities)
         print("channel map:", dataset,channel_map[dataset])    

    # path initialization
    train_loaders = []
    val_loaders = []
    val_loader={}
    data_loader_map = {}    
    img_path=Database_config.img_path
    seg_path=Database_config.seg_path
    model_save_path = Training_config.model_save_path
    load_model_path = Training_config.load_model_path         

    # get dataloader
    for dataset in datasetlist:
        print("Training: ",dataset)
        val_size = total_size[dataset]-train_size[dataset]
        images= sorted(glob(os.path.join(img_path[dataset], "*.*")))
        segs = sorted(glob(os.path.join(seg_path[dataset],"*.*"))) 
        train_loader_one, val_loader[dataset] = utils.create_dataloader(val_size=val_size, images=images,segs=segs, workers=Training_config.workers,train_batch_size=Training_config.train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size[dataset],cropped_input_size=cropped_input_size)
        data_loader_map[dataset] = len(train_loaders)
        train_loaders.append(train_loader_one)
        val_loaders.append(val_loader[dataset])
     
    # initialize GPU
    print("Running on GPU:" + str(args.device_id))
    print("Running for epochs:" + str(epochs))
    cuda_id = "cuda:" + str(args.device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)
    
    # initialize metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    sensitivity_metric = ConfusionMatrixMetric(include_background=True, metric_name='sensitivity', reduction="mean", get_not_nans=False)    
    precision_metric = ConfusionMatrixMetric(include_background=True, metric_name='precision', reduction="mean", get_not_nans=False)   
    IOU_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # initialize the model (only show multiunet here)
    print("in_channels= ",len(total_modalities))
    print("batch size = ",Training_config.train_batch_size)
    if Training_config.model_type == "UNET":
        print("TRAINING WITH UNET")
        model = res_unet(in_channels=len(total_modalities)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Training_config.lr)
        epoched=0
        # load pre-trained weights
        if Training_config.load_pre_trained_model:
            print("LOADING MODEL: ", load_model_path)
            checkpoint = torch.load(load_model_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id})
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoched=checkpoint['epoch']+1   

    # defined loss function
    loss_function = DiceLoss(sigmoid=True)    

    # initialize the best metric
    best_metric={}
    best_metric_epoch={}
    for dataset in datasetlist:
        best_metric[dataset] = -1
        best_metric_epoch[dataset] = -1    

    metric_values = list()

    #training
    for epoch in range(epoched,epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        # drop learning rate 
        if Training_config.drop_learning_rate and epoch >= Training_config.drop_learning_rate_epoch:
            for g in optimizer.param_groups:
                g['lr'] = Training_config.drop_learning_rate_value

        for batch_data in zip(*train_loaders): 
            step += 1
            outputs = []
            labels = []
            for dataset in datasetlist:
                #Only for BRATS    BRATS may use different ground truth
                if dataset == "BRATS":
                    loader_index = data_loader_map["BRATS"]
                    batch = batch_data[loader_index]                
                    if randomly_drop:
                        modalities_remaining, batch[img_index] = utils.rand_set_channels_to_zero(channels["BRATS"], batch[img_index])
                        for i in range (batch[label_index].shape[0]):
                        # For BRATS because edema can only be seen on some modalities can use different ground truth for different sets of modalities in input (this need the ground truth file that have multiple channels and for each channel it contain a different gound truth)
                            if (0 not in modalities_remaining[i]) and (3 not in modalities_remaining[i]):
                                # Edema cannot be seen so change segmentation to labels without edema
                                seg_channel = 1
                            else:
                                seg_channel = 0
                            if Training_config.BRATS_two_channel_seg:                            
                                label[i,:,:,:,:] = batch[label_index][i,[seg_channel],:,:,:].to(device)                                
                            else:
                                #default setting of our work: not using different labels
                                label = batch[label_index].to(device)
                    else:                        
                        label = batch[label_index].to(device)                         
                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    input_data[:,channel_map["BRATS"],:,:,:] = batch[img_index]
                    input_data = input_data.to(device)        
                    out = model(input_data)
                    outputs.append(out)
                    labels.append(label)
                else: #other databases are similar
                    loader_index = data_loader_map[dataset]
                    batch = batch_data[loader_index]

               
                    if randomly_drop:
                        _, batch[img_index] = utils.rand_set_channels_to_zero(channels[dataset], batch[img_index])          #ATLAS WILL ALWAYS BE ONE    
                  
              
                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    input_data[:,channel_map[dataset],:,:,:] = batch[img_index]
                    input_data = input_data.to(device)
                    
                    label = batch[label_index].to(device)
                    out = model(input_data)
                    outputs.append(out)
                    labels.append(label)
            optimizer.zero_grad()
            combined_outs = torch.cat(outputs, dim=0)
            combined_labels = torch.cat(labels,dim=0)
            loss = loss_function(combined_outs, combined_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = data_size  // Training_config.train_batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            #wandb.log({"loss":loss.item(),"epoch":epoch+1})
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        # save model
        if (epoch+1) % 50 == 0:            
            model_save_name = model_save_path + args.save_name + "_Epoch_" + str(epoch) + ".pth"
            opt_save_name=model_save_path + args.save_name + "_checkpoint_Epoch_" + str(epoch) + ".pt"
            torch.save(model.state_dict(), model_save_name)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss ,
            }, opt_save_name)
            print("Saved Model")
            
        # validation
        if (epoch + 1) % Training_config.val_interval == 0:
            model.eval()
            with torch.no_grad():
                seg_channel = 0
                val_images = None
                val_labels = None
                val_outputs = None
                metric={}
                dice_metric.reset()
                sensitivity_metric.reset()
                precision_metric.reset()                
                IOU_metric.reset()                
                for dataset in datasetlist:
                    metric[dataset]={}
                    loader_index = data_loader_map[dataset]
                    for val_data in val_loader[dataset]:
                        # batch = val_data[loader_index]                       
                        input_data = torch.from_numpy(np.zeros((1,len(total_modalities),val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
                        input_data[:,channel_map[dataset],:,:,:] = val_data[0]
                        input_data = input_data.to(device)
                        if dataset == "BRATS" and Training_config.BRATS_two_channel_seg:
                            label = val_data[1][:,[0],:,:,:].to(device)                      
                        else:                        
                            label = val_data[1].to(device)                        
                        roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                        sw_batch_size = 1
                        #using sliding window for the whole 3D image
                        val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=label)
                        sensitivity_metric(y_pred=val_outputs, y=label)
                        precision_metric(y_pred=val_outputs, y=label)                        
                        IOU_metric(y_pred=val_outputs, y=label)           
                    metric[dataset]["dice"] = dice_metric.aggregate().item()
                    metric[dataset]["sensitivity"] = sensitivity_metric.aggregate()[0].item()
                    metric[dataset]["precision"] = precision_metric.aggregate()[0].item()                    
                    metric[dataset]["IOU"] = IOU_metric.aggregate().item()            
                    dice_metric.reset()
                    sensitivity_metric.reset()
                    precision_metric.reset()                    
                    IOU_metric.reset()
                    if metric[dataset]["dice"] > best_metric[dataset]:
                        best_metric[dataset] = metric[dataset]["dice"]
                        best_metric_epoch[dataset] = epoch + 1
                        if epoch>1:
                            model_save_name = model_save_path + args.save_name + "_BEST_"+dataset+".pth"
                            torch.save(model.state_dict(), model_save_name)                   
                            print("saved new best metric model")
                    print(
                        "current epoch: {} current mean dice {}: {:.4f} best mean dice {}: {:.4f} at epoch {}".format(
                            epoch + 1,dataset,metric[dataset]["dice"],dataset, best_metric[dataset], best_metric_epoch[dataset]
                        )
                    )
                    #wandb log  
                    #here only use wandb log to show other metric
                    #wandb.log({"epoch_val":epoch+1,"mdice_"+dataset:metric[dataset]["dice"], "sensitivity_"+dataset:metric[dataset]["sensitivity"],"precision_"+dataset:metric[dataset]["precision"],"mIOU_"+dataset:metric[dataset]["IOU"]})
                   
