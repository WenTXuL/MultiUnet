
class Training_config:
        
        epoch=600
        workers = 2 #numworker
        train_batch_size = 2 
        val_interval = 4 # the number of epochs between the validation  
        lr = 1e-3
        load_pre_trained_model = False # if true will load pre-train model
        BRATS_two_channel_seg = False # Using different segmentation ground truths for different sets of modalities on BRATS   if true, the input that dropped the FLAIR and T2 will use ground truth that not contain edema     default false 
        model_type = "UNET" # default model type unet
        cropped_input_size = [128,128,128]    

        drop_learning_rate = True
        drop_learning_rate_epoch = 150 # epoch at which to decrease the learning rate
        drop_learning_rate_value = 1e-4
        # model_save_path 
        model_save_path = "models/"
        load_model_path="models/xxxx.pt"  #for the check point

class Database_config:
        # In our work, we put all the samples in one folder. As a quick setting, We sorted the images with the file name and split training and testing databases depending on this order.
        # set size and channel (when using new database, information needs to be set here)
        channels={}
        #the modalities for each database
        channels['BRATS'] = ["FLAIR", "T1", "T1c", "T2"]
        channels['ATLAS'] = ["T1"]
        channels['MSSEG'] = ["FLAIR","T1","T1c","T2","PD"] 
        channels['ISLES'] = ["FLAIR", "T1", "T2", "DWI"]
        channels['WMH'] = ["FLAIR", "T1"]
        channels['VOETS'] = ["T1"]
        channels['TBI'] = ["FLAIR", "T1", "T2", "SWI"]
        train_size={}
        #size for each database
        #training set size
        train_size['BRATS'] = 444
        train_size['ATLAS'] = 459
        train_size['MSSEG'] = 37
        train_size['ISLES'] = 20
        train_size['WMH'] = 42
        train_size['VOETS'] = 30  
        total_size={}
        total_size['BRATS'] = 484
        total_size['ATLAS'] = 654
        total_size['MSSEG'] = 53
        total_size['ISLES']= 28  
        total_size['WMH']= 60
        total_size['VOETS']= 57  
        img_path={}
        seg_path={}
        img_path["BRATS"] = "data/BRATS/Images"
        
        if Training_config.BRATS_two_channel_seg:
                    #Need the ground truth file that has multiple channels, and each channel contains a different ground truth. !!!!Only need when you want to use multiple ground truths for BRATS, or you can ignore this
                    seg_path["BRATS"] = "data/BRATS/Labels_multiple"
        else:
                    #default setting
                    seg_path["BRATS"] = "data/BRATS/Labels"
        img_path["ATLAS"]= "data/ATLAS/Images"
        seg_path["ATLAS"]= "data/ATLAS/Labels"
        img_path["MSSEG"]="data/MSSEG/Images"
        seg_path["MSSEG"]="data/MSSEG/Labels"
        img_path["ISLES"]= "data/ISLES/Images"
        seg_path["ISLES"]= "data/ISLES/Labels"
        img_path["WMH"]="data/WMH/Images"
        seg_path["WMH"]="data/WMH/Labels"
        #!!!only for test.py
        val_size={}
        val_size["BRATS"]=40
        val_size["ATLAS"]=195
        val_size["ISLES"]=28
        val_size["MSSEG"]=16    
        val_size["WMH"]=18


class Test_config:
        save_segs= False # True to save the segmentation outputs 
        save_path = ""    
        model_file_path = "models/Train_BRATS_TBI_ATLAS_MSSEG_WMH.pth" #the path of the train model
        model_net_type = "UNet" #the type of the pre-train model
        model_modalities_trained_on = 6 #the number of modalities include in training 
        model_channel_map = {"VOETS2":[1,5],"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES":[1,3,5,0], "TBI":[1,3,5,2], "WMH":[1,3],"VOETS":[3]} #The allocated channel index of the modalities(each channel) in the testing databases (start from 0) For example "ATLAS": [3] means the T1 modality in ATLAS will be allocated to the forth channel

class Finetune_config:
        # simialr to training    
        epoch=600
        workers = 2
        train_batch_size = 2 
        val_interval = 4
        lr = 1e-3
        BRATS_two_channel_seg = False 
        model_type = "UNET"
        cropped_input_size = [128,128,128]     
        drop_learning_rate = True
        drop_learning_rate_epoch = 150 
        drop_learning_rate_value = 1e-4
        # model_save_path
        model_save_path = "models/"




