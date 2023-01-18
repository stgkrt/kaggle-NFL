#!/usr/bin/env python
# coding: utf-8

# # NFL Baseline
# - create target_df (distance in tracking_df is lower than threshold=3)
# https://www.kaggle.com/code/stgkrtua/nfl-creatatraindataset-targetdf
# - create dataset save frames in target_df
# https://www.kaggle.com/code/stgkrtua/nfl-createdataset-saveframes
# - check saved images
# https://www.kaggle.com/code/stgkrtua/nfl-checkdataset-plotsavedimage

# # import libraries

# general
import os
import gc
import pickle
import glob
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import cv2
import matplotlib.pyplot as plt
import time
import math

import sys
sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')
import timm


# deep learning
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold

# loss metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix

import mlflow
import wandb
# warningの表示方法の設定
import warnings
warnings.filterwarnings("ignore")


# # Set Configurations
CFG = {
        "kaggle" : False,
        "DEBUG" : False,
        # model config
        "model_name" : "swin_s3_tiny_224",
        "out_features" : 20,
        "inp_channels": 3*2,
        "pretrained" : True,
        "features" : ['x_position_1', 'y_position_1', 'x_position_2', 'y_position_2', 
                      'speed_1', 'distance_1', 'direction_1', 'orientation_1','acceleration_1', 'sa_1', 
                      'speed_2', 'distance_2', 'direction_2', 'orientation_2', 'acceleration_2', 'sa_2',
                      'players_dis', 'is_ground'],
        "track_features_x_1" : ['x_position_shift-6_1','x_position_shift-5_1', 'x_position_shift-4_1',
                                'x_position_shift-3_1','x_position_shift-2_1', 'x_position_shift-1_1', 
                                'x_position_shift0_1','x_position_shift1_1', 'x_position_shift2_1', 
                                'x_position_shift3_1','x_position_shift4_1', 'x_position_shift5_1'],    
        "track_features_y_1" : ['y_position_shift-6_1','y_position_shift-5_1', 'y_position_shift-4_1',
                                'y_position_shift-3_1','y_position_shift-2_1', 'y_position_shift-1_1',
                                'y_position_shift0_1','y_position_shift1_1', 'y_position_shift2_1',
                                'y_position_shift3_1','y_position_shift4_1', 'y_position_shift5_1'],
        "track_features_x_2" : ['x_position_shift-6_2','x_position_shift-5_2', 'x_position_shift-4_2',
                                'x_position_shift-3_2','x_position_shift-2_2', 'x_position_shift-1_2',
                                'x_position_shift0_2','x_position_shift1_2', 'x_position_shift2_2',
                                'x_position_shift3_2','x_position_shift4_2', 'x_position_shift5_2'],
        "track_features_y_2" : ['y_position_shift-6_2','y_position_shift-5_2', 'y_position_shift-4_2',
                                'y_position_shift-3_2','y_position_shift-2_2', 'y_position_shift-1_2',
                                'y_position_shift0_2','y_position_shift1_2', 'y_position_shift2_2',
                                'y_position_shift3_2','y_position_shift4_2', 'y_position_shift5_2'],
        # learning config
        "n_epoch" : 5,
        "lr" : 5e-5,
        "T_max" : 10,
        "min_lr" : 1e-8,
        "weight_decay" : 1e-6,

        # etc
        "print_freq" : 1000,
        "random_seed" : 21,

        # data config    
        "img_size" : (224, 224),
        "batch_size" : 32,
        "num_workers" : 8,
        "masksize_helmet_ratio" : 4, # helmetサイズにこの係数をかけたサイズだけ色を残して後は黒塗りする
        "TRAIN_VIDEO_NUM" : 200,
        "VALID_VIDEO_NUM" : 10,
        "sample_num" : -1, 

        "EXP_CATEGORY" : "exps",
        "EXP_NAME" : "local_exp001_swins3tiny",
}

if CFG["DEBUG"]:
    CFG["EXP_CATEGORY"] = "DEBUG"
    CFG["EXP_NAME"] = "DEBUG"
    CFG["n_epoch"] = 2
    CFG["sample_num"] = 1000
    CFG["batch_size"] = 32

if CFG["kaggle"]:
    CFG["INPUT_DIR"] = "/kaggle/input/"
    CFG["OUTPUT_DIR"] = "/kaggle/working/"
    CFG["TRAIN_HELMET_CSV"] = os.path.join(CFG["INPUT_DIR"], "nfl-player-contact-detection", "train_baseline_helmets.csv")
    CFG["TRAIN_TRACKING_CSV"] = os.path.join(CFG["INPUT_DIR"], "nfl-player-contact-detection", "train_player_tracking.csv")
    CFG["TRAIN_VIDEO_META_CSV"] = os.path.join(CFG["INPUT_DIR"], "nfl-player-contact-detection", "train_video_metadata.csv")
    CFG["TRAIN_LABEL_CSV"] = os.path.join(CFG["INPUT_DIR"], "nfl-player-contact-detection", "train_labels.csv")
    CFG["TARGET_CSV"] = os.path.join(CFG["INPUT_DIR"], "nfl-createdataset-saveframes-shift", "saved_frame_target.csv")
    CFG["TRAIN_E_IMG_DIR"] = os.path.join(CFG["INPUT_DIR"], "nfl-createdataset-saveframes-shift", "train_images")
    CFG["TRAIN_S_IMG_DIR"] = os.path.join(CFG["INPUT_DIR"], "nfl-createdataset-saveframes-shift-sview", "train_images")
    CFG["MODEL_DIR"] = CFG["OUTPUT_DIR"]
else:
    CFG["INPUT_DIR"] = "/workspace/input"
    CFG["OUTPUT_DIR"] = "/workspace/output"
    CFG["TRAIN_HELMET_CSV"] = os.path.join(CFG["INPUT_DIR"], "train_baseline_helmets.csv")
    CFG["TRAIN_TRACKING_CSV"] = os.path.join(CFG["INPUT_DIR"], "train_player_tracking.csv")
    CFG["TRAIN_VIDEO_META_CSV"] = os.path.join(CFG["INPUT_DIR"], "train_video_metadata.csv")
    CFG["TRAIN_LABEL_CSV"] = os.path.join(CFG["INPUT_DIR"], "train_labels.csv")
    CFG["TARGET_CSV"] = os.path.join(CFG["INPUT_DIR"], "target_fillna0_shift_2.csv")
    CFG["TRAIN_E_IMG_DIR"] = os.path.join(CFG["INPUT_DIR"], "train_frames")
    CFG["TRAIN_S_IMG_DIR"] = CFG["TRAIN_E_IMG_DIR"]
    CFG["MODEL_DIR"] = os.path.join(CFG["OUTPUT_DIR"], CFG["EXP_NAME"] ,"model")
    
if not CFG["kaggle"] and not CFG["DEBUG"]:
    os.mkdir(os.path.join(CFG["OUTPUT_DIR"], CFG["EXP_NAME"]))
    os.mkdir(CFG["MODEL_DIR"])


if CFG["kaggle"]:
    os.environ["WANDB_SILENT"] = "true"
    WANDB_CONFIG = {'competition': 'NFL', '_wandb_kernel': 'taro'}
    # Secrets
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("wandb")

    get_ipython().system('wandb login $secret_value_0')
    #! TODO : logger settings
    wandb.init(project=WANDB_CONFIG["competition"], config=CFG, group=CFG["EXP_CATEGORY"], name=CFG["EXP_NAME"])

else:
    mlflow.set_tracking_uri("/workspace/mlruns")
    experiment = mlflow.get_experiment_by_name(CFG["EXP_CATEGORY"])
    if experiment is None:  # 当該Experiment存在しないとき、新たに作成
        experiment_id = mlflow.create_experiment(name=CFG["EXP_CATEGORY"])
    else: # 当該Experiment存在するとき、IDを取得
        experiment_id = experiment.experiment_id


# # Utils
def logging_metrics_epoch(fold, epoch, train_loss_avg, valid_loss_avg, score, threshold, tn_best, fp_best, fn_best, tp_best):
    if CFG["kaggle"]:
        wandb.log({"loss avg":{f"train/fold{fold}": train_loss_avg,
                                f"valid/fold{fold}": valid_loss_avg}}, step=epoch)
        wandb.log({"Metircs" : {f"score/fold{fold}":score,
                                f"score threshold/fold{fold}":threshold,
                                f"tn/fold{fold}":tn_best,
                                f"fp/fold{fold}":fp_best,
                                f"fn/fold{fold}":fn_best,
                                f"tp/fold{fold}":tp_best,}}, step=epoch)
    else:
        mlflow.log_metric(f"fold{fold} train loss avg", train_loss_avg, step=epoch)
        mlflow.log_metric(f"fold{fold} valid loss avg", valid_loss_avg, step=epoch)
        mlflow.log_metric(f"fold{fold} score", score, step=epoch)
        mlflow.log_metric(f"fold{fold} score threshold", threshold, step=epoch)
        mlflow.log_metric(f"fold{fold} tn", tn_best, step=epoch)
        mlflow.log_metric(f"fold{fold} fp", fp_best, step=epoch)
        mlflow.log_metric(f"fold{fold} fn", fn_best, step=epoch)
        mlflow.log_metric(f"fold{fold} tp", tp_best, step=epoch)

def seed_everything(seed=CFG["random_seed"]):
    #os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed%(2**32-1))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False
seed_everything()

def asMinutes(s):
    """Convert Seconds to Minutes."""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    """Accessing and Converting Time Data."""
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ## Dataset Utils
def set_inimg_window(crop_pos, mask_size, img_size=(720, 1280)):#crop_pos = [left, top, right, bot]
    if mask_size[1] >= img_size[0]:
        top, bot = 0, img_size[1]
    else:
        top=(crop_pos[1] + crop_pos[3])//2 - mask_size[1]//2
        bot=(crop_pos[1] + crop_pos[3])//2 + mask_size[1]//2
        if top < 0:
            bot = bot - top
            top = 0
        elif bot > img_size[0]:
            top = top - (bot-img_size[0])
            bot = img_size[0]

    if mask_size[0] >= img_size[1]:
        left, right = 0, img_size[1]
    else:
        left = (crop_pos[0] + crop_pos[2])//2 - mask_size[0]//2
        right = (crop_pos[0] + crop_pos[2])//2 + mask_size[0]//2
        if left < 0:
            right = right - left
            left = 0
        elif right > img_size[1]:
            left = left - (right - img_size[1])
            right = img_size[1]
    crop_area = np.array([left, top, right, bot]).astype(np.int)
    return crop_area

def get_crop_area(p1_helmet, p2_helmet, input_size=(720, 1280)):#helmet[left, width, top, height]
    if (p1_helmet[1]==0 and p1_helmet[3]==0) and (p2_helmet[1]==0 and p2_helmet[3]==0):
        crop_area = [0, 0, input_size[1], input_size[0]]
        # print("bose player's helmet is not detected.")
        return crop_area
    elif (p2_helmet[1]==0 and p2_helmet[3]==0) and (p1_helmet[1] != 0 and p1_helmet[3]!=0):
        # print("p1 detected.")
        crop_x_center, crop_y_center = p1_helmet[0] + (p1_helmet[1])//2, p1_helmet[2] + (p1_helmet[3])//2
        helmet_base_size = (p1_helmet[1] + p1_helmet[3])*0.5*CFG["masksize_helmet_ratio"]*4
        output_size = [helmet_base_size, helmet_base_size]
    elif (p1_helmet[1]==0 and p1_helmet[3]==0) and (p2_helmet[1]!=0 and p2_helmet[3]!=0):
        # print("p2 detected.")
        crop_x_center, crop_y_center = p2_helmet[0] + (p2_helmet[1])//2, p2_helmet[2] + (p2_helmet[3])//2
        helmet_base_size = (p2_helmet[1] + p2_helmet[3])*0.5*CFG["masksize_helmet_ratio"]*4
        output_size = [helmet_base_size, helmet_base_size]
    else:
    #     print("p1 and p2 detected.")
        p1_x_center, p1_y_center = p1_helmet[0] + (p1_helmet[1])//2, p1_helmet[2] + (p1_helmet[3])//2
        p2_x_center, p2_y_center = p2_helmet[0] + (p2_helmet[1])//2, p2_helmet[2] + (p2_helmet[3])//2
        crop_x_center, crop_y_center = (p1_x_center + p2_x_center)//2, (p1_y_center + p2_y_center)//2
        helmet_base_size = (abs(p1_x_center - p2_x_center) + abs(p1_y_center - p2_y_center))*0.5 + ((p1_helmet[1] + p2_helmet[1])*0.5 + (p1_helmet[3] + p2_helmet[3])*0.5)*0.5*CFG["masksize_helmet_ratio"]*2
        output_size = [helmet_base_size, helmet_base_size]
    
    # print("crop center", crop_x_center, crop_y_center)
    crop_left = crop_x_center - output_size[1]//2
    crop_top = crop_y_center - output_size[0]//2
    crop_right = crop_x_center + output_size[1]//2
    crop_bot = crop_y_center + output_size[0]//2
    crop_area = [crop_left, crop_top, crop_right, crop_bot]
    crop_area = set_inimg_window(crop_area, output_size)
    return crop_area

def get_playermasked_img(img, helmet_pos, img_size=(720, 1280, 3)):#helmet pos = [left, width, top, height]
    if helmet_pos[1] == 0 and helmet_pos[3] == 0:
        mask = np.ones_like(img)
        return mask
    mask_size=(helmet_pos[1]+helmet_pos[3])*0.5*CFG["masksize_helmet_ratio"]# helmetの大きさによってplayerの範囲も変更
    helmet_area = [helmet_pos[0], helmet_pos[2], helmet_pos[0]+helmet_pos[1], helmet_pos[2]+helmet_pos[3]]#[left, top, right, bot]
    player_area = set_inimg_window(helmet_area, (mask_size,mask_size))
    mask = np.zeros(img_size, dtype=np.float)
    cv2.rectangle(mask, [player_area[0], player_area[1]], [player_area[2], player_area[3]], (255, 255, 255), -1)
    mask = np.clip(mask, 0, 1).astype(np.float)
    return mask

# # Dataset
class NFLDataset(Dataset):
    def __init__(self, target_df, transform=None):
        self.target_df = target_df
        self.features = target_df[CFG["features"]].values
        self.track_features_x_1 = target_df[CFG["track_features_x_1"]].values
        self.track_features_y_1 = target_df[CFG["track_features_y_1"]].values
        self.track_features_x_2 = target_df[CFG["track_features_x_2"]].values
        self.track_features_y_2 = target_df[CFG["track_features_y_2"]].values
        self.transform = transform

    def __len__(self):
        return len(self.target_df)

    def __getitem__(self, idx):
        target_info = self.target_df.iloc[idx]
        features = self.features[idx]
        track_x_1 = self.track_features_x_1[idx]
        track_y_1 = self.track_features_y_1[idx]
        track_x_2 = self.track_features_x_2[idx]
        track_y_2 = self.track_features_y_2[idx]
        track_features = np.concatenate([track_x_1[np.newaxis, :],
                                         track_y_1[np.newaxis, :],
                                         track_x_2[np.newaxis, :],
                                         track_y_2[np.newaxis, :]])

        target = target_info.contact
        # read frame image
        game_play = target_info.game_play
        frame = target_info.frame
        if CFG["kaggle"]:
            file_id = f"{game_play}_Endzone_{frame:05}.png"
        else:
            file_id = f"{game_play}_Endzone_{frame:04}.jpg"
        filename = os.path.join(CFG["TRAIN_E_IMG_DIR"], file_id)
        input_img = None
        img = cv2.imread(filename)
        if img is None:
            img = np.zeros((224, 224, 3))
            img = np.transpose(img, (2, 0, 1)).astype(np.float)
            img = torch.tensor(img, dtype=torch.float)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # player highlight mask
            player1 = target_info.nfl_player_id_1
            player2 = target_info.nfl_player_id_2
            p1_helmet = np.array([target_info.E_left_1, target_info.E_width_1,
                                target_info.E_top_1, target_info.E_height_1]).astype(np.int)
            p2_helmet = np.array([target_info.E_left_2, target_info.E_width_2,
                                target_info.E_top_2, target_info.E_height_2]).astype(np.int)
            
            mask1 = get_playermasked_img(img, p1_helmet)# helmet=[left, width, top, height]
            mask2 = get_playermasked_img(img, p2_helmet)
            mask = np.clip(mask1 + mask2, 0, 1).astype(np.float)
            img = mask*img
            
            # crop players area
            crop_area = get_crop_area(p1_helmet, p2_helmet)# crop_area=[left, top, right, bot]

            img = img[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2], :]
            img = cv2.resize(img, dsize=CFG["img_size"])
            img = img / 255. # convert to 0-1
            img = np.transpose(img, (2, 0, 1)).astype(np.float)
            img = torch.tensor(img, dtype=torch.float)
        input_img = img
        # sideline_vies
        if CFG["kaggle"]:
            file_id = f"{game_play}_Sideline_{frame:05}.png"
        else:
            file_id = f"{game_play}_Sideline_{frame:04}.jpg"
        filename = os.path.join(CFG["TRAIN_S_IMG_DIR"], file_id)
        img = cv2.imread(filename)
        if img is None:
            img = np.zeros((224, 224, 3))
            img = np.transpose(img, (2, 0, 1)).astype(np.float)
            img = torch.tensor(img, dtype=torch.float)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # player highlight mask
            player1 = target_info.nfl_player_id_1
            player2 = target_info.nfl_player_id_2
            p1_helmet = np.array([target_info.S_left_1, target_info.S_width_1,
                                target_info.S_top_1, target_info.S_height_1]).astype(np.int)
            p2_helmet = np.array([target_info.S_left_2, target_info.S_width_2,
                                target_info.S_top_2, target_info.S_height_2]).astype(np.int)
            mask1 = get_playermasked_img(img, p1_helmet)# helmet=[left, width, top, height]
            mask2 = get_playermasked_img(img, p2_helmet)
            mask = np.clip(mask1 + mask2, 0, 1).astype(np.float)
            img = mask*img
            
            # crop players area
            crop_area = get_crop_area(p1_helmet, p2_helmet)# crop_area=[left, top, right, bot]
            img = img[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2], :]
            img = cv2.resize(img, dsize=CFG["img_size"])
            img = img / 255. # convert to 0-1
            img = np.transpose(img, (2, 0, 1)).astype(np.float)
            img = torch.tensor(img, dtype=torch.float)
        input_img = np.concatenate([input_img, img], axis=0)
        target = torch.tensor(target, dtype=torch.float)
        features = torch.tensor(features, dtype=torch.float)
        track_features = torch.tensor(track_features, dtype=torch.float)
        return input_img, features, track_features, target


# # Model
class NFLNet(nn.Module):
    def __init__(
        self,
        model_name = CFG["model_name"],
        out_features = CFG["out_features"],
        inp_channels= CFG["inp_channels"],
        pretrained = CFG["pretrained"]
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels, num_classes=out_features)
        self.mlp = nn.Sequential(
                        nn.Linear(len(CFG["features"]), 32),
                        nn.LayerNorm(32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                    )
        self.conv1 = nn.Sequential(
                        nn.Conv1d(4, 1, 5),
                        nn.Linear(len(CFG["track_features_y_2"])-4, 32),
                        nn.ReLU(),
                    )
        self.fc = nn.Linear(out_features+32+32, 1)

    def forward(self, image, features, track_features):
        output = self.model(image)
        features = self.mlp(features)
        track_features = self.conv1(track_features)
        output = self.fc(torch.cat([output, features, torch.squeeze(track_features)], dim=1))
        return output


# # train fn
def train_fn(train_loader, model, criterion, epoch ,optimizer, scheduler):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    start = end = time.time()
    for batch_idx, (images, features, track_features, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking = True).float()
        targets = targets.to(device, non_blocking = True).float().view(-1, 1)      
        features = features.to(device, non_blocking = True).float()
        track_features = track_features.to(device, non_blocking = True).float()
        preds = model(images, features, track_features)
        
        loss = criterion(preds, targets)
        losses.update(loss.item(), CFG["batch_size"]) 
        targets = targets.detach().cpu().numpy().ravel().tolist()
        preds = torch.sigmoid(preds).detach().cpu().numpy().ravel().tolist()

        loss.backward() # パラメータの勾配を計算
        optimizer.step() # モデル更新
        optimizer.zero_grad() # 勾配の初期化
                
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % CFG["print_freq"] == 0 or batch_idx == (len(train_loader)-1):
            print('\t Epoch: [{0}][{1}/{2}] '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    .format(
                        epoch, batch_idx, len(train_loader), batch_time=batch_time, loss=losses,
                        remain=timeSince(start, float(batch_idx+1)/len(train_loader)),
            ))
        del preds, images, features, targets
    gc.collect()
    torch.cuda.empty_cache()
    return losses.avg


# # valid fn
def valid_fn(model, valid_loader, criterion):
    model.eval()# モデルを検証モードに設定
    test_targets = []
    test_preds = []

    batch_time = AverageMeter()
    losses = AverageMeter()
    start = end = time.time()
    for batch_idx, (images, features, track_features, targets) in enumerate(valid_loader):
        images = images.to(device, non_blocking = True).float()
        targets = targets.to(device, non_blocking = True).float().view(-1, 1)
        features = features.to(device, non_blocking = True).float()
        track_features = track_features.to(device, non_blocking = True).float()
        with torch.no_grad():
            preds = model(images, features, track_features)
            loss = criterion(preds, targets)
        losses.update(loss.item(), CFG["batch_size"])
        batch_time.update(time.time() - end)

        targets = targets.detach().cpu().numpy().ravel().tolist()
        preds = torch.sigmoid(preds).detach().cpu().numpy().ravel().tolist()

        test_preds.extend(preds)
        test_targets.extend(targets)
        # score = matthews_corrcoef(preds, targets)
        if batch_idx % CFG["print_freq"] == 0 or batch_idx == (len(valid_loader)-1):
            print('\t EVAL: [{0}/{1}] '
                'Elapsed {remain:s} '
                'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                .format(
                    batch_idx, len(valid_loader), batch_time=batch_time, loss=losses,
                    remain=timeSince(start, float(batch_idx+1)/len(valid_loader)),
                ))
        del preds, images, features, targets
        gc.collect()
        torch.cuda.empty_cache()
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    return test_targets, test_preds, losses.avg


# # Train loop
def training_loop(target_df):
    # set model & learning fn
    model = NFLNet()
    model = model.to(device)
    if CFG["kaggle"]:
        wandb.watch(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"], amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["T_max"], eta_min=CFG["min_lr"], last_epoch=-1)

    oof_df = pd.DataFrame()
    fold = 0
    # separate train/valid data 
    train_df = target_df[target_df["game_play"].isin(train_game_plays)]
    valid_df = target_df[target_df["game_play"].isin(valid_game_plays)]
    train_dataset = NFLDataset(train_df)
    valid_dataset = NFLDataset(valid_df)
    train_loader = DataLoader(train_dataset,batch_size=CFG["batch_size"], shuffle = True,
                                num_workers = CFG["num_workers"], pin_memory = True)
    valid_loader = DataLoader(valid_dataset,batch_size=CFG["batch_size"], shuffle = True,
                                num_workers = CFG["num_workers"], pin_memory = True)

    # training
    best_score = -np.inf
    start_time = end = time.time()
    for epoch in range(1, CFG["n_epoch"] + 1):
        print(f'\t === epoch: {epoch}: training ===')
        train_loss_avg = train_fn(train_loader, model, criterion, epoch ,optimizer, scheduler)
        valid_targets, valid_preds, valid_loss_avg = valid_fn(model, valid_loader, criterion)

        valid_score = -np.inf
        valid_threshold = 0
        tn_best, fp_best, fn_best, tp_best = 0, 0, 0, 0
        for idx in range(1, 10, 1):
            thr = idx*0.1
            valid_targets = (np.array(valid_targets) > thr).astype(np.int32)
            valid_binary_preds = (np.array(valid_preds) > thr).astype(np.int32)
            score_tmp = matthews_corrcoef(valid_targets, valid_binary_preds)
            cm = confusion_matrix(valid_targets, valid_binary_preds)
            tn, fp, fn, tp = cm.flatten()
            if score_tmp > valid_score:
                valid_score = score_tmp 
                valid_threshold = thr
                tn_best, fp_best, fn_best, tp_best = tn, fp, fn, tp
        elapsed = time.time() - start_time
        print(f'\t epoch:{epoch}, avg train loss:{train_loss_avg:.4f}, avg valid loss:{valid_loss_avg:.4f}, score:{valid_score:.4f}(th={valid_threshold}) ::: time:{elapsed:.2f}s')
        scheduler.step()
        # validationスコアがbestを更新したらモデルを保存する
        if valid_score > best_score:
            best_score = valid_score
            model_name = CFG["model_name"]
            torch.save(model.state_dict(), f'{CFG["MODEL_DIR"]}/{model_name}_fold{fold}.pth')
            print(f'\t Epoch {epoch} - Save Best Score: {best_score:.4f}. Model is saved.')
            contact_id = valid_df["contact_id"].values
            _oof_df = pd.DataFrame({
                "contact_id" : contact_id,
                "pred" : valid_preds,
                "contact" : valid_targets,
                "fold" : fold,
            })
        logging_metrics_epoch(fold, epoch, train_loss_avg, valid_loss_avg, valid_score, valid_threshold, tn_best, fp_best, fn_best, tp_best)

    del train_loader, train_dataset, valid_loader, valid_dataset
    oof_df = pd.concat([oof_df, _oof_df], axis = 0)
    del _oof_df
    gc.collect()
    torch.cuda.empty_cache()
    return oof_df

if __name__=="__main__":
    # device optimization
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')


    target_df = pd.read_csv(CFG["TARGET_CSV"])
    # target_game_plays = target_df["game_play"].unique()[:50]
    train_game_plays = target_df["game_play"].unique()[:CFG["TRAIN_VIDEO_NUM"]]
    valid_game_plays = target_df["game_play"].unique()[-CFG["VALID_VIDEO_NUM"]:]
    target_game_plays = list(set(train_game_plays) | set(valid_game_plays))
    CFG["target_game_plays"] = list(target_game_plays)
    target_df = target_df[target_df["game_play"].isin(target_game_plays)]
    print("train games = ", train_game_plays)
    print("valid games = ", valid_game_plays)
    
    if CFG["DEBUG"]:
        target_df = target_df.sample(CFG["sample_num"])
    elif CFG["sample_num"] != -1:
        target_df = target_df.sample(CFG["sample_num"])

    target_df["is_ground"] = (target_df["nfl_player_id_2"] == "G").astype(np.int)
    print("data num=", len(target_df))
    print("positive num = ", len(target_df[target_df["contact"]==1]))
    print("negative num =", len(target_df[target_df["contact"]!=1]))


    # run exp
    if CFG["kaggle"]:
        oof_df = training_loop(target_df)
        wandb.finish()
        oof_filename = os.path.join(CFG["OUTPUT_DIR"], "oof_df.csv")
        oof_df.to_csv(oof_filename, index=False)
    else:
        with mlflow.start_run(experiment_id=experiment_id, run_name=CFG["EXP_NAME"]) as run:
            mlflow.log_dict(CFG, "configuration.yaml")
            mlflow.log_param("positive data num", len(target_df[target_df["contact"]==1]))
            mlflow.log_param("negative data num", len(target_df[target_df["contact"]==0]))
            oof_df = training_loop(target_df)
            oof_filename = os.path.join(CFG["OUTPUT_DIR"], CFG["EXP_NAME"], "oof_df.csv")
            oof_df.to_csv(oof_filename, index=False)

