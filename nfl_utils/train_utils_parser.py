

import os
import yaml

import pandas as pd
import numpy as np

# basic torch and model functions
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import timm

# loaders and train functions
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torch
import torchvision
import torch.optim as optim

# augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# cross validation scoring
from sklearn.model_selection import GroupKFold
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score

import sys
# sys.path.append(os.path.join(os.path.abspath(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__)))
print(os.path.join(os.path.dirname(__file__)))
from log_utils import *

if torch.cuda.is_available():
    device = torch.device('cuda')
   
kaggle = False 
if not kaggle:
    INPUT_DIR = "/workspace/input"

with open(os.path.join(INPUT_DIR , "Config.yaml")) as f:
    CFG = yaml.safe_load(f)

"""
import argparse
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

group = parser.add_argument_group('Model parameters')
group.add_argument("--model_name", help="timm model_type", default="tf_efficientnet_b0")
group.add_argument("--inp_channels", help="use channels if RGB, set 3", default=3)
group.add_argument("--num_img_feature", help="timm model output feature num from back bone model", default=5)
group.add_argument("--pretrained", help="if use timm pretrained model, set True", default=True)

group = group.add_argument_group('Training parameters')
group.add_argument("--n_epoch", help="training epoch num", default=10)
group.add_argument("--n_folds", help="cross validation fold num", default=3)
group.add_argument("--train_folds", help="use folds list", default=[0, 1, 2])
group.add_argument("--lr", default=1e-4)
group.add_argument("--T_max", default=10)
group.add_argument("--min_lr", default=1e-8)
group.add_argument("--weight_decay", default=1e-6)
group.add_argument("--print_freq", help="training situation print freq", default=1000)

group = group.add_argument_group('Dataset parameters')
group.add_argument("--batch_size", help="train/valid batchsize", default=128)
group.add_argument("--num_workers", default=2)

group = group.add_argument_group('Experiment Directory')
parser.add_argument("--EXP_NAME", help="experiment name", default="DEBUG")
parser.add_argument("--kaggle", help="if kaggle set True, else False", default="False")

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

args, args_text = _parse_args()
"""
# -------------------------------
# Augmentations
# -------------------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.3, 0.3), p=0.5),
])

# -------------------------------
# Dataset
# -------------------------------
class NFLDataset(Dataset):
    def __init__(self, target_df, transform=None):
        self.target_df = target_df
        self.transform = transform

    def __len__(self):
        return len(self.target_df)

    def __getitem__(self, idx):
        target_info = self.target_df.iloc[idx]
        target = target_info.contact
        # read frame image
        game_play = target_info.game_play
        frame = target_info.frame
        view = target_info.EorS
        is_End = int(view=="Endzone")
        contact_id = target_info.contact_id
        contact_fileid = f"{contact_id}_{view}.jpg"
        contact_filename = os.path.join(CFG["CONTACT_IMG_DIR"], contact_fileid)
        img = cv2.imread(contact_filename)
        if img is None:
            img = np.zeros((224, 224, 3))
            img = np.transpose(img, (2, 0, 1)).astype(np.float)
            img = torch.tensor(img, dtype=torch.float)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=img)["image"]
            img = img / 255. # convert to 0-1
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = torch.tensor(img, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)

        return img, target, is_End
    
    
# -------------------------------
# Model
# -------------------------------
class NFLNet(nn.Module):
    def __init__(
        self,
        model_name = CFG["model_name"],
        out_features = CFG["num_img_feature"],
        inp_channels= CFG["inp_channels"],
        pretrained = CFG["pretrained"],
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels, num_classes=out_features)
        self.fc = nn.Linear(out_features, 1)

    def forward(self, img):
        img_emb = self.model(img)
        output = self.fc(img_emb)
        return output, img_emb

def train_fn(train_loader, model, criterion, epoch ,optimizer, scheduler):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    start, end = time.time(), time.time()
    for batch_idx, (images, targets, _) in enumerate(train_loader):
        images = images.to(device, non_blocking = True).float()
        targets = targets.to(device, non_blocking = True).float().view(-1, 1)      
        preds, _ = model(images)
        
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
        del preds, images, targets
    gc.collect()
    torch.cuda.empty_cache()
    return losses.avg


def valid_fn(model, valid_loader, criterion):
    model.eval()# モデルを検証モードに設定
    test_targets = []
    test_preds = []
    img_embs = []

    batch_time = AverageMeter()
    losses = AverageMeter()
    start, end = time.time(), time.time()
    view_list = []
    for batch_idx, (images, targets, is_End) in enumerate(valid_loader):
        images = images.to(device, non_blocking = True).float()
        targets = targets.to(device, non_blocking = True).float().view(-1, 1)
        with torch.no_grad():
            preds, img_emb = model(images)
            loss = criterion(preds, targets)
        losses.update(loss.item(), CFG["batch_size"])
        batch_time.update(time.time() - end)

        img_emb = img_emb.detach().cpu().numpy()
        img_embs.extend(img_emb)

        targets = targets.detach().cpu().numpy().ravel().tolist()
        preds = torch.sigmoid(preds).detach().cpu().numpy().ravel().tolist()

        test_preds.extend(preds)
        test_targets.extend(targets)
        view_list.extend(is_End.numpy().tolist())
        # score = matthews_corrcoef(preds, targets)
        if batch_idx % CFG["print_freq"] == 0 or batch_idx == (len(valid_loader)-1):
            print('\t EVAL: [{0}/{1}] '
                'Elapsed {remain:s} '
                'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                .format(
                    batch_idx, len(valid_loader), batch_time=batch_time, loss=losses,
                    remain=timeSince(start, float(batch_idx+1)/len(valid_loader)),
                ))
        del preds, images, targets
        gc.collect()
        torch.cuda.empty_cache()
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    return test_targets, test_preds, img_embs, view_list, losses.avg


def training_loop(target_df):
    # set model & learning fn
    oof_df = pd.DataFrame()
    kf = GroupKFold(n_splits=CFG["n_folds"])
    for fold, (idx_train, idx_valid) in enumerate(kf.split(target_df, target_df["contact_id"], target_df["game_play"])):
        print("---")
        print(f"fold {fold} start training...")
        model = NFLNet()
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"], amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG["T_max"], eta_min=CFG["min_lr"], last_epoch=-1)

        if not fold in CFG["train_folds"]:
            print(f"fold{fold} is skip")
            continue
        # separate train/valid data 
        train_df = target_df.iloc[idx_train]
        valid_df = target_df.iloc[idx_valid]
        print("train target contact")
        print(train_df["contact"].value_counts())
        print("valid target contact")
        print(valid_df["contact"].value_counts())
        # separate train/valid data 
        train_dataset = NFLDataset(train_df, train_transform)
        # valid_dataset = NFLDataset(valid_df, valid_transform)
        valid_dataset = NFLDataset(valid_df,)
        train_loader = DataLoader(train_dataset,batch_size=CFG["batch_size"], shuffle=True,
                                    num_workers = CFG["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_dataset,batch_size=CFG["batch_size"], shuffle = False,
                                    num_workers = CFG["num_workers"], pin_memory = True)

        # training
        best_score = -np.inf
        best_auc = -np.inf
        start_time, end = time.time(), time.time()
        for epoch in range(1, CFG["n_epoch"] + 1):
            print(f'\t === epoch: {epoch}: training ===')
            train_loss_avg = train_fn(train_loader, model, criterion, epoch ,optimizer, scheduler)
            valid_targets, valid_preds, valid_embs, view_list, valid_loss_avg = valid_fn(model, valid_loader, criterion)

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
            elapsed = (time.time() - start_time)/60
            auc_score = roc_auc_score(valid_targets, valid_preds)
            print(f'\t epoch:{epoch}, avg train loss:{train_loss_avg:.4f}, avg valid loss:{valid_loss_avg:.4f}')
            print(f'\t score:{valid_score:.4f}(th={valid_threshold}) AUC = {auc_score:.4f}=> time:{elapsed:.2f} min')
            scheduler.step()
            # validationスコアがbestを更新したらモデルを保存する
            if valid_score > best_score:
                best_score = valid_score
                model_name = CFG["model_name"]
                torch.save(model.state_dict(), f'{MODEL_DIR}/{model_name}_fold{fold}.pth')
                print(f'\t Epoch {epoch} - Save Best Score: {best_score:.4f}. Model is saved.')
                contact_id = valid_df["contact_id"].values
                _oof_df = pd.DataFrame({
                    "contact_id" : contact_id,
                    "pred" : valid_preds,
                    "contact" : valid_targets,
                    "fold" : fold,
                    "is_End" : view_list,
                })
                img_emb_colname = [f"img_emb_{idx}" for idx in range(CFG["num_img_feature"])]
                img_emb_df = pd.DataFrame(valid_embs, columns=img_emb_colname)
                _oof_df = pd.concat([_oof_df, img_emb_df], axis=1)
            
            logging_metrics_epoch(fold, epoch, train_loss_avg, valid_loss_avg, valid_score, valid_threshold, tn_best, fp_best, fn_best, tp_best, auc_score)

        del train_loader, train_dataset, valid_loader, valid_dataset
        oof_df = pd.concat([oof_df, _oof_df], axis = 0)
        del _oof_df
        gc.collect()
        torch.cuda.empty_cache()
    return oof_df

if __name__=="__main__":
    oof_df = training_loop()