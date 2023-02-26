
import time
import math

import wandb
import mlflow

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
        
        
def logging_metrics_epoch(fold, epoch, train_loss_avg, valid_loss_avg, score, threshold, tn_best, fp_best, fn_best, tp_best, auc_score):
    if CFG["kaggle"]:
        wandb.log({"loss avg":{f"train/fold{fold}": train_loss_avg,
                                f"valid/fold{fold}": valid_loss_avg}}, step=epoch)
        wandb.log({"Metircs" : {f"score/fold{fold}":score,
                                f"score threshold/fold{fold}":threshold,
                                f"tn/fold{fold}":tn_best,
                                f"fp/fold{fold}":fp_best,
                                f"fn/fold{fold}":fn_best,
                                f"tp/fold{fold}":tp_best,
                                f"auc/fold{fold}":auc_score,
                               }}, step=epoch)
    else:
        mlflow.log_metric(f"fold{fold} train loss avg", train_loss_avg, step=epoch)
        mlflow.log_metric(f"fold{fold} valid loss avg", valid_loss_avg, step=epoch)
        mlflow.log_metric(f"fold{fold} score", score, step=epoch)
        mlflow.log_metric(f"fold{fold} score threshold", threshold, step=epoch)
        mlflow.log_metric(f"fold{fold} tn", tn_best, step=epoch)
        mlflow.log_metric(f"fold{fold} fp", fp_best, step=epoch)
        mlflow.log_metric(f"fold{fold} fn", fn_best, step=epoch)
        mlflow.log_metric(f"fold{fold} tp", tp_best, step=epoch)
        mlflow.log_metric(f"fold{fold} auc", auc_score, step=epoch)