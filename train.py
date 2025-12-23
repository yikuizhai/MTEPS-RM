#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import trange
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score
from model import UNet
import tool.dataset as d
import random
import warnings
from skimage.segmentation import slic
from skimage.util import img_as_float
from collections import OrderedDict
import gc


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==================== 工具函数 ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='M2PS training with module-stripped checkpoint')
    parser.add_argument('-d', '--Dataset_name', default="CDD", type=str)
    parser.add_argument('-t', '--train_ratio', default=0.1, type=float)
    parser.add_argument('--std_per_teacher', default=2, type=int)
    parser.add_argument('--tea_model_num', default=2, type=int)
    parser.add_argument('-g', '--gpus', default='0', type=str)
    parser.add_argument('--epoch_num', default=100, type=int)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_test', default=4, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--mc_dropout_T', default=4, type=int)
    parser.add_argument('--target_segments', default=200, type=int)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--rotate_interval', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mc_cpu_offload', action='store_true', default=False,
                        help="MC Dropout outputs offload to CPU to save GPU memory")
    return parser.parse_args()

# ---------------- MC Dropout ----------------
def enable_mc_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout_predict(model, x1, x2, T=10, device='cuda', half=False, cpu_offload=False):
    """单模型 MC Dropout，支持半精度与 CPU offload"""
    model.eval()
    enable_mc_dropout(model)
    mean_pred = 0
    var_pred = 0
    with torch.no_grad():
        for t in range(T):
            with autocast(enabled=half):
                out = torch.softmax(model(x1, x2), dim=1)
            if cpu_offload:
                out = out.cpu()
            if t == 0:
                mean_pred = out
                var_pred = torch.zeros_like(out)
            else:
                delta = out - mean_pred
                mean_pred += delta / (t + 1)
                var_pred += delta * (out - mean_pred)
        var_pred /= (T - 1 + 1e-8)
    uncertainty = var_pred[:, 1, :, :]  # 假设 class 1 为变化
    return mean_pred, uncertainty

def student_group_mc_dropout_predict(group_models, x1, x2, T=10, device='cuda', half=False, cpu_offload=False):
    """学生组 MC Dropout，分批计算，节省显存"""
    mean_preds = None
    uncertainties = None
    for idx, model in enumerate(group_models):
        mean_pred, uncertainty = mc_dropout_predict(model, x1, x2, T, device, half, cpu_offload)
        if mean_preds is None:
            mean_preds = mean_pred
            uncertainties = uncertainty
        else:
            mean_preds += mean_pred
            uncertainties += uncertainty
    mean_preds /= len(group_models)
    uncertainties /= len(group_models)
    return mean_preds, uncertainties

# ---------------- Superpixel Soft Smoothing ----------------
_SUPERPIXEL_CACHE = OrderedDict()
_SUPERPIXEL_CACHE_MAX = 256  # 可调整，内存大可以调高，建议先 256

def superpixel_soft_smoothing(pseudo_label, target_segments=200, sigma=0.5):
    if target_segments <= 0:
        return pseudo_label
    B, H, W = pseudo_label.shape
    smoothed = torch.zeros_like(pseudo_label)
    for b in range(B):
        arr = pseudo_label[b].detach().cpu().numpy()
        key = hash(arr.tobytes())
        if key in _SUPERPIXEL_CACHE:
            segments = _SUPERPIXEL_CACHE[key]
            # 把这个 key 移到末尾，表示最近使用（LRU）
            _SUPERPIXEL_CACHE.move_to_end(key)
        else:
            img = img_as_float(arr)
            segments = slic(img, n_segments=target_segments, start_label=0, channel_axis=None)
            # 插入缓存，保持 LRU 上限
            _SUPERPIXEL_CACHE[key] = segments
            if len(_SUPERPIXEL_CACHE) > _SUPERPIXEL_CACHE_MAX:
                _SUPERPIXEL_CACHE.popitem(last=False)  # 弹出最老的
        for sid in np.unique(segments):
            mask = segments == sid
            region = arr[mask]
            if region.size == 0:
                continue
            weight = np.exp(-((region - region.mean()) ** 2) / (2 * sigma ** 2))
            val = float(np.sum(region * weight) / (np.sum(weight) + 1e-8))
            smoothed[b][mask] = val
    return smoothed.to(pseudo_label.device)


# ---------------- Rotate 教师列表 ----------------
def rotate_teachers(teachers, epoch, rotate_interval=2):
    n = len(teachers)
    if n <= 1:
        return teachers
    shift = (epoch // rotate_interval) % n
    if shift == 0:
        return teachers
    return teachers[-shift:] + teachers[:-shift]

# ---------------- 评估函数 ----------------
def teacher_group_predict(tea_models, x1, x2):
    preds = [torch.softmax(m(x1, x2), dim=1) for m in tea_models]
    return torch.mean(torch.stack(preds, dim=0), dim=0)

def evaluate(test_loader, model_group, device):
    all_preds, all_labels = [], []
    for m in model_group:
        m.eval()
    with torch.no_grad():
        for i1, i2, label, _, _ in test_loader:
            i1, i2, label = i1.to(device), i2.to(device), label.to(device)
            pred = teacher_group_predict(model_group, i1, i2)
            pred_label = pred.argmax(dim=1).cpu().numpy()
            gt_label = label.cpu().numpy()
            all_preds.append(pred_label)
            all_labels.append(gt_label)
    all_preds = np.concatenate(all_preds).flatten().astype(np.int32)
    all_labels = np.concatenate(all_labels).flatten().astype(np.int32)
    return (precision_score(all_labels, all_preds),
            recall_score(all_labels, all_preds),
            f1_score(all_labels, all_preds),
            jaccard_score(all_labels, all_preds),
            accuracy_score(all_labels, all_preds))

# ==================== Helper ====================
def get_state_dict_clean(model):
    """统一去掉 DataParallel 的 module 前缀"""
    if isinstance(model, DataParallel):
        return {k: v.cpu() for k, v in model.module.state_dict().items()}
    else:
        return {k: v.cpu() for k, v in model.state_dict().items()}
# ==================== 主函数 ====================
def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志路径
    log_root = f'Run_logging_{args.Dataset_name}_{str(args.train_ratio).replace(".", "_")}/'
    Log_path = os.path.join(log_root, f'{args.Dataset_name}_{args.std_per_teacher}std_{args.tea_model_num}tea/')
    os.makedirs(Log_path, exist_ok=True)
    log_file = os.path.join(Log_path, 'train_log.txt')
    checkpoint_path = os.path.join(Log_path, 'last_checkpoint.pth')

    # 数据路径
    DATA_PATH = {
        'LEVIR': r'D:\DL\Dataset\LEVIR-CD',
        'CDD': r'D:\DL\Dataset\CDD',
        'SYSU': r'D:\DL\Dataset\SYSU-CD',
        'UAV': r'D:\DL\Dataset\UAV-CD+_256\UAV-CD3.0_256',
        'DSIFN': r'D:\DL\Dataset\DSIFN'
    }.get(args.Dataset_name)
    if DATA_PATH is None:
        raise ValueError("未知数据集名")

    TRAIN_w_TXT_PATH = f'super_txt/{args.Dataset_name}{args.train_ratio}/wlabel.txt'
    TRAIN_o_TXT_PATH = f'super_txt/{args.Dataset_name}{args.train_ratio}/olabel.txt'
    TEST_TXT_PATH = os.path.join(DATA_PATH, 'test.txt')

    # 数据加载
    wtrain_data = d.Dataset(DATA_PATH, DATA_PATH, TRAIN_w_TXT_PATH, 'train', transform=True)
    otrain_data = d.Dataset(DATA_PATH, DATA_PATH, TRAIN_o_TXT_PATH, 'train', transform=True)
    test_data = d.Dataset(DATA_PATH, DATA_PATH, TEST_TXT_PATH, 'test', transform=False)

    wtrain_loader = DataLoader(wtrain_data, batch_size=args.batch_size_train, shuffle=True,
                               num_workers=args.num_workers, pin_memory=args.pin_memory)
    otrain_loader = DataLoader(otrain_data, batch_size=args.batch_size_train, shuffle=True,
                               num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.pin_memory)

    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()

    # 模型初始化
    tea_models = [UNet(3, 2).to(device) for _ in range(args.tea_model_num)]
    std_groups = [[UNet(3, 2).to(device) for _ in range(args.std_per_teacher)]
                  for _ in range(args.tea_model_num)]

    if torch.cuda.device_count() > 1:
        tea_models = [DataParallel(m) for m in tea_models]
        for i in range(len(std_groups)):
            std_groups[i] = [DataParallel(m) for m in std_groups[i]]

    # 优化器
    all_std_params = [p for group in std_groups for m in group for p in m.parameters()]
    all_tea_params = [p for m in tea_models for p in m.parameters()]
    opt_std = optim.Adam(all_std_params, lr=1e-4, betas=(0.9, 0.999))
    opt_tea = optim.Adam(all_tea_params, lr=1e-4, betas=(0.9, 0.999))
    scaler = GradScaler()

    # resume
    start_epoch = 0
    best_f1 = 0.0
    if args.resume and os.path.exists(checkpoint_path):
        print("🔁 Resume training from checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        start_epoch = ckpt['epoch'] + 1
        best_f1 = ckpt.get('best_f1', 0.0)
        for i, m in enumerate(tea_models):
            try:
                m.load_state_dict(ckpt['tea_model_states'][i])
            except RuntimeError:
                new_state = {}
                for k, v in ckpt['tea_model_states'][i].items():
                    if k.startswith('module.'):
                        new_state[k[len('module.'):]] = v
                    else:
                        new_state['module.' + k] = v
                m.load_state_dict(new_state)
        for i, group in enumerate(std_groups):
            for j, m in enumerate(group):
                try:
                    m.load_state_dict(ckpt['std_model_states'][i][j])
                except RuntimeError:
                    new_state = {}
                    for k, v in ckpt['std_model_states'][i][j].items():
                        if k.startswith('module.'):
                            new_state[k[len('module.'):]] = v
                        else:
                            new_state['module.' + k] = v
                    m.load_state_dict(new_state)
        opt_std.load_state_dict(ckpt['opt_std'])
        opt_tea.load_state_dict(ckpt['opt_tea'])
        print(f"Resumed at epoch {start_epoch}, best F1: {best_f1:.4f}")

    from itertools import cycle

    # ---------------- training loop ----------------
    for epoch in range(start_epoch, args.epoch_num):
        tea_models = rotate_teachers(tea_models, epoch, rotate_interval=args.rotate_interval)
        for m in tea_models:
            m.train()
        for group in std_groups:
            for m in group:
                m.train()

        dataloader = iter(zip(cycle(wtrain_loader), otrain_loader))
        total_loss_epoch = 0.0

        with trange(len(otrain_loader), desc=f"Epoch {epoch}") as t:
            for _ in t:
                (i1_w, i2_w, label_w, _, _), (i1_o, i2_o, _, _, _) = next(dataloader)
                i1_w, i2_w, label_w = i1_w.to(device), i2_w.to(device), label_w.to(device).long()
                i1_o, i2_o = i1_o.to(device), i2_o.to(device)

                for tea, std_group in zip(tea_models, std_groups):
                    # ---------- 前向全程半精度 ----------
                    with autocast():
                        # 有监督
                        pred_w_std = [m(i1_w, i2_w) for m in std_group]
                        pred_w_std_cat = torch.cat(pred_w_std, dim=0)
                        label_w_expanded = label_w.expand(len(std_group), -1, -1, -1).reshape(-1, *label_w.shape[1:])
                        pred_w_tea = tea(i1_w, i2_w)
                        loss_w = (CE(pred_w_std_cat, label_w_expanded) +
                                  CE(pred_w_tea, label_w))
                    # ---------- 无监督 ----------
                    if args.mc_dropout_T > 0:
                        stu_ensemble, uncertainty = student_group_mc_dropout_predict(
                            std_group, i1_o, i2_o, args.mc_dropout_T,
                            device=device, half=True, cpu_offload=args.mc_cpu_offload)
                    else:
                        preds = [torch.softmax(m(i1_o, i2_o), dim=1) for m in std_group]
                        stu_ensemble = torch.mean(torch.stack(preds, dim=0), dim=0)
                        uncertainty = torch.zeros_like(stu_ensemble[:, 0, :, :])

                    refined_label = superpixel_soft_smoothing(stu_ensemble[:, 1, :, :], args.target_segments)
                    tea_cluster_pred = teacher_group_predict(tea_models, i1_o, i2_o)
                    tea_hard = tea_cluster_pred.argmax(dim=1).float()

                    # 无监督 loss
                    pred_o_stds = []
                    for m in std_group:
                        with torch.no_grad():
                            p = torch.softmax(m(i1_o, i2_o), dim=1)
                            if args.mc_dropout_T > 0 and args.mc_cpu_offload:
                                p = p.cpu()
                            pred_o_stds.append(p)
                    pred_o_stds = torch.stack(pred_o_stds).to(device)
                    student_preds = pred_o_stds[:, :, 1, :, :]
                    tea_pred_class1 = tea(i1_o, i2_o)[:, 1, :, :].unsqueeze(0)
                    pred_o = torch.cat([tea_pred_class1, student_preds], dim=0)
                    label_o = torch.cat([refined_label.unsqueeze(0),
                                         tea_hard.unsqueeze(0).expand(len(std_group), *refined_label.shape)], dim=0)
                    alpha = torch.exp(-uncertainty)
                    weights = torch.cat([torch.ones_like(refined_label).unsqueeze(0),
                                         alpha.unsqueeze(0).expand(len(std_group), *alpha.shape)], dim=0)

                    # ---------- 计算 loss ----------
                    with autocast():
                        loss_o = ((pred_o - label_o) ** 2 * weights).mean().float()
                        loss = loss_w + loss_o

                    scaler.scale(loss).backward()
                    total_loss_epoch += loss.item()

                # ---------- 优化器 step ----------
                scaler.step(opt_std)
                scaler.step(opt_tea)
                scaler.update()
                opt_std.zero_grad()
                opt_tea.zero_grad()
                t.set_postfix(loss=round(total_loss_epoch / (t.n + 1e-6), 4))

                # 每 100 个 step 做一次强回收
                if t.n % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()


        # ---------------- 评估 ----------------
        best_teacher_idx = -1
        best_teacher_f1 = -1
        best_metrics = None
        for i, tea_model in enumerate(tea_models):
            precision, recall, f1, iou, oa = evaluate(test_loader, [tea_model], device)
            if f1 > best_teacher_f1:
                best_teacher_f1 = f1
                best_teacher_idx = i
                best_metrics = (precision, recall, f1, iou, oa)

        precision, recall, f1, iou, oa = best_metrics
        log_line = (f"Epoch {epoch} | Best_Teacher {best_teacher_idx} | "
                    f"Precision {precision:.4f} | Recall {recall:.4f} | "
                    f"F1 {f1:.4f} | IoU {iou:.4f} | OA {oa:.4f}")
        print(log_line)
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

        # ---------------- 保存最佳教师及学生 ----------------
        if f1 > best_f1:
            best_f1 = f1
            torch.save(get_state_dict_clean(tea_models[best_teacher_idx]),
                       os.path.join(Log_path, f"Best_teacher_model.pth"))
            for j, m in enumerate(std_groups[best_teacher_idx]):
                torch.save(get_state_dict_clean(m),
                           os.path.join(Log_path, f"Best_student_g{best_teacher_idx}_s{j}.pth"))

        # ---------------- checkpoint ----------------
        torch.save({
            'epoch': epoch,
            'best_f1': best_f1,
            'tea_model_states': [get_state_dict_clean(m) for m in tea_models],
            'std_model_states': [[get_state_dict_clean(m) for m in group] for group in std_groups],
            'opt_std': opt_std.state_dict(),
            'opt_tea': opt_tea.state_dict(),
        }, checkpoint_path)

    print("✅ 训练完成，日志与模型已保存。")

# ==================== 入口 ====================
if __name__ == "__main__":
    args = parse_args()
    print("\n" + "=" * 30 + " 超参数配置 " + "=" * 30)
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("=" * 80 + "\n")
    set_seed(args.seed)
    main()
