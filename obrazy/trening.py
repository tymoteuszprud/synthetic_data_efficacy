"""Interfejs eksperymentów ResNet-18 z kolejką i archiwum wyników."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obrazy.dane import read_manifest, split_from_manifest, manifest_digest
import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import os
import cv2
from PIL import Image
from collections import defaultdict
import json
import ast
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", message="I don't know how to infer vegalite type from 'empty'")
warnings.filterwarnings("ignore", message="`add_rows` is deprecated")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.transforms import v2
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


import threading
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx


GLOBAL_IMAGE_CACHE = {}
PRETRAIN_CACHE_DIR = "pretrain_cache"
if not os.path.exists(PRETRAIN_CACHE_DIR):
    os.makedirs(PRETRAIN_CACHE_DIR)

def _load_queue_from_disk():

    jobs = []
    if os.path.exists("job_queue.csv"):
        try:
            df_q = pd.read_csv("job_queue.csv")
            for _, row in df_q.iterrows():
                job = row.to_dict()
                for k in ["params", "data_sources", "cfg_mix", "aug_params"]:
                    if k in job and pd.notna(job[k]):
                        try:

                            val = job[k]
                            if isinstance(val, str):
                                try:
                                    job[k] = ast.literal_eval(val)
                                except:
                                    val_json = val.replace("'", '"').replace("False", "false").replace("True", "true")
                                    job[k] = json.loads(val_json)
                        except Exception as e:
                            print(f"Nie udało się sparsować pola {k}: {e}")
                jobs.append(job)
        except Exception as e:
            print(f"Błąd ładowania kolejki: {e}")
    return jobs


class HistoryManager:
    BASE_DIR = "experiments_archive"

    @classmethod
    def init_db(cls):
        if not os.path.exists(cls.BASE_DIR):
            os.makedirs(cls.BASE_DIR)

    @classmethod
    def save_experiment(cls, exp_data):
        run_id = exp_data["id"]
        exp_dir = os.path.join(cls.BASE_DIR, run_id)
        os.makedirs(exp_dir, exist_ok=True)


        model_state = exp_data.pop("best_model_state", None)
        if model_state:
            torch.save(model_state, os.path.join(exp_dir, "best_model.pth"))


        metrics = {
            "loss": exp_data.pop("metrics_df_loss"),
            "acc": exp_data.pop("metrics_df_acc"),
            "lr": exp_data.pop("metrics_df_lr")
        }
        for name, df in metrics.items():
            df.to_csv(os.path.join(exp_dir, f"metrics_{name}.csv"))


        with open(os.path.join(exp_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(exp_data, f, indent=4, default=str)


        exp_data["best_model_state"] = model_state
        exp_data["metrics_df_loss"] = metrics["loss"]
        exp_data["metrics_df_acc"] = metrics["acc"]
        exp_data["metrics_df_lr"] = metrics["lr"]

    @classmethod
    def load_all_experiments(cls):
        cls.init_db()
        experiments = []
        if not os.path.exists(cls.BASE_DIR): return []

        run_dirs = sorted(os.listdir(cls.BASE_DIR), reverse=True)
        for run_id in run_dirs:
            path = os.path.join(cls.BASE_DIR, run_id)
            meta_path = os.path.join(path, "metadata.json")
            if os.path.isdir(path) and os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        exp = json.load(f)


                    exp["metrics_df_loss"] = pd.read_csv(os.path.join(path, "metrics_loss.csv"), index_col=0) if os.path.exists(os.path.join(path, "metrics_loss.csv")) else pd.DataFrame()
                    exp["metrics_df_acc"] = pd.read_csv(os.path.join(path, "metrics_acc.csv"), index_col=0) if os.path.exists(os.path.join(path, "metrics_acc.csv")) else pd.DataFrame()
                    exp["metrics_df_lr"] = pd.read_csv(os.path.join(path, "metrics_lr.csv"), index_col=0) if os.path.exists(os.path.join(path, "metrics_lr.csv")) else pd.DataFrame()


                    model_path = os.path.join(path, "best_model.pth")
                    if os.path.exists(model_path):
                        exp["best_model_state"] = torch.load(model_path, map_location="cpu")

                    experiments.append(exp)
                except Exception as e:
                    print(f"Błąd wczytywania eksperymentu {run_id}: {e}")
        return experiments

HistoryManager.init_db()

@st.cache_resource
def get_global_state():

    return {
        "is_training": False, "stop_requested": False, "is_processing": False,
        "status_text": "Oczekujące na zadania...", "progress_pct": 0,
        "loss_df": pd.DataFrame(columns=["Train Loss", "Train Loss (Smooth)", "Val Loss"]),
        "acc_df": pd.DataFrame(columns=["Train Acc", "Train Acc (Smooth)", "Val Acc"]),
        "lr_df": pd.DataFrame(columns=["Learning Rate (FC)", " "]),
        "metrics": {}, "current_job": None,
        "job_queue": _load_queue_from_disk(),
        "experiments": HistoryManager.load_all_experiments(),
        "logs": [], "test_preds": [], "test_trues": [], "classes": [],
        "best_model_state": None, "last_test_paths": ([], [], []),
        "completed_jobs": [], "auto_scale": True,
    }

def reset_active_training_state():

    GLOBAL_STATE["loss_df"] = pd.DataFrame(columns=["Train Loss", "Train Loss (Smooth)", "Val Loss"])
    GLOBAL_STATE["acc_df"] = pd.DataFrame(columns=["Train Acc", "Train Acc (Smooth)", "Val Acc"])
    GLOBAL_STATE["lr_df"] = pd.DataFrame(columns=["Learning Rate (FC)", " "])
    GLOBAL_STATE["metrics"] = {}
    GLOBAL_STATE["progress_pct"] = 0
    GLOBAL_STATE["status_text"] = "Oczekujące na zadania..."
    GLOBAL_STATE["current_job"] = None
    GLOBAL_STATE["logs"] = []
    GLOBAL_STATE["test_preds"] = []
    GLOBAL_STATE["test_trues"] = []
    GLOBAL_STATE["best_model_state"] = None


    GLOBAL_STATE["stop_requested"] = False
    GLOBAL_STATE["is_processing"] = False

def stop_and_reset_active():

    GLOBAL_STATE["stop_requested"] = True
    GLOBAL_STATE["is_processing"] = False


    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reset_active_training_state()

def clear_pending_queue():

    GLOBAL_STATE["job_queue"] = []
    save_queue()

GLOBAL_STATE = get_global_state()

st.set_page_config(page_title="Sim-to-Real Lab", layout="wide", page_icon="🧪")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_results_to_session():

    st.session_state.logs = list(GLOBAL_STATE["logs"])
    st.session_state.test_preds = list(GLOBAL_STATE["test_preds"])
    st.session_state.test_trues = list(GLOBAL_STATE["test_trues"])
    st.session_state.classes = list(GLOBAL_STATE["classes"])
    st.session_state.best_model_state = GLOBAL_STATE["best_model_state"]
    st.session_state.last_test_paths = GLOBAL_STATE["last_test_paths"]

if "logs" not in st.session_state: st.session_state.logs = []
if "test_preds" not in st.session_state: st.session_state.test_preds, st.session_state.test_trues, st.session_state.classes = [], [], []
if "best_model_state" not in st.session_state: st.session_state.best_model_state = None
if "last_test_paths" not in st.session_state: st.session_state.last_test_paths = ([], [], [])


sync_results_to_session()


if "data_sources" not in st.session_state:
    st.session_state.data_sources = [
        {"id": 1, "path": "obrazy_rzeczywiste", "domain": "Real", "role": "Main", "is_val_test": True, "val_pct": 15, "test_pct": 15, "train_pct": 100},
        {"id": 2, "path": "obrazy_syntetyczne", "domain": "Synthetic", "role": "Main", "is_val_test": False, "val_pct": 0, "test_pct": 0, "train_pct": 100}
    ]

def set_seed(seed):

    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_queue():

    if len(GLOBAL_STATE["job_queue"]) > 0:
        pd.DataFrame(GLOBAL_STATE["job_queue"]).to_csv("job_queue.csv", index=False)
    elif os.path.exists("job_queue.csv"):
        os.remove("job_queue.csv")

def set_module_requires_grad(module, req_grad):

    for param in module.parameters():
        param.requires_grad = req_grad

def build_resnet18(num_classes, drop_rate, mults):


    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    set_module_requires_grad(model.conv1, mults['stem'] > 0)
    set_module_requires_grad(model.bn1, mults['stem'] > 0)
    set_module_requires_grad(model.layer1, mults['l1'] > 0)
    set_module_requires_grad(model.layer2, mults['l2'] > 0)
    set_module_requires_grad(model.layer3, mults['l3'] > 0)
    set_module_requires_grad(model.layer4, mults['l4'] > 0)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(num_ftrs, num_classes))
    return model.to(DEVICE)

def build_optimizer(model, p, mults, base_lr):

    param_groups = []

    if mults['stem'] > 0:
        stem_params = list(model.conv1.parameters()) + list(model.bn1.parameters())
        param_groups.append({'params': stem_params, 'lr': base_lr * mults['stem']})
    if mults['l1'] > 0: param_groups.append({'params': model.layer1.parameters(), 'lr': base_lr * mults['l1']})
    if mults['l2'] > 0: param_groups.append({'params': model.layer2.parameters(), 'lr': base_lr * mults['l2']})
    if mults['l3'] > 0: param_groups.append({'params': model.layer3.parameters(), 'lr': base_lr * mults['l3']})
    if mults['l4'] > 0: param_groups.append({'params': model.layer4.parameters(), 'lr': base_lr * mults['l4']})
    if mults['fc'] > 0: param_groups.append({'params': model.fc.parameters(), 'lr': base_lr * mults['fc']})

    if p["opt"] == "AdamW":
        return torch.optim.AdamW(param_groups, weight_decay=p["wd"], betas=(p["beta1"], p["beta2"]))
    elif p["opt"] == "SGD":
        return torch.optim.SGD(param_groups, momentum=p["momentum"], weight_decay=p["wd"])
    else:
        return torch.optim.Adam(param_groups, weight_decay=p["wd"], betas=(p["beta1"], p["beta2"]))

class RAMDataset(Dataset):


    def __init__(self, file_paths, labels, class_to_idx, aug_flags=None, aug_params=None, desc="Ładowanie"):
        self.images_tensor, self.labels, self.flags = None, [], []
        if not file_paths: return

        total = len(file_paths)
        if "GLOBAL_STATE" in globals():
            GLOBAL_STATE["status_text"] = f"{desc} (Przygotowywanie danych)..."

        raw_images = []
        paths_to_load = []
        idxs_to_load = []


        for idx, path in enumerate(file_paths):
            if path in GLOBAL_IMAGE_CACHE:
                raw_images.append(GLOBAL_IMAGE_CACHE[path])
                self.labels.append(class_to_idx[labels[idx]])
                self.flags.append(torch.tensor(aug_flags[idx], dtype=torch.uint8) if aug_flags else torch.tensor([0, 0], dtype=torch.uint8))
            else:
                paths_to_load.append(path)
                idxs_to_load.append(idx)


        if paths_to_load:
            def load_and_process(i):
                try:
                    p = paths_to_load[i]
                    if not os.path.exists(p) or os.path.getsize(p) == 0: return None
                    img_bgr = cv2.imread(p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    if img_bgr is None: return None
                    img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_np, (256, 256), interpolation=cv2.INTER_LINEAR)
                    tensor = torch.from_numpy(img_resized).permute(2, 0, 1)
                    return p, tensor
                except: return None

            max_workers = min(32, (os.cpu_count() or 4) * 2)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                loaded = list(executor.map(load_and_process, range(len(paths_to_load))))

            for res in loaded:
                if res is not None:
                    p, tensor = res
                    GLOBAL_IMAGE_CACHE[p] = tensor


            for idx in idxs_to_load:
                path = file_paths[idx]
                if path in GLOBAL_IMAGE_CACHE:
                    raw_images.append(GLOBAL_IMAGE_CACHE[path])
                    self.labels.append(class_to_idx[labels[idx]])
                    self.flags.append(torch.tensor(aug_flags[idx], dtype=torch.uint8) if aug_flags else torch.tensor([0, 0], dtype=torch.uint8))

        if raw_images:
            self.images_tensor = torch.stack(raw_images)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
            self.flags = torch.stack(self.flags)


            if torch.cuda.is_available():
                try:

                    ds_size_mb = (self.images_tensor.nelement()) / (1024 * 1024)
                    free_vram_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)

                    if ds_size_mb < (free_vram_mb * 0.6):
                        self.images_tensor = self.images_tensor.to(DEVICE)
                        self.labels = self.labels.to(DEVICE)
                        self.flags = self.flags.to(DEVICE)
                        if "GLOBAL_STATE" in globals():
                            GLOBAL_STATE["status_text"] = f"🚀 {desc}: Dataset w VRAM ({int(ds_size_mb)}MB)"
                except: pass

        if "GLOBAL_STATE" in globals():
            GLOBAL_STATE["status_text"] = f"✅ Gotowe: {desc} ({len(raw_images)} obrazów)"

    def __len__(self):
        return len(self.labels) if self.labels is not None else 0

    def __getitem__(self, idx):
        return self.images_tensor[idx], self.labels[idx], self.flags[idx]

    @property
    def is_on_gpu(self):

        return self.images_tensor.is_cuda if self.images_tensor is not None else False


@st.cache_data(ttl=600)
def count_source_images(path):

    if not os.path.exists(path): return -1
    valid_exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    count = 0
    try:
        for c_name in os.listdir(path):
            c_dir = os.path.join(path, c_name)
            if os.path.isdir(c_dir):
                for f in os.listdir(c_dir):
                    if f.endswith(valid_exts): count += 1
    except:
        return -1
    return count


@st.cache_data(ttl=3600)
def get_files_in_dir(path):

    if not os.path.exists(path): return []
    valid_exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    all_files = []
    try:
        for c_name in os.listdir(path):
            c_dir = os.path.join(path, c_name)
            if os.path.isdir(c_dir):
                files = [os.path.join(c_dir, f) for f in os.listdir(c_dir) if f.endswith(valid_exts)]
                for f in files:
                    all_files.append((c_name, f))
    except:
        pass
    return all_files

def extract_and_split_data(sources, configs, manifest_path="manifest_obrazow.json", seed=42, variant="B", budget=100):
    manifest = read_manifest(manifest_path)
    return split_from_manifest(sources, manifest, seed, variant, int(budget))


cfg_mix = None
val_test_transforms = v2.Compose([v2.CenterCrop(size=(224, 224)), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_pretrain_fingerprint(current_job, run_seed, num_classes):

    all_sources = sorted([
        {
            "path": s["path"], "domain": s["domain"], "role": s["role"],
            "is_val_test": s.get("is_val_test", False),
            "val_pct": s.get("val_pct", 0),
            "test_pct": s.get("test_pct", 0),
            "train_pct": s.get("train_pct", 100)
        }
        for s in current_job["data_sources"]
    ], key=lambda x: x["path"])

    p = current_job["params"]
    config = {
        "sources": all_sources,
        "manifest_digest": manifest_digest(current_job.get("manifest_path", "manifest_obrazow.json")),
        "seed": run_seed,
        "num_classes": num_classes,
        "lr": p.get("lr_pretrain", p["lr"]),
        "epochs": p["epochs"],
        "batch_size": p["batch_size"],
        "opt": p["opt"],
        "sched": p["sched"],
        "sched_p": p["sched_p"],
        "mults": p["mults"],
        "wd": p["wd"],
        "dropout": p["dropout"],
        "label_smoothing": p.get("label_smoothing", 0.0),
        "aug_params": current_job.get("aug_params") if current_job.get("use_augmentation") else None,
        "dataset_for_stats": current_job.get("dataset_for_stats")
    }
    import hashlib
    import json
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()

def load_job_into_form(job):
    st.session_state.form_variant = job.get('variant', 'B')
    st.session_state.form_budget = int(job.get('budget', 100))
    st.session_state.form_manifest = job.get('manifest_path', 'manifest_obrazow.json')
    st.session_state.input_exp_name = job["exp_id"]
    st.session_state.form_runs_count = job.get("runs_total", 1)
    st.session_state.form_base_seed = job.get("seed_base", 42)
    p = job["params"]
    st.session_state.form_epochs = p["epochs"]
    st.session_state.form_batch_size = p["batch_size"]
    st.session_state.form_lr = p["lr"]
    st.session_state.form_lr_pretrain = p.get("lr_pretrain", p["lr"])
    st.session_state.form_opt_type = p["opt"]
    st.session_state.form_sched_type = p["sched"]
    st.session_state.form_dropout_rate = int(p["dropout"] * 100)
    st.session_state.form_label_smoothing = p.get("label_smoothing", 0.1)
    st.session_state.data_sources = job["data_sources"]
    st.session_state.form_lr_mults = p["mults"]
    st.session_state.form_sched_params = p["sched_p"]
    st.session_state.form_use_mix = job["cfg_mix"].get("use_mix", False)
    st.session_state.form_mix_cfg = job["cfg_mix"] if isinstance(job["cfg_mix"], dict) else {"train": {"real_pct": 50}}
    st.session_state.form_use_aug = job["use_augmentation"]
    st.session_state.form_aug_params = job["aug_params"]
    st.session_state.form_use_norm = job.get("dataset_for_stats") is not None
    st.session_state.form_weight_decay = p.get("wd", 0.01)
    st.session_state.form_auto_save = job.get("auto_save", True)
    st.session_state.form_aggregate_results = job.get("aggregate_results", True)

def cb_edit_job(idx):
    job = GLOBAL_STATE["job_queue"].pop(idx)
    load_job_into_form(job)
    save_queue()

def cb_clone_job(idx):
    new_job = json.loads(json.dumps(GLOBAL_STATE["job_queue"][idx]))
    new_job["exp_id"] += "_copy"
    GLOBAL_STATE["job_queue"].insert(idx+1, new_job)
    save_queue()

def cb_move_job(idx, direction):
    if direction == "up" and idx > 0:
        GLOBAL_STATE["job_queue"].insert(idx-1, GLOBAL_STATE["job_queue"].pop(idx))
    save_queue()

def cb_delete_job(idx):
    GLOBAL_STATE["job_queue"].pop(idx)
    save_queue()

def cb_clone_from_history(exp):

    job = {
        "exp_id": f"{exp['job_id']}_clone",
        "runs_total": exp['params'].get('runs_total', 1),
        "seed_base": exp['params'].get('seed_base', 42),
        "params": exp['params'],
        "data_sources": exp['data_sources'],
        "cfg_mix": exp.get('cfg_mix', {"use_mix": False}),
        "use_augmentation": 'aug_params' in exp,
        "aug_params": exp.get('aug_params', {}),
        "dataset_for_stats": exp.get('dataset_for_stats')
    }
    load_job_into_form(job)

@st.fragment
def render_chart_controls():
    GLOBAL_STATE["auto_scale"] = st.checkbox("Automatycznie dopasuj skalę Y (ignoruj szczyty)", value=GLOBAL_STATE.get("auto_scale", True))

@st.fragment
def render_sidebar():
    global cfg_mix
    st.header("⚙️ Nowe Zadanie")


    if "input_exp_name" not in st.session_state: st.session_state.input_exp_name = f"Exp_{len(GLOBAL_STATE['logs'])+len(GLOBAL_STATE['job_queue'])+1:02d}"
    if "form_runs_count" not in st.session_state: st.session_state.form_runs_count = 1
    if "form_base_seed" not in st.session_state: st.session_state.form_base_seed = 42
    if "form_epochs" not in st.session_state: st.session_state.form_epochs = 35
    if "form_batch_size" not in st.session_state: st.session_state.form_batch_size = 64
    if "form_lr" not in st.session_state: st.session_state.form_lr = 0.001
    if "form_lr_pretrain" not in st.session_state: st.session_state.form_lr_pretrain = 0.001
    if "form_opt_type" not in st.session_state: st.session_state.form_opt_type = "AdamW"
    if "form_sched_type" not in st.session_state: st.session_state.form_sched_type = "OneCycleLR"
    if "form_dropout_rate" not in st.session_state: st.session_state.form_dropout_rate = 50
    if "form_lr_mults" not in st.session_state: st.session_state.form_lr_mults = {'stem': 1.0, 'l1': 1.0, 'l2': 1.0, 'l3': 1.0, 'l4': 1.0, 'fc': 1.0}
    if "form_sched_params" not in st.session_state: st.session_state.form_sched_params = {"pct_start": 0.3, "factor": 0.5, "patience": 5, "step_size": 10, "gamma": 0.1, "T_max": 25, "eta_min": 0.0}
    if "form_use_mix" not in st.session_state: st.session_state.form_use_mix = False
    if "form_mix_cfg" not in st.session_state: st.session_state.form_mix_cfg = {"train": {"real_pct": 50}, "val": {"real_pct": 100}, "test": {"real_pct": 100}}
    if "form_use_aug" not in st.session_state: st.session_state.form_use_aug = True
    if "form_aug_params" not in st.session_state: st.session_state.form_aug_params = {"crop_scale": 0.8, "rot_deg": 15, "jit_f": 0.2}
    if "form_use_norm" not in st.session_state: st.session_state.form_use_norm = True
    if "form_label_smoothing" not in st.session_state: st.session_state.form_label_smoothing = 0.1
    if "form_weight_decay" not in st.session_state: st.session_state.form_weight_decay = 0.01
    if "form_auto_save" not in st.session_state: st.session_state.form_auto_save = True
    if "form_aggregate_results" not in st.session_state: st.session_state.form_aggregate_results = True

    exp_name = st.text_input("Nazwa eksperymentu", key="input_exp_name")


    add_job_container = st.container()

    st.divider()

    tab_conf1, tab_conf2 = st.tabs(["📁 Dane", "🧠 Model"])

    with tab_conf1:
        with st.expander("📁 Menedżer Datasetów", expanded=True):
            for i, src in enumerate(st.session_state.data_sources):
                img_count = count_source_images(src["path"])
                status_icon = "✅" if img_count >= 0 else "❌"

                st.markdown(f"**Źródło {i+1}** {status_icon}")
                src["path"] = st.text_input("Ścieżka", value=src["path"], key=f"p_{i}")
                if img_count >= 0:
                    st.caption(f"Znaleziono obrazów: {img_count}")

                c1, c2 = st.columns(2)
                with c1: src["domain"] = st.selectbox("Typ", ["Real", "Synthetic"], index=0 if src["domain"]=="Real" else 1, key=f"d_{i}")
                with c2: src["role"] = st.selectbox("Rola", ["Main", "Pretrain"], index=0 if src["role"]=="Main" else 1, key=f"r_{i}")


                if src["role"] == "Main":
                    ivt = st.checkbox("Źródło walidacji i testu", value=src.get("is_val_test", False), key=f"ivt_{i}")
                    src["is_val_test"] = ivt
                    c1, c2, c3 = st.columns(3)
                    with c1: src["train_pct"] = st.number_input("TR %", 0, 1000, src.get("train_pct", 100), key=f"tp_{i}")
                    if ivt:
                        with c2: src["val_pct"] = st.number_input("VAL %", 0, 100, src.get("val_pct", 15), key=f"vp_{i}")
                        with c3: src["test_pct"] = st.number_input("TST %", 0, 100, src.get("test_pct", 15), key=f"tsp_{i}")

                if st.button("Usuń Źródło", key=f"del_{i}"):
                    st.session_state.data_sources.pop(i)
                    st.rerun()

            if st.button("➕ Dodaj Dataset"):
                st.session_state.data_sources.append({"id": random.randint(100,999), "path": "nowy_folder", "domain": "Real", "role": "Main", "split": [70, 15, 15]})
                st.rerun()

        with st.expander("⚖️ Kompozycja Faz (% Real / % Synth)", expanded=False):
            use_mix = st.checkbox("Wymuś specyficzny podział faz", value=st.session_state.form_use_mix)
            cfg_mix_val = st.session_state.form_mix_cfg if isinstance(st.session_state.form_mix_cfg, dict) else {"train": {"real_pct": 50}}
            c1_m, c2_m, c3_m = st.columns(3)
            with c1_m: tr_r = st.number_input("TR %R", 0, 100, cfg_mix_val.get("train", {}).get("real_pct", 50), disabled=not use_mix)
            cfg_mix = {
                "use_mix": use_mix,
                "train": {"real_pct": tr_r}
            }

        with st.expander("📊 Normalizacja & Augmentacje", expanded=False):
            use_normalization = st.checkbox("Normalizacja ImageNet", value=st.session_state.form_use_norm)
            use_augmentation = st.checkbox("Augmentacje Treningowe", value=st.session_state.form_use_aug)
            if use_augmentation:
                c1, c2 = st.columns(2)
                with c1: crop_scale = st.slider("Crop Scale Min", 0.1, 1.0, st.session_state.form_aug_params["crop_scale"])
                with c2: rot_deg = st.number_input("Rotation Deg", 0, 180, int(st.session_state.form_aug_params["rot_deg"]))
                jit_f = st.slider("Color Jitter", 0.0, 1.0, st.session_state.form_aug_params["jit_f"])
            else:
                crop_scale, rot_deg, jit_f = 0.8, 15, 0.2

            st.divider()

    with tab_conf2:
        with st.expander("🧠 Diff. Learning Rates", expanded=False):
            st.caption("Mnożniki LR (0.0 = zamrożone)")
            m_stem = st.slider("Stem", 0.0, 2.0, st.session_state.form_lr_mults['stem'], 0.01)
            m_l1 = st.slider("Layer 1", 0.0, 2.0, st.session_state.form_lr_mults['l1'], 0.01)
            m_l2 = st.slider("Layer 2", 0.0, 2.0, st.session_state.form_lr_mults['l2'], 0.01)
            m_l3 = st.slider("Layer 3", 0.0, 2.0, st.session_state.form_lr_mults['l3'], 0.01)
            m_l4 = st.slider("Layer 4", 0.0, 2.0, st.session_state.form_lr_mults['l4'], 0.01)
            m_fc = st.slider("FC (Głowa)", 0.0, 5.0, st.session_state.form_lr_mults['fc'], 0.1)
            lr_mults = {'stem': m_stem, 'l1': m_l1, 'l2': m_l2, 'l3': m_l3, 'l4': m_l4, 'fc': m_fc}

        with st.expander("🚀 Optymalizator & Sched.", expanded=True):
            c1, c2 = st.columns(2)
            with c1: epochs = st.number_input("Epoki", 1, 300, st.session_state.form_epochs)
            with c2: batch_size = st.number_input("Batch Size (większy = lepsze GPU %)", 1, 2048, st.session_state.form_batch_size, help="Dla kart 16GB VRAM i ResNet18 zalecane 128-512 dla wysokiej wydajności.")
            c_lr1, c_lr2 = st.columns(2)
            with c_lr1: lr = st.number_input("Main LR", value=st.session_state.form_lr, format="%.5f")
            with c_lr2: lr_pretrain = st.number_input("Pretrain LR", value=st.session_state.form_lr_pretrain, format="%.5f")
            wd = st.number_input("Weight Decay", value=st.session_state.form_weight_decay, format="%.5f")
            c1, c2 = st.columns(2)
            with c1: auto_save = st.checkbox("📁 Auto-Export", value=st.session_state.form_auto_save)
            with c2: agg_results = st.checkbox("📊 Agregacja", value=st.session_state.form_aggregate_results, help="Tworzy wspólny plik podsumowujący wszystkie runy.")

            opts = ["AdamW", "Adam", "SGD"]
            opt_type = st.selectbox("Optymalizator", opts, index=opts.index(st.session_state.form_opt_type) if st.session_state.form_opt_type in opts else 0)

            scheds = ["OneCycleLR", "ReduceLROnPlateau", "CosineAnnealingWarmRestarts", "CosineAnnealingLR", "StepLR"]
            sched_type = st.selectbox("Scheduler (Strategia LR)", scheds, index=scheds.index(st.session_state.form_sched_type) if st.session_state.form_sched_type in scheds else 0)

            with st.container(border=True):
                st.caption("Parametry dodatkowe")
                c_d, c_s = st.columns(2)
                with c_d: dropout_rate = st.slider("Dropout %", 0, 100, st.session_state.form_dropout_rate) / 100.0
                with c_s: label_smoothing = st.slider("Label Smoothing", 0.0, 0.5, st.session_state.form_label_smoothing, 0.01)

                runs_count = st.number_input("Powtórzenia (Runs)", 1, 10, st.session_state.form_runs_count)
                base_seed = st.number_input("Bazowy Seed", value=st.session_state.form_base_seed)

            with st.container(border=True):
                st.caption(f"Konfiguracja Schedulera")
                sched_params = {}
                fsp = st.session_state.form_sched_params
                if sched_type == "OneCycleLR":
                    st.info("Najszybsza zbieżność. Wymaga znajomości max_lr.")
                    sched_params["pct_start"] = st.slider("Faza wzrostu (pct_start)", 0.1, 0.5, fsp.get("pct_start", 0.3), help="Jaka część treningu to zwiększanie LR")
                    sched_params["div_factor"] = st.number_input("Div Factor", 1, 100, fsp.get("div_factor", 25), help="Początkowy LR = max_lr / div_factor")
                elif sched_type == "ReduceLROnPlateau":
                    st.info("Zmniejsza LR, gdy strata walidacyjna przestaje spadać.")
                    c1, c2 = st.columns(2)
                    with c1: sched_params["factor"] = st.slider("Mnożnik (factor)", 0.1, 0.9, fsp.get("factor", 0.5))
                    with c2: sched_params["patience"] = st.number_input("Cierpliwość (patience)", 1, 50, fsp.get("patience", 5))
                elif sched_type == "CosineAnnealingWarmRestarts":
                    st.info("Cykliczne zmiany LR z restartami.")
                    c1, c2 = st.columns(2)
                    with c1: sched_params["T_0"] = st.number_input("Okres (T_0)", 1, 100, fsp.get("T_0", 10))
                    with c2: sched_params["T_mult"] = st.number_input("Mnożnik Okresu (T_mult)", 1, 5, fsp.get("T_mult", 1))
                elif sched_type == "StepLR":
                    st.info("Stałe obniżanie co N epok.")
                    c1, c2 = st.columns(2)
                    with c1: sched_params["step_size"] = st.number_input("Co ile epok", 1, 100, fsp.get("step_size", 10))
                    with c2: sched_params["gamma"] = st.slider("Mnożnik (gamma)", 0.1, 0.9, fsp.get("gamma", 0.1))
                elif sched_type == "CosineAnnealingLR":
                    st.info("Stopniowe obniżanie LR do eta_min zgodnie z cosinusem.")
                    c1, c2 = st.columns(2)
                    with c1: sched_params["T_max"] = st.number_input("T_max (Epoki)", 1, 300, fsp.get("T_max", epochs))
                    with c2: sched_params["eta_min"] = st.number_input("eta_min", 0.0, 1.0, fsp.get("eta_min", 0.0), format="%.6f")
                elif sched_type == "ExponentialLR":
                    sched_params["gamma"] = st.slider("Gamma", 0.9, 1.0, fsp.get("gamma", 0.95), step=0.01)


    variant = st.selectbox('Wariant protokołu', list('ABCDE'), key='form_variant')
    budget = st.selectbox('Poziom budżetu', [1, 3, 10, 30, 100], key='form_budget')
    manifest_path = st.text_input('Manifest podziałów i referencji', value='manifest_obrazow.json', key='form_manifest')
    if add_job_container.button("➕ Dodaj do kolejki", use_container_width=True, type="primary"):

        final_cfg_mix = cfg_mix if (cfg_mix is not None and "train" in cfg_mix) else {
            "use_mix": False,
            "train": {"real_pct": 50}, "val": {"real_pct": 100}, "test": {"real_pct": 100}
        }

        job = {
            "variant": variant, "budget": budget, "manifest_path": manifest_path,
            "exp_id": exp_name, "runs_total": runs_count, "runs_done": 0, "seed_base": base_seed,
            "params": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "lr_pretrain": lr_pretrain, "wd": wd, "dropout": dropout_rate, "label_smoothing": label_smoothing, "opt": opt_type, "beta1": 0.9, "beta2": 0.999, "sched": sched_type, "sched_p": sched_params, "mults": lr_mults},
            "auto_save": auto_save, "aggregate_results": agg_results,
            "data_sources": [{k: v for k, v in src.items()} for src in st.session_state.data_sources],
            "cfg_mix": final_cfg_mix,
            "use_augmentation": use_augmentation,
            "aug_params": {
                "crop_scale": crop_scale, "rot_deg": rot_deg, "jit_f": jit_f,
            },
            "dataset_for_stats": "ImageNet" if use_normalization else None
        }
        GLOBAL_STATE["job_queue"].append(job)
        save_queue()
        st.toast(f"Zadanie {exp_name} dodane!")

    if GLOBAL_STATE["job_queue"] or GLOBAL_STATE.get("completed_jobs"):
        st.markdown("### Kolejka Zadań")
        for i, j in enumerate(GLOBAL_STATE["job_queue"]):
            cols = st.columns([0.6, 0.1, 0.1, 0.1, 0.1])
            cols[0].caption(f"⏳ {i+1}. **{j['exp_id']}** ({j['runs_done']}/{j['runs_total']})")

            if cols[1].button("🔼", key=f"up_{i}", disabled=i==0, on_click=cb_move_job, args=(i, "up")):
                pass

            if cols[2].button("✏️", key=f"editjob_{i}", help="Modyfikuj to zadanie (pobierz do formularza i usuń z kolejki)", on_click=cb_edit_job, args=(i,)):
                pass

            if cols[3].button("👯", key=f"clonejob_{i}", help="Klonuj to zadanie", on_click=cb_clone_job, args=(i,)):
                pass

            if cols[4].button("❌", key=f"deljob_{i}", on_click=cb_delete_job, args=(i,)):
                pass

        for i, j in enumerate(reversed(GLOBAL_STATE.get("completed_jobs", []))):
            st.caption(f"✅ **{j['exp_id']}** (Zakończone)")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Uruchom", type="primary", use_container_width=True, disabled=len(GLOBAL_STATE["job_queue"])==0 or GLOBAL_STATE["is_processing"]):
            GLOBAL_STATE["is_processing"] = True
            GLOBAL_STATE["stop_requested"] = False
            st.rerun()
    with c2:
        if st.button("⏹ Stop & Reset", use_container_width=True, help="Natychmiast przerywa aktywny trening i czyści widok."):
            stop_and_reset_active()
            st.rerun()

    if GLOBAL_STATE["job_queue"]:
        if st.button("🗑 Wyczyść Kolejkę", use_container_width=True):
            clear_pending_queue()
            st.rerun()

with st.sidebar:
    render_sidebar()



def training_worker():
    import time
    while True:
        if GLOBAL_STATE["stop_requested"]:
            GLOBAL_STATE["is_processing"] = False
            GLOBAL_STATE["stop_requested"] = False
            GLOBAL_STATE["is_training"] = False

        if GLOBAL_STATE["is_processing"] and len(GLOBAL_STATE["job_queue"]) > 0:
            if not GLOBAL_STATE["is_training"]:
                GLOBAL_STATE["is_training"] = True

            current_job = GLOBAL_STATE["job_queue"][0]
            GLOBAL_STATE["current_job"] = current_job
            p = current_job["params"]

            if current_job["runs_done"] < current_job["runs_total"]:
                run_idx = current_job["runs_done"]
                c_seed = current_job["seed_base"] + run_idx
                run_id = f"{current_job['exp_id']}_Run{run_idx+1}"

                GLOBAL_STATE["status_text"] = f"⚙️ Przetwarzam: {run_id} | Seed: {c_seed}"
                set_seed(c_seed)


                GLOBAL_STATE["loss_df"] = pd.DataFrame(columns=["Train Loss", "Train Loss (Smooth)", "Val Loss"])
                GLOBAL_STATE["acc_df"] = pd.DataFrame(columns=["Train Acc", "Train Acc (Smooth)", "Val Acc"])
                GLOBAL_STATE["lr_df"] = pd.DataFrame(columns=["Learning Rate (FC)", " "])
                GLOBAL_STATE["metrics"] = {}
                GLOBAL_STATE["progress_pct"] = 0
                GLOBAL_STATE["test_preds"], GLOBAL_STATE["test_trues"] = [], []

                use_augmentation = current_job.get("use_augmentation", True)
                aug_params = current_job.get("aug_params", {"crop_scale": 0.8, "rot_deg": 15, "jit_f": 0.2})

                dataset_for_stats = current_job.get("dataset_for_stats")
                use_normalization = dataset_for_stats is not None

                splits, class_names, class_to_idx = extract_and_split_data(current_job["data_sources"], current_job["cfg_mix"], current_job.get("manifest_path", "manifest_obrazow.json"), c_seed, current_job["variant"], current_job["budget"])
                GLOBAL_STATE["classes"] = class_names

                loaders = {}
                for phase in ["pretrain", "train", "val", "test"]:
                    if len(splits[phase][0]) > 0:
                        dataset = RAMDataset(splits[phase][0], splits[phase][1], class_to_idx, aug_flags=splits[phase][2], aug_params=aug_params, desc=f"Ładuję {phase}")
                        if len(dataset) > 0:
                            loaders[phase] = DataLoader(
                                dataset,
                                batch_size=p["batch_size"],
                                shuffle=(phase in ["pretrain", "train"]),
                                num_workers=0,
                                pin_memory=not dataset.is_on_gpu
                            )

                if "test" in loaders: GLOBAL_STATE["last_test_paths"] = splits["test"]

                model = build_resnet18(len(class_names), p["dropout"], p["mults"])
                criterion = nn.CrossEntropyLoss(label_smoothing=p.get("label_smoothing", 0.0))

                common_norm = [v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] if use_normalization else []

                if use_augmentation and aug_params:
                    train_transforms_custom = v2.Compose([
                        v2.RandomResizedCrop(size=(224, 224), scale=(aug_params["crop_scale"], 1.0), antialias=True),
                        v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation(degrees=aug_params["rot_deg"]),
                        v2.ColorJitter(brightness=aug_params["jit_f"], contrast=aug_params["jit_f"], saturation=aug_params["jit_f"]),
                        *common_norm
                    ])
                else:
                    train_transforms_custom = v2.Compose([v2.CenterCrop(size=(224, 224)), *common_norm])

                val_test_transforms_custom = v2.Compose([v2.CenterCrop(size=(224, 224)), *common_norm])

                def run_training_phase(phase_name, train_ldr, val_ldr, phase_lr, train_trans=None, val_trans=None, reset_metrics=True, index_offset=0):
                    GLOBAL_STATE["status_text"] = f"⚙️ {run_id} | {phase_name}"
                    if not train_ldr or len(train_ldr) == 0:
                        return False, []
                    optimizer = build_optimizer(model, p, p["mults"], phase_lr)

                    if p["sched"] == "OneCycleLR":
                        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g['lr'] for g in optimizer.param_groups], steps_per_epoch=len(train_ldr), epochs=p["epochs"], pct_start=p["sched_p"].get("pct_start", 0.3), div_factor=p["sched_p"].get("div_factor", 25))
                    elif p["sched"] == "ReduceLROnPlateau":
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=p["sched_p"].get("factor", 0.5), patience=p["sched_p"].get("patience", 5))
                    elif p["sched"] == "CosineAnnealingWarmRestarts":
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=p["sched_p"].get("T_0", 10), T_mult=p["sched_p"].get("T_mult", 1))
                    elif p["sched"] == "CosineAnnealingLR":
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p["sched_p"].get("T_max", p["epochs"]), eta_min=p["sched_p"].get("eta_min", 0.0))
                    elif p["sched"] == "StepLR":
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=p["sched_p"].get("step_size", 10), gamma=p["sched_p"].get("gamma", 0.1))
                    elif p["sched"] == "ExponentialLR":
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=p["sched_p"].get("gamma", 0.95))

                    if reset_metrics:
                        GLOBAL_STATE["loss_df"] = pd.DataFrame(columns=["Train Loss", "Train Loss (Smooth)", "Val Loss"])
                        GLOBAL_STATE["acc_df"] = pd.DataFrame(columns=["Train Acc", "Train Acc (Smooth)", "Val Acc"])
                        GLOBAL_STATE["lr_df"] = pd.DataFrame(columns=["Learning Rate (FC)", " "])

                    best_v_loss = float('inf')
                    patience_counter = 0
                    phase_records = []

                    alpha = 0.3
                    prev_t_loss_smooth, prev_t_acc_smooth = None, None
                    prev_t_loss, prev_v_loss = None, None
                    prev_t_acc, prev_v_acc = None, None

                    scaler = torch.amp.GradScaler('cuda')

                    for epoch in range(p["epochs"]):
                        if GLOBAL_STATE["stop_requested"]: break

                        r_loss = torch.tensor(0.0, device=DEVICE)
                        correct = torch.tensor(0, device=DEVICE)
                        total = 0
                        model.train()
                        for images, labels, flags in train_ldr:
                            if GLOBAL_STATE["stop_requested"]: break

                            images = images.to(DEVICE, dtype=torch.float32, non_blocking=True) / 255.0
                            labels = labels.to(DEVICE, non_blocking=True)
                            flags = flags.to(DEVICE, non_blocking=True)


                            if train_trans: images = train_trans(images)

                            optimizer.zero_grad(set_to_none=True)

                            with torch.amp.autocast('cuda'):
                                outputs = model(images)
                                loss = criterion(outputs, labels)

                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            if p["sched"] == "OneCycleLR": scheduler.step()


                            with torch.no_grad():
                                r_loss += loss.detach() * images.size(0)
                                _, preds = outputs.max(1)
                                correct += preds.eq(labels).sum()
                            total += labels.size(0)


                        t_loss, t_acc = r_loss.item() / total, (correct.item() / total) * 100
                        v_loss, v_acc = 0.0, 0.0

                        if val_ldr:
                            model.eval()
                            v_loss_sum = torch.tensor(0.0, device=DEVICE)
                            v_correct = torch.tensor(0, device=DEVICE)
                            v_total = 0
                            with torch.no_grad():
                                for images, labels, flags in val_ldr:
                                    if GLOBAL_STATE["stop_requested"]: break
                                    images = images.to(DEVICE, dtype=torch.float32) / 255.0
                                    if val_trans: images = val_trans(images)
                                    labels = labels.to(DEVICE)

                                    with torch.amp.autocast('cuda'):
                                        outputs = model(images)
                                        loss = criterion(outputs, labels)

                                    v_loss_sum += loss.detach() * images.size(0)
                                    _, v_preds = outputs.max(1)
                                    v_correct += v_preds.eq(labels).sum()
                                    v_total += labels.size(0)

                            v_loss, v_acc = v_loss_sum.item() / v_total, (v_correct.item() / v_total) * 100

                            if p["sched"] == "ReduceLROnPlateau": scheduler.step(v_loss)
                            if v_loss < best_v_loss:
                                best_v_loss, patience_counter = v_loss, 0
                                torch.save(model.state_dict(), f"best_model_{run_id}.pth")
                            else:
                                patience_counter += 1

                            torch.save(model.state_dict(), f"last_model_{run_id}.pth")

                        if p["sched"] in ["StepLR", "CosineAnnealingWarmRestarts", "CosineAnnealingLR", "ExponentialLR"]:
                            scheduler.step()

                        curr_lr = optimizer.param_groups[-1]['lr']

                        if prev_t_loss_smooth is None:
                            t_loss_smooth, t_acc_smooth = t_loss, t_acc
                        else:
                            t_loss_smooth = alpha * t_loss + (1 - alpha) * prev_t_loss_smooth
                            t_acc_smooth = alpha * t_acc + (1 - alpha) * prev_t_acc_smooth

                        if GLOBAL_STATE.get("auto_scale", True) and epoch > 0:
                            max_y_limit = max(0.5, prev_t_loss_smooth * 2)
                            plot_t_loss = min(t_loss, max_y_limit)
                            plot_t_loss_smooth = min(t_loss_smooth, max_y_limit)
                            plot_v_loss = min(v_loss, max_y_limit) if val_ldr else None
                        else:
                            plot_t_loss = t_loss
                            plot_t_loss_smooth = t_loss_smooth
                            plot_v_loss = v_loss if val_ldr else None

                        prev_t_loss_smooth, prev_t_acc_smooth = t_loss_smooth, t_acc_smooth

                        new_loss = pd.DataFrame([[plot_t_loss, plot_t_loss_smooth, plot_v_loss]], columns=["Train Loss", "Train Loss (Smooth)", "Val Loss"], index=[epoch + 1 + index_offset])
                        new_acc = pd.DataFrame([[t_acc, t_acc_smooth, v_acc if val_ldr else None]], columns=["Train Acc", "Train Acc (Smooth)", "Val Acc"], index=[epoch + 1 + index_offset])
                        new_lr = pd.DataFrame([[curr_lr, None]], columns=["Learning Rate (FC)", " "], index=[epoch + 1 + index_offset])

                        GLOBAL_STATE["loss_df"] = pd.concat([GLOBAL_STATE["loss_df"], new_loss])
                        GLOBAL_STATE["acc_df"] = pd.concat([GLOBAL_STATE["acc_df"], new_acc])
                        GLOBAL_STATE["lr_df"] = pd.concat([GLOBAL_STATE["lr_df"], new_lr])

                        GLOBAL_STATE["metrics"] = {
                            "t_loss": (t_loss, t_loss - prev_t_loss if prev_t_loss is not None else None),
                            "v_loss": (v_loss, v_loss - prev_v_loss if prev_v_loss is not None else None) if val_ldr else None,
                            "t_acc": (t_acc, t_acc - prev_t_acc if prev_t_acc is not None else None),
                            "v_acc": (v_acc, v_acc - prev_v_acc if prev_v_acc is not None else None) if val_ldr else None,
                            "lr": curr_lr
                        }

                        prev_t_loss, prev_v_loss = t_loss, v_loss
                        prev_t_acc, prev_v_acc = t_acc, v_acc

                        GLOBAL_STATE["progress_pct"] = int(((epoch + 1) / p["epochs"]) * 100)

                        if p["sched"] == "ReduceLROnPlateau" and patience_counter >= p["sched_p"]["patience"]:
                            break

                        record = {
                            "Run ID": run_id, "Phase": phase_name, "Epoch": epoch + 1 + index_offset,
                            "Train Loss": t_loss, "Train Acc": t_acc, "Val Loss": v_loss if val_ldr else None, "Val Acc": v_acc if val_ldr else None,
                            "LR": curr_lr, "Optimizer": p["opt"], "Base LR": phase_lr, "Batch Size": p["batch_size"], "WD": p["wd"], "Dropout": p["dropout"],
                            "Scheduler": p["sched"], "Seed": c_seed,
                            "Normalization": str(dataset_for_stats)
                        }
                        for k, v in p["sched_p"].items(): record[f"Sched_{k}"] = v
                        for k, v in p["mults"].items(): record[f"LR_Mult_{k}"] = v
                        for ph in ["train", "val", "test"]:
                            record[f"Mix_{ph}_Real%"] = current_job["cfg_mix"].get(ph, {}).get("real_pct", "N/A")
                        if use_augmentation and aug_params:
                            for k, v in aug_params.items(): record[f"Aug_{k}"] = v
                        record["Data Sources"] = " | ".join([f"{s['path']} ({s['domain']}, {s['role']}, VT:{s.get('is_val_test',False)} V:{s.get('val_pct',0)}% T:{s.get('test_pct',0)}%)" for s in current_job["data_sources"]])

                        phase_records.append(record)

                    if os.path.exists(f"best_model_{run_id}.pth"):
                        model.load_state_dict(torch.load(f"best_model_{run_id}.pth"))
                    return True, phase_records


                has_training_data = "train" in loaders or "pretrain" in loaders
                if not has_training_data:
                    GLOBAL_STATE["status_text"] = f"❌ {run_id}: Brak danych treningowych! Nieprawidłowe ścieżki katalogów."
                    GLOBAL_STATE["is_training"] = False
                    GLOBAL_STATE["is_processing"] = False
                    time.sleep(3)
                    continue

                success = True
                run_history = []
                if "pretrain" in loaders:
                    pre_hash = get_pretrain_fingerprint(current_job, c_seed, len(class_names))
                    cache_path = os.path.join(PRETRAIN_CACHE_DIR, f"{pre_hash}.pth")
                    log_cache_path = os.path.join(PRETRAIN_CACHE_DIR, f"{pre_hash}_logs.json")

                    if os.path.exists(cache_path):
                        GLOBAL_STATE["status_text"] = f"🚀 {run_id} | Wczytuję Pretrain z Cache..."
                        model.load_state_dict(torch.load(cache_path, map_location=DEVICE))
                        success = True
                        if os.path.exists(log_cache_path):
                            with open(log_cache_path, "r") as f:
                                phase_logs = json.load(f)

                            GLOBAL_STATE["loss_df"] = pd.DataFrame(columns=["Train Loss", "Train Loss (Smooth)", "Val Loss"])
                            GLOBAL_STATE["acc_df"] = pd.DataFrame(columns=["Train Acc", "Train Acc (Smooth)", "Val Acc"])
                            GLOBAL_STATE["lr_df"] = pd.DataFrame(columns=["Learning Rate (FC)", " "])

                            for log in phase_logs:
                                log["Run ID"] = run_id

                                ep = log["Epoch"]
                                if isinstance(ep, (int, float)):
                                    GLOBAL_STATE["loss_df"].loc[ep] = [log["Train Loss"], log["Train Loss"], log.get("Val Loss")]
                                    GLOBAL_STATE["acc_df"].loc[ep] = [log["Train Acc"], log["Train Acc"], log.get("Val Acc")]
                                    GLOBAL_STATE["lr_df"].loc[ep] = [log["LR"], None]
                        else:
                            phase_logs = [{"Run ID": run_id, "Phase": "Pretrain (Cached)", "Epoch": "N/A", "Train Acc": "N/A"}]
                    else:
                        success, phase_logs = run_training_phase("Pretrain", loaders["pretrain"], loaders.get("val"), p.get("lr_pretrain", p["lr"]), train_trans=train_transforms_custom, val_trans=val_test_transforms_custom, reset_metrics=True, index_offset=0)
                        if success:
                            torch.save(model.state_dict(), cache_path)
                            with open(log_cache_path, "w") as f:
                                json.dump(phase_logs, f)

                    run_history.extend(phase_logs)

                if success and "train" in loaders and not GLOBAL_STATE["stop_requested"]:
                    offset = len(GLOBAL_STATE["loss_df"])
                    success, phase_logs = run_training_phase("Fine-Tuning", loaders["train"], loaders.get("val"), p["lr"], train_trans=train_transforms_custom, val_trans=val_test_transforms_custom, reset_metrics=False, index_offset=offset)
                    run_history.extend(phase_logs)


                all_preds, all_trues = [], []
                test_acc_val = 0.0
                if "test" in loaders:
                    GLOBAL_STATE["status_text"] = f"🧪 {run_id} | Testowanie..."
                    model.eval()
                    if os.path.exists(f"best_model_{run_id}.pth"):
                        model.load_state_dict(torch.load(f"best_model_{run_id}.pth"))

                    t_correct, t_total = 0, 0
                    with torch.no_grad():
                        for images, labels, flags in loaders["test"]:
                            images = images.to(DEVICE, dtype=torch.float32) / 255.0
                            images = val_test_transforms_custom(images)
                            labels = labels.to(DEVICE)
                            outputs = model(images)
                            _, preds = outputs.max(1)
                            all_preds.extend(preds.cpu().numpy())
                            all_trues.extend(labels.cpu().numpy())
                            t_total += labels.size(0)
                            t_correct += preds.eq(labels).sum().item()
                    test_acc_val = t_correct / t_total if t_total > 0 else 0


                experiment_data = {
                    "id": run_id,
                    "job_id": current_job["exp_id"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "params": p,
                    "data_sources": current_job["data_sources"],
                    "manifest_path": current_job.get("manifest_path", "manifest_obrazow.json"),
                    "variant": current_job["variant"], "budget": current_job["budget"], "seed": c_seed,
                    "metrics_df_loss": GLOBAL_STATE["loss_df"].copy(),
                    "metrics_df_acc": GLOBAL_STATE["acc_df"].copy(),
                    "metrics_df_lr": GLOBAL_STATE["lr_df"].copy(),
                    "test_preds": list(all_preds),
                    "test_trues": list(all_trues),
                    "classes": list(class_names),
                    "best_model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "last_test_paths": GLOBAL_STATE["last_test_paths"],
                    "test_acc": test_acc_val,
                    "dataset_for_stats": current_job.get("dataset_for_stats")
                }


                HistoryManager.save_experiment(experiment_data)


                if current_job.get("auto_save", True):
                    try:

                        exp_base_dir = os.path.join("results", current_job["exp_id"])
                        export_dir = os.path.join(exp_base_dir, f"Run{run_idx+1}")
                        os.makedirs(export_dir, exist_ok=True)


                        df_loss = experiment_data["metrics_df_loss"]
                        df_acc = experiment_data["metrics_df_acc"]
                        df_lr = experiment_data["metrics_df_lr"]
                        df_all = pd.concat([df_loss, df_acc, df_lr], axis=1)
                        df_all.to_csv(os.path.join(export_dir, "training_metrics.csv"))


                        params_flat = {
                            "exp_id": current_job["exp_id"],
                            "run_id": run_id,
                            "seed": c_seed,
                            "epochs": p["epochs"],
                            "batch_size": p["batch_size"],
                            "lr": p["lr"],
                            "lr_pretrain": p.get("lr_pretrain", p["lr"]),
                            "wd": p["wd"],
                            "opt": p["opt"],
                            "sched": p["sched"],
                            "dropout": p["dropout"],
                            "label_smoothing": p["label_smoothing"]
                        }

                        for i, src in enumerate(current_job["data_sources"]):
                            prefix = f"src{i+1}_{src['domain']}_{src['role']}"
                            params_flat[f"{prefix}_is_val_test"] = src.get("is_val_test", False)
                            params_flat[f"{prefix}_val_pct"] = src.get("val_pct", 0)
                            params_flat[f"{prefix}_ts_pct"] = src.get("test_pct", 0)


                        for k, v in p["sched_p"].items(): params_flat[f"sched_{k}"] = v
                        pd.DataFrame([params_flat]).to_csv(os.path.join(export_dir, "config.csv"), index=False)


                        if os.path.exists(f"best_model_{run_id}.pth"):
                            import shutil
                            shutil.move(f"best_model_{run_id}.pth", os.path.join(export_dir, "best_model.pth"))
                        if os.path.exists(f"last_model_{run_id}.pth"):
                            import shutil
                            shutil.move(f"last_model_{run_id}.pth", os.path.join(export_dir, "last_model.pth"))


                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt

                        def save_plot(df, title, filename):
                            plt.figure(figsize=(10, 6))
                            for col in df.columns:
                                if "Smooth" not in col and df[col].notna().any():
                                    plt.plot(df.index, df[col], alpha=0.3)
                                    plt.plot(df.index, df[col].rolling(window=max(1, len(df)//10)).mean(), label=col, linewidth=2)
                                elif df[col].notna().any():
                                    plt.plot(df.index, df[col], label=col, linewidth=2)
                            plt.title(title)
                            plt.xlabel("Epoch")
                            plt.legend()
                            plt.grid(True)
                            plt.savefig(os.path.join(export_dir, filename))
                            plt.close()

                        save_plot(experiment_data["metrics_df_loss"], f"Loss - {run_id}", "loss_chart.png")
                        save_plot(experiment_data["metrics_df_acc"], f"Accuracy - {run_id}", "acc_chart.png")
                        save_plot(experiment_data["metrics_df_lr"], f"Learning Rate - {run_id}", "lr_chart.png")

                    except Exception as e:
                        print(f"Błąd podczas auto-exportu runu: {e}")

                GLOBAL_STATE["experiments"].append(experiment_data)

                GLOBAL_STATE["logs"].extend(run_history)
                current_job["runs_done"] += 1
                save_queue()


                if current_job["runs_done"] >= current_job["runs_total"] and current_job.get("aggregate_results", True):
                    try:
                        exp_id = current_job["exp_id"]
                        exp_base_dir = os.path.join("results", exp_id)
                        os.makedirs(exp_base_dir, exist_ok=True)


                        job_exps = [e for e in GLOBAL_STATE["experiments"] if e.get("job_id") == exp_id]
                        if job_exps:
                            summary_rows = []
                            for e in job_exps:
                                row = {
                                    "Run ID": e["id"],
                                    "Test Acc": e.get("test_acc", 0),
                                    "Best Val Loss": e["metrics_df_loss"]["Val Loss"].min() if not e["metrics_df_loss"].empty else None,
                                    "Best Val Acc": e["metrics_df_acc"]["Val Acc"].max() if not e["metrics_df_acc"].empty else None,
                                    "Final Train Loss": e["metrics_df_loss"]["Train Loss"].iloc[-1] if not e["metrics_df_loss"].empty else None,
                                    "Final Train Acc": e["metrics_df_acc"]["Train Acc"].iloc[-1] if not e["metrics_df_acc"].empty else None,
                                }
                                summary_rows.append(row)

                            df_summary = pd.DataFrame(summary_rows)


                            numeric_cols = df_summary.select_dtypes(include=[np.number]).columns
                            if not df_summary.empty:
                                stats_mean = df_summary[numeric_cols].mean().to_dict()
                                stats_mean["Run ID"] = "MEAN"
                                stats_std = df_summary[numeric_cols].std().to_dict()
                                stats_std["Run ID"] = "STD"
                                df_summary = pd.concat([df_summary, pd.DataFrame([stats_mean, stats_std])], ignore_index=True)

                            df_summary.to_csv(os.path.join(exp_base_dir, "experiment_summary_aggregated.csv"), index=False)


                            all_losses = []
                            for e in job_exps:
                                if not e["metrics_df_loss"].empty:
                                    df_l = e["metrics_df_loss"].copy()
                                    df_l["Run"] = e["id"]
                                    all_losses.append(df_l)

                            if all_losses:
                                df_curves = pd.concat(all_losses)

                                df_avg_curves = df_curves.groupby(df_curves.index).mean(numeric_only=True)
                                df_avg_curves.to_csv(os.path.join(exp_base_dir, "average_learning_curves.csv"))

                    except Exception as e:
                        print(f"Błąd podczas agregacji eksperymentu: {e}")


                del loaders, model, splits
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            if GLOBAL_STATE["stop_requested"] or current_job["runs_done"] >= current_job["runs_total"]:
                finished_job = GLOBAL_STATE["job_queue"].pop(0)
                if "completed_jobs" not in GLOBAL_STATE: GLOBAL_STATE["completed_jobs"] = []
                GLOBAL_STATE["completed_jobs"].append(finished_job)
                save_queue()

                GLOBAL_STATE["is_processing"] = False
                GLOBAL_STATE["is_training"] = False
                if GLOBAL_STATE["stop_requested"]:
                    GLOBAL_STATE["status_text"] = "🛑 Trening zatrzymany przez użytkownika."
                    GLOBAL_STATE["stop_requested"] = False
                else:
                    if len(GLOBAL_STATE["job_queue"]) > 0:
                        GLOBAL_STATE["is_processing"] = True
                    else:
                        GLOBAL_STATE["status_text"] = "🎉 Wszystkie zadania zakończone!"
        else:
            GLOBAL_STATE["is_training"] = False
        time.sleep(1)

@st.cache_resource
def start_worker():

    ctx = get_script_run_ctx()
    t = threading.Thread(target=training_worker, daemon=True)
    if ctx:
        add_script_run_ctx(t, ctx)
    t.start()
    return t

start_worker()

@st.fragment(run_every=1)
def render_live_dashboard():
    status = GLOBAL_STATE["status_text"]
    if "🎉" in status: st.success(status)
    elif "⚙️" in status: st.info(status)
    elif "❌" in status: st.error(status)
    else: st.warning(status)

    if GLOBAL_STATE["is_training"]:
        st.progress(GLOBAL_STATE["progress_pct"], text=f"Postęp epoki: {GLOBAL_STATE['progress_pct']}%")

    has_data = not GLOBAL_STATE["loss_df"].empty
    if has_data:
        st.divider()
        render_chart_controls()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("📉 Strata")
            st.line_chart(GLOBAL_STATE["loss_df"])
        with c2:
            st.subheader("🎯 Celność")
            st.line_chart(GLOBAL_STATE["acc_df"])
        with c3:
            st.subheader("⚡ LR")
            st.line_chart(GLOBAL_STATE["lr_df"])
    else:
        st.info("💡 Rozpocznij trening, aby zobaczyć wykresy na żywo.")


val_test_transforms_basic = v2.Compose([v2.CenterCrop(size=(224, 224))])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients, self.activations = None, None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class):
        self.model.eval()
        self.model.zero_grad()
        score = self.model(input_tensor)[0, target_class]
        score.backward()
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = F.relu(torch.sum(weights * self.activations, dim=1, keepdim=True)).squeeze().cpu().detach().numpy()
        cam = cv2.resize(cam, (224, 224))
        return (cam - np.min(cam)) / (np.max(cam) + 1e-8)

st.title("🧪 Laboratorium MLOps: Sim-to-Real")

tab_active, tab_archive = st.tabs(["🚀 Aktywny Trening", "📚 Archiwum Eksperymentów"])

with tab_active:
    c_left, c_right = st.columns([0.7, 0.3])

    with c_left:
        render_live_dashboard()

    with c_right:
        st.subheader("📋 Kolejka i Status")
        if GLOBAL_STATE["job_queue"]:
            for i, j in enumerate(GLOBAL_STATE["job_queue"]):
                st.caption(f"⏳ {i+1}. **{j['exp_id']}** ({j['runs_done']}/{j['runs_total']})")
        else:
            st.write("Brak zadań w kolejce.")

        st.divider()
        st.subheader("🎲 Sanity Check")
        if st.button("Losuj Próbki Panelu"):
            current_cfg_mix = cfg_mix if 'cfg_mix' in globals() and cfg_mix is not None else {"use_mix": False}
            splits_preview, _, _ = extract_and_split_data(st.session_state.data_sources, current_cfg_mix)
            for phase_name, (paths, labels, augs) in splits_preview.items():
                if paths:
                    st.write(f"**{phase_name.upper()}**")
                    idx = random.randint(0, len(paths)-1)
                    raw_np = np.array(Image.open(paths[idx]).convert('RGB').resize((150, 150)))
                    st.image(raw_np, caption=labels[idx])

with tab_archive:
    if not GLOBAL_STATE["experiments"]:
        st.info("Brak zakończonych eksperymentów w tej sesji.")
    else:
        exp_list = [f"{e['id']} ({e['timestamp']})" for e in GLOBAL_STATE["experiments"]]
        selected_exp_name = st.selectbox("Eksperyment do analizy", exp_list)
        selected_idx = exp_list.index(selected_exp_name)
        exp = GLOBAL_STATE["experiments"][selected_idx]

        st.header(f"📊 Wyniki: {exp['id']}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test Accuracy", f"{exp.get('test_acc', 0):.2%}")
        c2.metric("Epoki", exp['params']['epochs'])
        c3.metric("Batch", exp['params']['batch_size'])
        c4.metric("Main LR", exp['params']['lr'])
        c5.metric("Pretrain LR", exp['params'].get('lr_pretrain', 'N/A'))

        if st.button("👯 Klonuj parametry do formularza", key=f"clone_hist_{exp['id']}", on_click=cb_clone_from_history, args=(exp,)):
            st.toast("Parametry sklonowane do formularza!")


        def prepare_export_csv(e):

            meta = {
                "Experiment ID": e['id'],
                "Job ID": e['job_id'],
                "Timestamp": e['timestamp'],
                "Test Accuracy": e.get('test_acc', 0),
                "Classes": "|".join(e['classes'])
            }

            for k, v in e['params'].items():
                if isinstance(v, dict):
                    for subk, subv in v.items(): meta[f"Param_{k}_{subk}"] = subv
                else: meta[f"Param_{k}"] = v

            df_meta = pd.DataFrame([meta])


            df_loss = e['metrics_df_loss'].reset_index().rename(columns={'index': 'Epoch'})
            df_acc = e['metrics_df_acc'].reset_index().rename(columns={'index': 'Epoch'})
            df_lr = e['metrics_df_lr'].reset_index().rename(columns={'index': 'Epoch'})

            df_epochs = df_loss.merge(df_acc, on='Epoch').merge(df_lr, on='Epoch')


            output = "=== METADATA & PARAMETERS ===\n"
            output += df_meta.to_csv(index=False)
            output += "\n=== TRAINING METRICS (CHART DATA) ===\n"
            output += df_epochs.to_csv(index=False)

            if e['test_preds']:
                output += "\n=== TEST PREDICTIONS ===\n"
                df_test = pd.DataFrame({"True Label": e['test_trues'], "Predicted Label": e['test_preds']})
                output += df_test.to_csv(index=False)

            return output

        csv_data = prepare_export_csv(exp)
        st.download_button(
            label="📥 Pobierz Pełny Raport (CSV)",
            data=csv_data,
            file_name=f"report_{exp['id']}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()
        t1, t2, t3, t4 = st.tabs(["📈 Wykresy", "🧮 Macierz Pomyłek", "🔥 Grad-CAM", "🔮 Predykcje"])

        with t1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Strata")
                st.line_chart(exp['metrics_df_loss'])
            with col2:
                st.subheader("Dokładność")
                st.line_chart(exp['metrics_df_acc'])

        with t2:
            if exp['test_preds']:
                cm = confusion_matrix(exp['test_trues'], exp['test_preds'])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=exp['classes'], yticklabels=exp['classes'], ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Brak danych testowych dla tego eksperymentu.")

        with t3:
            st.subheader("Analiza Wyjaśnialności (Best Model)")
            if st.button("Generuj Grad-CAM dla 3 próbek", key=f"btn_gc_{exp['id']}"):
                paths, labels, _ = exp['last_test_paths']
                class_names = exp['classes']
                class_to_idx = {c: i for i, c in enumerate(class_names)}

                model_cam = build_resnet18(len(class_names), 0.0, {'stem':1,'l1':1,'l2':1,'l3':1,'l4':1,'fc':1})
                model_cam.load_state_dict(exp['best_model_state'])
                grad_cam = GradCAM(model_cam, model_cam.layer4[-1])

                use_norm = exp.get('dataset_for_stats') is not None
                cam_transforms = v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    *( [v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] if use_norm else [] )
                ])

                cols = st.columns(3)
                samples = random.sample(range(len(paths)), min(3, len(paths)))
                for i, s_idx in enumerate(samples):
                    img_path = paths[s_idx]
                    label_idx = class_to_idx[labels[s_idx]]
                    raw_pil = Image.open(img_path).convert('RGB').resize((224, 224))
                    raw_np = np.array(raw_pil)
                    tensor_img = cam_transforms(raw_pil).unsqueeze(0).to(DEVICE)
                    cam_mask = grad_cam.generate(tensor_img, target_class=label_idx)
                    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                    overlay = cv2.addWeighted(raw_np, 0.6, heatmap, 0.4, 0)
                    with cols[i]:
                        st.image(raw_np, caption=f"Prawda: {labels[s_idx]}")
                        st.image(overlay, caption="Grad-CAM")

        with t4:
            st.subheader("🔮 Przetestuj na własnych zdjęciach")
            uploaded_files = st.file_uploader("Przeciągnij i upuść zdjęcia", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key=f"uploader_{exp['id']}")

            if uploaded_files:
                class_names = exp['classes']
                model_pred = build_resnet18(len(class_names), 0.0, {'stem':1,'l1':1,'l2':1,'l3':1,'l4':1,'fc':1})
                model_pred.load_state_dict(exp['best_model_state'])
                model_pred.eval()

                use_norm = exp.get('dataset_for_stats') is not None
                inference_transforms = v2.Compose([
                    v2.Resize((224, 224)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    *( [v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] if use_norm else [] )
                ])

                for i, file in enumerate(uploaded_files):
                    with st.container(border=True):
                        c1, c2 = st.columns([0.4, 0.6])
                        img = Image.open(file).convert('RGB')
                        input_tensor = inference_transforms(img).unsqueeze(0).to(DEVICE)

                        with torch.no_grad():
                            logits = model_pred(input_tensor)
                            probs = F.softmax(logits, dim=1)[0]

                        with c1:
                            st.image(img, use_container_width=True)

                        with c2:
                            st.write("**Prawdopodobieństwa klas:**")
                            prob_dict = {class_names[j]: float(probs[j]) for j in range(len(class_names))}
                            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

                            for cls, p in sorted_probs:
                                st.progress(p, text=f"{cls}: {p:.2%}")
