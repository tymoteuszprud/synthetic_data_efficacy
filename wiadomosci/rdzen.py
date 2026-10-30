"""Jeden rdzeń treningu i oceny, z opcjonalnym losowaniem klas.

Oba tryby korzystają z procedury oceny podstawowego eksperymentu: porównania
pierwszych tokenów etykiet, torch.no_grad i średniej strat partii. Końcowa
partia oceny odpowiada partii treningowej, walidacyjna klasyfikacja ma rozmiar 8.
Losowanie zmienia wyłącznie wystąpienia w danych treningowych. Domyślnie A ma
dwie kolejne epoki (syntetyczna i rzeczywista), a B-E po jednej.
"""

import os
from pathlib import Path
import sys
import gc
import re
import csv
import json
import time
import shutil
import random
import numpy as np
import pandas as pd
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import TrainerCallback

MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"

REAL_DATA_FILE = "wiadomosci_rzeczywiste.csv"

SYNTH_DATA_FILE = "wiadomosci_syntetyczne.csv"

EXTRACTED_INDICES_FILE = "indeksy_referencji.csv"

RESULTS_FILE = "experiment_results.json"
BALANCED = False
_RUN_FINGERPRINT = ""
_RUN_PROVENANCE = {}
_CACHE_ROOT = "."
_EXPECTED_PLAN = None

BATCH_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 2

MAX_SEQ_LENGTH = 512

EPOCHS_STANDARD = 1

LR_STANDARD = 2e-5

EPOCHS_PRETRAIN = 1

LR_PRETRAIN = 2e-5

EPOCHS_FINETUNE = 1

LR_FINETUNE = 5e-6

SYSTEM_PROMPT = "You are a precise spam classification AI. Your only task is to classify the provided SMS message as 'SPAM' or 'HAM'."

# Format wejścia pozostaje zgodny z treningiem klasyfikatora.
def format_chatml_prompts(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []

    for instruction, output in zip(instructions, outputs):
        text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        text += f"<|im_start|>user\n{instruction}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)

    return {"text": texts}

def prepare_hf_dataset(df, tokenizer):
    dataset_dicts = []
    for _, row in df.iterrows():
        dataset_dicts.append({
            "instruction": str(row["message"]),
            "output": str(row["label"]).upper()
        })
    dataset = Dataset.from_list(dataset_dicts)
    dataset = dataset.map(format_chatml_prompts, batched=True)
    return dataset, dataset_dicts

# Referencje są wyłączone z losowania walidacji i testu.
from .dane import split_real_dataset

from .dane import get_training_subset, prepare_training_stages, pretrain_cache_path
from .balansowanie import save_manifest


def cache_directory(seed):
    return pretrain_cache_path(_CACHE_ROOT, _RUN_FINGERPRINT, seed, BALANCED)


def remove_cache_directory(path):
    root = (Path(_CACHE_ROOT) / 'pretrain_cache').resolve()
    target = Path(path).resolve()
    if root not in target.parents:
        raise ValueError('Pamięć modelu znajduje się poza katalogiem wyników.')
    shutil.rmtree(target)


def prepare_experiment_data(df_real, split_info, df_synth, variant, seed, percentage):
    subset = get_training_subset(df_real, split_info, percentage, seed)
    return prepare_training_stages(subset, df_synth, variant, seed, percentage, BALANCED)


def verify_training_plan(training, pretrain, sampling, pretrain_sampling):
    if _EXPECTED_PLAN is None:
        return
    if training['_sample_id'].tolist() != _EXPECTED_PLAN['train_rows']:
        raise ValueError('Próbki treningowe nie zgadzają się z audytem.')
    if sampling != _EXPECTED_PLAN['sampling']:
        raise ValueError('Losowanie klas nie zgadza się z audytem.')
    if pretrain is not None:
        if pretrain['_sample_id'].tolist() != _EXPECTED_PLAN['pretrain_rows']:
            raise ValueError('Próbki uczenia wstępnego nie zgadzają się z audytem.')
        if pretrain_sampling != _EXPECTED_PLAN['pretrain_sampling']:
            raise ValueError('Losowanie uczenia wstępnego nie zgadza się z audytem.')

def get_experiment_train_size(variant, seed, percentage, df_real, split_info, df_synth):
    df_train_real = get_training_subset(df_real, split_info, percentage, seed)
    if variant == "A":
        cache_dir = str(cache_directory(seed))
        history_path = os.path.join(cache_dir, "pretrain_history.json")
        if os.path.exists(cache_dir) and os.path.exists(history_path):
            return len(df_train_real) * EPOCHS_FINETUNE
        else:
            return len(df_synth) * EPOCHS_PRETRAIN + len(df_train_real) * EPOCHS_FINETUNE
    elif variant == "B":
        return len(df_train_real) * EPOCHS_STANDARD
    elif variant == "C":
        return len(df_train_real) * 2 * EPOCHS_STANDARD
    elif variant == "D":
        return (len(df_train_real) + len(df_synth)) * EPOCHS_STANDARD
    elif variant == "E":
        return len(df_train_real) * EPOCHS_STANDARD
    return 0

# Decyzja porównuje pierwsze tokeny etykiet HAM i SPAM.
def evaluate_classification(model, tokenizer, dataset_dict_list, batch_size=8):
    was_training = model.training
    model.eval()
    correct = 0
    total = len(dataset_dict_list)


    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    predictions = []
    targets = []
    logits_list = []


    prompts = []
    for item in dataset_dict_list:
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)
        targets.append(item['output'].strip().upper())

    original_padding_side = actual_tokenizer.padding_side
    actual_tokenizer.padding_side = "left"


    spam_token_id = actual_tokenizer.encode("SPAM", add_special_tokens=False)[0]
    ham_token_id = actual_tokenizer.encode("HAM", add_special_tokens=False)[0]

    try:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = actual_tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")

            with torch.no_grad():
                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])


            next_token_logits = outputs.logits[:, -1, :]


            ham_logits = next_token_logits[:, ham_token_id]
            spam_logits = next_token_logits[:, spam_token_id]


            for h_logit, s_logit in zip(ham_logits, spam_logits):
                h_val = round(h_logit.item(), 4)
                s_val = round(s_logit.item(), 4)
                logits_list.append([h_val, s_val])

                if s_logit > h_logit:
                    predictions.append("SPAM")
                else:
                    predictions.append("HAM")
    finally:
        actual_tokenizer.padding_side = original_padding_side
        for obj in [model, getattr(model, "model", None)]:
            if obj is not None and hasattr(obj, "rope_deltas"):
                obj.rope_deltas = None
        if was_training:
            model.train()


    tp, fp, tn, fn = 0, 0, 0, 0
    for pred, target in zip(predictions, targets):
        is_spam_pred = "SPAM" in pred
        is_spam_target = (target == "SPAM")

        if is_spam_pred == is_spam_target:
            correct += 1
            if is_spam_target:
                tp += 1
            else:
                tn += 1
        else:
            if is_spam_target:
                fn += 1
            else:
                fp += 1

    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1_0_5 = 1.25 * precision * recall / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_0.5": f1_0_5,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predictions": predictions,
        "targets": targets,
        "logits": logits_list
    }

def evaluate_loss(model, tokenizer, dataset_hf, batch_size=8):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_steps = 0

    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    original_padding_side = actual_tokenizer.padding_side
    actual_tokenizer.padding_side = "right"

    texts = dataset_hf["text"]

    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = actual_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH
                ).to("cuda")

                labels = inputs["input_ids"].clone()
                if actual_tokenizer.pad_token_id is not None:
                    labels[labels == actual_tokenizer.pad_token_id] = -100

                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1
    finally:
        actual_tokenizer.padding_side = original_padding_side
        if was_training:
            model.train()

    return total_loss / total_steps if total_steps > 0 else 0.0

def analyze_keywords(test_dataset_raw, pred_metrics):
    stop_words = {'the', 'a', 'to', 'and', 'i', 'you', 'u', 'is', 'in', 'it', 'for', 'of', 'on', 'my', 'me', 'have', 'that', 'your', 'with', 'at', 'this', 'get', 'or', 'be', 'are', 'so', 'can', 'not', 'but', 'ur', 'from'}

    word_stats = {}
    for item, pred, target in zip(test_dataset_raw, pred_metrics["predictions"], pred_metrics["targets"]):
        text = item["instruction"]
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(w for w in words if len(w) > 2 and w not in stop_words)

        is_spam_pred = "SPAM" in pred
        is_spam_target = (target == "SPAM")
        is_correct = (is_spam_pred == is_spam_target)

        for w in unique_words:
            if w not in word_stats:
                word_stats[w] = {"count": 0, "correct": 0}
            word_stats[w]["count"] += 1
            if is_correct:
                word_stats[w]["correct"] += 1

    keyword_list = []
    for w, stats in word_stats.items():
        if stats["count"] >= 3:
            accuracy = stats["correct"] / stats["count"]
            keyword_list.append({
                "word": w,
                "count": stats["count"],
                "accuracy": accuracy
            })


    certain = sorted(keyword_list, key=lambda x: (-x["accuracy"], -x["count"]))[:10]
    confusing = sorted(keyword_list, key=lambda x: (x["accuracy"], -x["count"]))[:10]

    return {
        "certain_keywords": certain,
        "confusing_keywords": confusing
    }

def get_steps_config(num_samples, batch_size, grad_accum):
    effective_batch_size = batch_size * grad_accum
    steps_per_epoch = num_samples // effective_batch_size
    if num_samples % effective_batch_size != 0:
        steps_per_epoch += 1


    eval_steps = max(1, steps_per_epoch // 5)
    logging_steps = max(1, steps_per_epoch // 10)
    return logging_steps, eval_steps

class CustomEvalCallback(TrainerCallback):
    def __init__(self, val_dataset_raw, tokenizer, trainer_ref):
        self.val_dataset_raw = val_dataset_raw
        self.tokenizer = tokenizer
        self.trainer_ref = trainer_ref
        self.epoch_logs = []
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        val_loss = metrics.get("eval_loss", None)


        model = kwargs.get("model", None)
        if model is None and self.trainer_ref.get("trainer") is not None:
            model = self.trainer_ref["trainer"].model


        classification_results = evaluate_classification(model, self.tokenizer, self.val_dataset_raw)
        val_acc = classification_results["accuracy"]

        train_loss = self.last_train_loss
        current_epoch = round(state.epoch, 4)
        current_step = state.global_step

        print(f"   [Step {current_step} | Epoch {current_epoch:.2f}] Train Loss: {train_loss if train_loss is not None else 'N/A'} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        self.epoch_logs.append({
            "step": current_step,
            "epoch": current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

# Wariant A ma dwa etapy; pozostałe jeden.
def run_single_experiment(variant, seed, percentage, df_real, split_info, df_synth, results_tracker, gui_queue=None, stats_tracker=None):
    if gui_queue is not None:
        gui_queue.put(("status", variant, seed, percentage, "Preparing", "", ""))

    t_start = time.time()

    print(f"\n{'='*70}")
    print(f" STARTING: Variant {variant} | Seed {seed} | Split {percentage}%")
    print(f"{'='*70}")


    df_val = df_real.iloc[split_info["val_indices"]].reset_index(drop=True)
    df_test = df_real.iloc[split_info["test_indices"]].reset_index(drop=True)

    df_train_final, df_pretrain, train_sampling_audit, pretrain_sampling_audit = prepare_experiment_data(
        df_real, split_info, df_synth, variant, seed, percentage)
    verify_training_plan(df_train_final, df_pretrain, train_sampling_audit, pretrain_sampling_audit)
    df_train_real = df_train_final if variant == 'A' else None
    sampling_audits = []
    for frame, sampling in ((df_pretrain, pretrain_sampling_audit), (df_train_final, train_sampling_audit)):
        if sampling is not None:
            save_manifest(frame, sampling, Path(_CACHE_ROOT) / 'manifests' / _RUN_FINGERPRINT,
                          variant, seed, percentage)
            sampling_audits.append(sampling)

    print(f"[INFO] Train size: {len(df_train_final)} | Val size: {len(df_val)} | Test size: {len(df_test)}")

    def get_batch_config(num_samples):
        if num_samples >= 8000:
            return 2, 8
        elif num_samples >= 3300:
            return 4, 4
        return BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS


    def load_model_helper(model_path=MODEL_NAME):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=False,
            full_finetuning=True
        )
        if hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings > 4096:
            model.config.max_position_embeddings = 4096
        tokenizer.pad_token = "<|im_end|>"
        tokenizer.eos_token = "<|im_end|>"
        return model, tokenizer

    history_log = []


    if variant == "A":
        cache_dir = str(cache_directory(seed))
        history_path = os.path.join(cache_dir, "pretrain_history.json")
        provenance_path = Path(cache_dir) / "provenance.json"

        use_cache = False
        if os.path.exists(cache_dir) and os.path.exists(history_path):
            print(f"[INFO] Found cached pretraining model for Seed {seed} in '{cache_dir}'. Verifying...")
            try:

                with open(history_path, "r", encoding="utf-8") as h_f:
                    pretrain_history = json.load(h_f)


                cached_provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
                expected_cache = dict(fingerprint=_RUN_FINGERPRINT, balanced=BALANCED,
                                      seed=seed, rows=df_pretrain['_sample_id'].tolist())
                if cached_provenance != expected_cache:
                    raise ValueError('Pochodzenie pamięci modelu nie zgadza się z konfiguracją.')
                model, tokenizer = load_model_helper(cache_dir)
                use_cache = True
                print(f"[INFO] Cache verified and loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Cache at '{cache_dir}' is corrupted or incomplete: {e}")
                print(f"[INFO] Cleaning up corrupted cache and rebuilding...")
                if os.path.exists(cache_dir):
                    remove_cache_directory(cache_dir)
                use_cache = False

        if not use_cache:
            t_train_start = time.time()
            if stats_tracker is not None:
                stats_tracker["total_prep_time"] += t_train_start - t_start
                stats_tracker["total_prep_count"] += 1

            if gui_queue is not None:
                gui_queue.put(("status", variant, seed, percentage, "Training (Pretrain)", "", ""))

            print(f"[INFO] Running pretraining on all synthetic data for Seed {seed}...")

            model, tokenizer = load_model_helper(MODEL_NAME)
            train_hf_synth, _ = prepare_hf_dataset(df_pretrain, tokenizer)
            val_hf, val_raw = prepare_hf_dataset(df_val, tokenizer)

            trainer_ref = {"trainer": None}
            eval_callback = CustomEvalCallback(val_raw, tokenizer, trainer_ref)

            current_bs, current_ga = get_batch_config(len(df_synth))
            print(f"[INFO] Using Pretrain Batch Size: {current_bs}, Grad Accum: {current_ga}")

            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=train_hf_synth,
                eval_dataset=val_hf,
                callbacks=[eval_callback],
                args=SFTConfig(
                    dataset_text_field="text",
                    max_length=MAX_SEQ_LENGTH,
                    per_device_train_batch_size=current_bs,
                    gradient_accumulation_steps=current_ga,
                    num_train_epochs=EPOCHS_PRETRAIN,
                    learning_rate=LR_PRETRAIN,
                    fp16=not torch.cuda.is_bf16_supported(),
                    bf16=torch.cuda.is_bf16_supported(),
                    logging_strategy="epoch",
                    eval_strategy="epoch",
                    optim="adamw_8bit",
                    seed=seed,
                    output_dir="./temp_trainer_output",
                    save_strategy="no",
                    gradient_checkpointing=True
                ),
            )
            trainer_ref["trainer"] = trainer
            trainer.train()

            t_train_end = time.time()
            if stats_tracker is not None:
                stats_tracker["total_train_time"] += t_train_end - t_train_start
                stats_tracker["total_train_samples"] += len(df_synth) * EPOCHS_PRETRAIN


            os.makedirs(cache_dir, exist_ok=True)
            model.save_pretrained(cache_dir)
            tokenizer.save_pretrained(cache_dir)


            provenance_path.write_text(json.dumps(dict(fingerprint=_RUN_FINGERPRINT, balanced=BALANCED,
                                                      seed=seed, rows=df_pretrain['_sample_id'].tolist())),
                                       encoding='utf-8')
            pretrain_history = eval_callback.epoch_logs
            with open(history_path, "w", encoding="utf-8") as h_f:
                json.dump(pretrain_history, h_f, indent=2)


            if os.path.exists("./temp_trainer_output"):
                shutil.rmtree("./temp_trainer_output")
            if "trainer" in locals():
                del trainer
            if "trainer_ref" in locals():
                trainer_ref.clear()
                del trainer_ref
            if "eval_callback" in locals():
                del eval_callback
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[INFO] Pretraining completed and cached in '{cache_dir}'.")


            model, tokenizer = load_model_helper(cache_dir)

        history_log = list(pretrain_history)

        t_train_start2 = time.time()
        if use_cache and stats_tracker is not None:
            stats_tracker["total_prep_time"] += t_train_start2 - t_start
            stats_tracker["total_prep_count"] += 1

        if gui_queue is not None:
            gui_queue.put(("status", variant, seed, percentage, "Training (Finetune)", "", ""))

        train_hf, _ = prepare_hf_dataset(df_train_real, tokenizer)
        val_hf, val_raw = prepare_hf_dataset(df_val, tokenizer)
        test_hf, test_raw = prepare_hf_dataset(df_test, tokenizer)

        trainer_ref = {"trainer": None}
        eval_callback_stage2 = CustomEvalCallback(val_raw, tokenizer, trainer_ref)

        current_bs, current_ga = get_batch_config(len(df_train_real)+len(df_synth))
        print(f"[INFO] Using Finetune Batch Size: {current_bs}, Grad Accum: {current_ga}")

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            callbacks=[eval_callback_stage2],
            args=SFTConfig(
                dataset_text_field="text",
                max_length=MAX_SEQ_LENGTH,
                per_device_train_batch_size=current_bs,
                gradient_accumulation_steps=current_ga,
                num_train_epochs=EPOCHS_FINETUNE,
                learning_rate=LR_FINETUNE,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_strategy="epoch",
                eval_strategy="epoch",
                optim="adamw_8bit",
                seed=seed,
                output_dir="./temp_trainer_output",
                save_strategy="no",
                gradient_checkpointing=True
            ),
        )
        trainer_ref["trainer"] = trainer
        print(f"[INFO] Launching finetuning stage on real {percentage}% split...")
        trainer.train()

        t_train_end2 = time.time()
        if stats_tracker is not None:
            stats_tracker["total_train_time"] += t_train_end2 - t_train_start2
            stats_tracker["total_train_samples"] += len(df_train_real) * EPOCHS_FINETUNE

        if os.path.exists("./temp_trainer_output"):
            shutil.rmtree("./temp_trainer_output")


        for log in eval_callback_stage2.epoch_logs:
            log["epoch"] = EPOCHS_PRETRAIN + log["epoch"]
            history_log.append(log)

    else:

        t_train_start = time.time()
        if stats_tracker is not None:
            stats_tracker["total_prep_time"] += t_train_start - t_start
            stats_tracker["total_prep_count"] += 1

        if gui_queue is not None:
            gui_queue.put(("status", variant, seed, percentage, "Training", "", ""))

        model, tokenizer = load_model_helper(MODEL_NAME)
        train_hf, _ = prepare_hf_dataset(df_train_final, tokenizer)
        val_hf, val_raw = prepare_hf_dataset(df_val, tokenizer)
        test_hf, test_raw = prepare_hf_dataset(df_test, tokenizer)

        trainer_ref = {"trainer": None}
        eval_callback = CustomEvalCallback(val_raw, tokenizer, trainer_ref)

        current_bs, current_ga = get_batch_config(len(df_train_final))
        print(f"[INFO] Using Standard Batch Size: {current_bs}, Grad Accum: {current_ga}")

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            callbacks=[eval_callback],
            args=SFTConfig(
                dataset_text_field="text",
                max_length=MAX_SEQ_LENGTH,
                per_device_train_batch_size=current_bs,
                gradient_accumulation_steps=current_ga,
                num_train_epochs=EPOCHS_STANDARD,
                learning_rate=LR_STANDARD,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_strategy="epoch",
                eval_strategy="epoch",
                optim="adamw_8bit",
                seed=seed,
                output_dir="./temp_trainer_output",
                save_strategy="no",
                gradient_checkpointing=True
            ),
        )
        trainer_ref["trainer"] = trainer
        print(f"[INFO] Launching standard training stage...")
        trainer.train()

        t_train_end = time.time()
        if stats_tracker is not None:
            stats_tracker["total_train_time"] += t_train_end - t_train_start
            stats_tracker["total_train_samples"] += len(df_train_final) * EPOCHS_STANDARD

        if os.path.exists("./temp_trainer_output"):
            shutil.rmtree("./temp_trainer_output")

        history_log = eval_callback.epoch_logs


    if gui_queue is not None:
        gui_queue.put(("status", variant, seed, percentage, "Evaluating", "", ""))

    t_eval_start = time.time()

    print(f"\n[INFO] Running final classification evaluation on Test split...")
    test_metrics = evaluate_classification(model, tokenizer, test_raw, batch_size=current_bs)


    test_loss = evaluate_loss(model, tokenizer, test_hf, batch_size=current_bs)

    t_eval_end = time.time()
    if stats_tracker is not None:
        stats_tracker["total_eval_time"] += t_eval_end - t_eval_start
        stats_tracker["total_eval_count"] += 1

    test_results = {
        "test_loss": test_loss,
        "test_acc": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "tp": test_metrics["tp"],
        "fp": test_metrics["fp"],
        "tn": test_metrics["tn"],
        "fn": test_metrics["fn"],
        "logits": test_metrics["logits"]
    }

    print(f" -> Test Loss: {test_loss:.4f} | Test Acc: {test_results['test_acc']*100:.2f}% | F1 Score: {test_results['f1']:.4f}")


    print(f"[INFO] Analyzing confusing and certain keywords...")
    keyword_results = analyze_keywords(test_raw, test_metrics)


    record = {
        "variant": variant,
        "seed": seed,
        "percentage": percentage,
        "history": history_log,
        "test_results": test_results,
        "keyword_analysis": keyword_results,
        "balanced": BALANCED,
        "balancing": sampling_audits,
        "evaluation_protocol": "shared_standard_v1",
        "provenance": _RUN_PROVENANCE
    }

    results_tracker.append(record)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_tracker, f, indent=2)

    print(f"[SUCCESS] Record saved for Variant {variant} Seed {seed} {percentage}% split.")

    if gui_queue is not None:
        gui_queue.put(("status", variant, seed, percentage, "Completed", f"{test_metrics['accuracy']*100:.2f}%", f"{test_loss:.4f}"))


    if "trainer" in locals():
        del trainer
    if "trainer_ref" in locals():
        trainer_ref.clear()
        del trainer_ref
    if "eval_callback" in locals():
        del eval_callback
    if "eval_callback_stage2" in locals():
        del eval_callback_stage2
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
