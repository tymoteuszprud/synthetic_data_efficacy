"""Generowanie obrazów w ComfyUI z przygotowanej puli opisów."""

import os
import csv
import json
import time
import random
import shutil
import urllib.request
import threading
import queue
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from obrazy.dane import read_manifest


COMFY_API_URL = "http://127.0.0.1:8188/prompt"
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", "comfy_output")
WORKFLOW_JSON_PATH = "przeplyw_generowania_obrazow.json"
SUMMARY_CSV_PATH = "klasy_obrazow.csv"
OUTPUT_DATASET_DIR = os.environ.get("SYNTHETIC_IMAGES_DIR", "obrazy_syntetyczne")
DESCRIPTIONS_TXT_PATH = "opisy_scen.txt"

BATCH_SIZE = 4
STEPS = 6
WIDTH = 512
HEIGHT = 512
TARGET_MULTIPLIER = 1.0


def next_image_path(directory):
    indices = [int(m.group(1)) for p in Path(directory).iterdir()
               if p.is_file() and (m := re.fullmatch(r'synth_(\d+)\.png', p.name))]
    return Path(directory) / f'synth_{max(indices, default=-1) + 1:05d}.png'


def load_descriptions(filepath):


    descriptions = {}
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Brak pliku opisów scen: {filepath}')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue


            parts = line.split(':', 1)
            if len(parts) != 2:
                continue

            file_path_part = parts[0].strip()
            prompt_text = parts[1].strip()


            class_name = file_path_part.replace('\\', '/').split('/')[0].lower()

            if class_name not in descriptions:
                descriptions[class_name] = []
            descriptions[class_name].append(prompt_text)

    print(f"[INFO] Załadowano opisy dla {len(descriptions)} klas.")
    return descriptions

DESCRIPTIONS_DICT = load_descriptions(DESCRIPTIONS_TXT_PATH)


def generate_prompt(sport_name: str) -> str:
    sport_clean = sport_name.lower()


    sport_prompts = DESCRIPTIONS_DICT.get(sport_clean)
    if not sport_prompts:
        raise ValueError(f'Brak opisów scen dla klasy: {sport_name}')


    base_prompt = random.choice(sport_prompts)


    quality_modifiers = "national geographic style, photorealistic, high quality, 8k"

    return f"{base_prompt}, {quality_modifiers}"


def build_payload(workflow_template: dict, prompt_text: str, unique_prefix: str, batch_size=BATCH_SIZE) -> dict:
    import copy
    if not 1 <= batch_size <= BATCH_SIZE:
        raise ValueError('Rozmiar partii musi należeć do zakresu 1–4.')
    workflow = copy.deepcopy(workflow_template)

    pos_node_id = None
    vae_decode_id = None

    for key, node in workflow.items():
        c_type = node.get("class_type")
        if c_type in ["KSampler", "SwarmKSampler"]:
            pos_link = node["inputs"].get("positive")
            if isinstance(pos_link, list) and len(pos_link) > 0:
                pos_node_id = str(pos_link[0])
        elif c_type == "VAEDecode":
            vae_decode_id = key

    for key, node in workflow.items():
        c_type = node.get("class_type", "")

        if c_type == "EmptySD3LatentImage":
            node["inputs"]["width"] = WIDTH
            node["inputs"]["height"] = HEIGHT
            node["inputs"]["batch_size"] = batch_size

        elif c_type == "CLIPTextEncode":
            if key == pos_node_id:
                node["inputs"]["text"] = prompt_text

        elif c_type in ["KSampler", "SwarmKSampler"]:
            seed_key = "noise_seed" if "noise_seed" in node["inputs"] else "seed"
            node["inputs"][seed_key] = random.randint(1, 0xFFFFFFFFFFFFFF)
            node["inputs"]["steps"] = STEPS

        elif c_type in ["SaveImage", "PreviewImage"]:
            node["class_type"] = "SaveImage"
            node["inputs"]["filename_prefix"] = unique_prefix

    save_node_exists = any("SaveImage" in n.get("class_type", "") for n in workflow.values())
    if not save_node_exists and vae_decode_id:
        workflow["9999"] = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": unique_prefix,
                "images": [vae_decode_id, 0]
            }
        }

    return {"prompt": workflow}


class GenerationDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Generowanie syntetycznych obrazów")
        self.geometry("900x700")
        self.configure(bg="#1e1e1e")
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        self.is_running = True
        self.msg_queue = queue.Queue()

        self.times_history = []
        self.total_generated = 0
        self.total_target = 1

        self.setup_ui()
        self.load_classes_data()

        self.worker_thread = threading.Thread(target=self.generation_loop, daemon=True)
        self.worker_thread.start()

        self.after(100, self.process_queue)

    def setup_ui(self):
        self.left_frame = tk.Frame(self, bg="#1e1e1e")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.img_label = tk.Label(self.left_frame, bg="#2d2d2d", text="Czekam na pierwszą generację...", fg="white", font=("Arial", 14))
        self.img_label.pack(fill=tk.BOTH, expand=True)

        self.prompt_label = tk.Text(self.left_frame, height=5, bg="#2d2d2d", fg="#a9b7c6", font=("Consolas", 10), wrap=tk.WORD)
        self.prompt_label.pack(fill=tk.X, pady=(10, 0))
        self.prompt_label.insert(tk.END, "Ostatni użyty prompt pojawi się tutaj...")
        self.prompt_label.config(state=tk.DISABLED)

        self.right_frame = tk.Frame(self, bg="#1e1e1e", width=300)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)

        def create_stat_label(parent, text):
            lbl = tk.Label(parent, text=text, bg="#1e1e1e", fg="#e0e0e0", font=("Arial", 12, "bold"), anchor="w")
            lbl.pack(fill=tk.X, pady=(10, 2))
            val = tk.Label(parent, text="-", bg="#1e1e1e", fg="#4CAF50", font=("Consolas", 14), anchor="w")
            val.pack(fill=tk.X)
            return val

        self.val_sport = create_stat_label(self.right_frame, "Aktualny Sport:")
        self.val_progress = create_stat_label(self.right_frame, "Postęp całkowity:")
        self.val_speed = create_stat_label(self.right_frame, "Prędkość (sek/obraz):")
        self.val_eta = create_stat_label(self.right_frame, "ETA:")

        self.btn_stop = tk.Button(self.right_frame, text="Zatrzymaj (Safe Stop)", bg="#d32f2f", fg="white", font=("Arial", 12, "bold"), command=self.stop_generation)
        self.btn_stop.pack(side=tk.BOTTOM, fill=tk.X, pady=20)

    def load_classes_data(self):
        manifest = read_manifest(os.environ.get('IMAGE_MANIFEST', 'manifest_obrazow.json'))
        selected = json.loads(os.environ.get('IMAGE_CLASSES', 'null')) or manifest['classes']
        if set(selected) - set(manifest['classes']):
            raise ValueError('Wybrane klasy muszą należeć do manifestu.')
        self.classes_data = [dict(id=i, name=name, target=manifest['synthetic_targets'][name], current=0)
                             for i, name in enumerate(selected)]
        for name in selected:
            generate_prompt(name)

        self.total_target = sum(c['target'] for c in self.classes_data)

        for cls in self.classes_data:
            class_dir = os.path.join(OUTPUT_DATASET_DIR, cls['name'])
            os.makedirs(class_dir, exist_ok=True)
            cls['current'] = len([f for f in os.listdir(class_dir) if f.lower().endswith('.png')])
            if cls['current'] > cls['target']:
                raise ValueError(f"Klasa {cls['name']} zawiera więcej obrazów niż cel; wymagana jest jawna kontrola katalogu.")

        self.total_generated = sum(c['current'] for c in self.classes_data)

    def stop_generation(self):
        self.is_running = False
        self.btn_stop.config(text="Zatrzymywanie...", state=tk.DISABLED)

    def process_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg['type'] == 'update_stats':
                    self.val_sport.config(text=f"{msg['sport']} ({msg['sport_curr']}/{msg['sport_tgt']})")
                    self.val_progress.config(text=f"{msg['total_gen']} / {self.total_target}")

                    self.times_history.append(msg['time_taken'])
                    if len(self.times_history) > 20:
                        self.times_history.pop(0)

                    avg_time = sum(self.times_history) / len(self.times_history)
                    self.val_speed.config(text=f"{avg_time:.2f} s/it")

                    images_left = self.total_target - msg['total_gen']
                    eta_seconds = images_left * avg_time
                    m, s = divmod(eta_seconds, 60)
                    h, m = divmod(m, 60)
                    self.val_eta.config(text=f"{int(h)}h {int(m)}m {int(s)}s")

                elif msg['type'] == 'update_image':
                    img_path = msg['path']
                    last_prompt = msg['prompt']

                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        img.thumbnail((600, 600))
                        photo = ImageTk.PhotoImage(img)
                        self.img_label.config(image=photo, text="")
                        self.img_label.image = photo

                    self.prompt_label.config(state=tk.NORMAL)
                    self.prompt_label.delete(1.0, tk.END)
                    self.prompt_label.insert(tk.END, last_prompt)
                    self.prompt_label.config(state=tk.DISABLED)

                elif msg['type'] == 'finished':
                    self.val_sport.config(text="ZAKOŃCZONO")
                    self.btn_stop.config(text="Wyjdź", command=self.destroy, state=tk.NORMAL, bg="#4CAF50")

        except queue.Empty:
            pass
        finally:
            if self.is_running or not self.msg_queue.empty():
                self.after(100, self.process_queue)

    def generation_loop(self):
        with open(WORKFLOW_JSON_PATH, 'r', encoding='utf-8') as f:
            workflow_template = json.load(f)

        while self.is_running:
            active_classes = [c for c in self.classes_data if c['current'] < c['target']]
            if not active_classes:
                self.msg_queue.put({'type': 'finished'})
                break

            for cls in active_classes:
                if not self.is_running:
                    break

                sport_name = cls['name']
                class_dir = os.path.join(OUTPUT_DATASET_DIR, sport_name)
                prompt_text = generate_prompt(sport_name)

                unique_hash = f"T_{int(time.time()*100)}_{sport_name.replace(' ', '')}"

                batch_size = min(BATCH_SIZE, cls['target'] - cls['current'])
                payload = build_payload(workflow_template, prompt_text, unique_hash, batch_size)
                req_data = json.dumps(payload).encode('utf-8')

                start_t = time.time()
                try:
                    req = urllib.request.Request(COMFY_API_URL, data=req_data, headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req) as response:
                        response.read()

                    processed_batch_files = []
                    timeout = 0
                    while True:
                        found_files = [f for f in os.listdir(COMFY_OUTPUT_DIR) if f.startswith(unique_hash)]

                        if len(found_files) >= batch_size:
                            found_files.sort()

                            all_ready = True
                            temp_paths = []
                            for f in found_files[:batch_size]:
                                f_path = os.path.join(COMFY_OUTPUT_DIR, f)
                                if os.path.getsize(f_path) == 0:
                                    all_ready = False
                                    break
                                temp_paths.append(f_path)

                            if all_ready:
                                processed_batch_files = temp_paths
                                break

                        time.sleep(0.5)
                        timeout += 0.5
                        if timeout > 900:
                            raise TimeoutError(f"Przekroczono czas oczekiwania na partię ({batch_size} obrazów).")

                    last_dest_file = ""
                    for src_file_path in processed_batch_files:
                        dest_file = str(next_image_path(class_dir))
                        if os.path.exists(dest_file):
                            raise FileExistsError(f'Plik docelowy już istnieje: {dest_file}')
                        shutil.move(src_file_path, dest_file)

                        cls['current'] += 1
                        self.total_generated += 1
                        last_dest_file = dest_file

                    elapsed = time.time() - start_t

                    self.msg_queue.put({
                        'type': 'update_stats',
                        'sport': sport_name,
                        'sport_curr': cls['current'],
                        'sport_tgt': cls['target'],
                        'total_gen': self.total_generated,
                        'time_taken': elapsed / batch_size
                    })

                    if last_dest_file:
                        self.msg_queue.put({
                            'type': 'update_image',
                            'path': last_dest_file,
                            'prompt': prompt_text
                        })

                except Exception as e:
                    print(f"[BŁĄD KRYTYCZNY] {sport_name}: {e}")
                    time.sleep(2)

if __name__ == "__main__":
    app = GenerationDashboard()
    app.mainloop()
