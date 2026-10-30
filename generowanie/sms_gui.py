"""Generowanie wiadomości HAM i SPAM przez lokalny serwer modelu."""

import os
import csv
import json
import time
import random
import threading
import queue
import tkinter as tk
from tkinter import messagebox, filedialog
import requests
from pathlib import Path


DEFAULT_SERVER_URL = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL_NAME = "auto"
DEFAULT_OUTPUT_FILE = os.environ.get("SMS_OUTPUT_FILE", "wiadomosci_syntetyczne.csv")
INPUT_EXAMPLES_FILE = "wiadomosci_referencyjne.csv"
WORDS_DICTIONARY_FILE = "slownik_angielski.json"
MESSAGE_LIMIT = 2


SYSTEM_PROMPT = "You are a helpful assistant that generates datasets of realistic SMS messages in structured JSON format."


def load_instruction(label, examples_str, seed_instruction):
    path = Path(__file__).resolve().parents[1] / 'instrukcje' / f'{label}.txt'
    return path.read_text(encoding='utf-8').rstrip('\r\n').format(
        examples_str=examples_str, seed_instruction=seed_instruction)


class ModernProgressBar(tk.Canvas):
    def __init__(self, parent, bg="#27272a", fg="#6366f1", height=15, **kwargs):
        super().__init__(parent, height=height, bg=bg, highlightthickness=0, bd=0, **kwargs)
        self.fg = fg
        self.bg = bg
        self.progress = 0.0
        self.bind("<Configure>", lambda e: self.draw())

    def set_progress(self, progress):
        self.progress = max(0.0, min(1.0, progress))
        self.draw()

    def draw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()


        self.create_rectangle(0, 0, w, h, fill=self.bg, width=0)


        if self.progress > 0:
            self.create_rectangle(0, 0, int(w * self.progress), h, fill=self.fg, width=0)


class SpamHamGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Generowanie wiadomości HAM i SPAM")
        self.root.geometry("780x700")
        self.root.configure(bg="#121214")
        self.root.resizable(True, True)


        try:
            self.root.tk.call('tk', 'scaling', 1.5)
        except Exception:
            pass


        self.is_generating = False
        self.worker_thread = None
        self.ui_queue = queue.Queue()


        self.session_generated_count = 0
        self.session_start_time = None


        self.spam_examples = []
        self.ham_examples = []
        self.dictionary_words = []


        self.goal_var = tk.StringVar(value="6000")
        self.output_file_var = tk.StringVar(value=DEFAULT_OUTPUT_FILE)
        self.server_url_var = tk.StringVar(value=DEFAULT_SERVER_URL)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL_NAME)


        self.spam_count_var = tk.StringVar(value="0")
        self.ham_count_var = tk.StringVar(value="0")
        self.total_count_var = tk.StringVar(value="0")
        self.speed_var = tk.StringVar(value="0.0 msg/min")
        self.status_var = tk.StringVar(value="Idle - Ready to load data files")


        self.script_dir = os.path.dirname(os.path.abspath(__file__))


        self.setup_ui()


        self.load_reference_files()


        self.root.after(100, self.poll_ui_queue)

    def style_button(self, btn, bg_color, hover_color, fg_color="#ffffff"):
        btn.config(
            bg=bg_color,
            fg=fg_color,
            activebackground=hover_color,
            activeforeground=fg_color,
            relief="flat",
            bd=0,
            padx=18,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2"
        )
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_color))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg_color))

    def setup_ui(self):

        title_frame = tk.Frame(self.root, bg="#1e1e24", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="Generowanie wiadomości HAM i SPAM",
            font=("Segoe UI", 15, "bold"),
            fg="#ffffff",
            bg="#1e1e24",
            anchor="w",
            padx=20
        )
        title_label.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.status_dot = tk.Canvas(title_frame, width=12, height=12, bg="#1e1e24", highlightthickness=0)
        self.status_dot.pack(side=tk.RIGHT, padx=25)
        self.set_status_dot_color("#f59e0b")


        container = tk.Frame(self.root, bg="#121214", padx=20, pady=15)
        container.pack(fill=tk.BOTH, expand=True)


        config_card = tk.LabelFrame(
            container,
            text=" GENERATOR CONFIGURATION ",
            font=("Segoe UI", 9, "bold"),
            fg="#a78bfa",
            bg="#1a1a1e",
            bd=1,
            relief="solid",
            padx=15,
            pady=12
        )
        config_card.pack(fill=tk.X, pady=(0, 15))


        config_card.columnconfigure(1, weight=1)


        tk.Label(config_card, text="LM Studio URL:", font=("Segoe UI", 9), fg="#a0a0ab", bg="#1a1a1e", anchor="w").grid(row=0, column=0, sticky="w", pady=4)
        url_entry = tk.Entry(config_card, textvariable=self.server_url_var, font=("Consolas", 9), bg="#27272a", fg="#ffffff", insertbackground="white", bd=1, relief="solid", highlightthickness=0)
        url_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=4)


        tk.Label(config_card, text="Model Name:", font=("Segoe UI", 9), fg="#a0a0ab", bg="#1a1a1e", anchor="w").grid(row=1, column=0, sticky="w", pady=4)
        model_entry = tk.Entry(config_card, textvariable=self.model_var, font=("Consolas", 9), bg="#27272a", fg="#ffffff", insertbackground="white", bd=1, relief="solid", highlightthickness=0)
        model_entry.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=4)


        bottom_config_frame = tk.Frame(config_card, bg="#1a1a1e")
        bottom_config_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        bottom_config_frame.columnconfigure(1, weight=1)
        bottom_config_frame.columnconfigure(3, weight=3)

        tk.Label(bottom_config_frame, text="Goal (msgs):", font=("Segoe UI", 9), fg="#a0a0ab", bg="#1a1a1e", anchor="w").grid(row=0, column=0, sticky="w")
        goal_entry = tk.Entry(bottom_config_frame, textvariable=self.goal_var, font=("Segoe UI", 9, "bold"), bg="#27272a", fg="#ffffff", insertbackground="white", width=8, bd=1, relief="solid", highlightthickness=0, justify="center")
        goal_entry.grid(row=0, column=1, sticky="w", padx=(10, 20))

        tk.Label(bottom_config_frame, text="Output File:", font=("Segoe UI", 9), fg="#a0a0ab", bg="#1a1a1e", anchor="w").grid(row=0, column=2, sticky="w")

        file_picker_frame = tk.Frame(bottom_config_frame, bg="#1a1a1e")
        file_picker_frame.grid(row=0, column=3, sticky="ew", padx=(10, 0))
        file_picker_frame.columnconfigure(0, weight=1)

        file_entry = tk.Entry(file_picker_frame, textvariable=self.output_file_var, font=("Segoe UI", 9), bg="#27272a", fg="#ffffff", insertbackground="white", bd=1, relief="solid", highlightthickness=0)
        file_entry.grid(row=0, column=0, sticky="ew")

        browse_btn = tk.Button(file_picker_frame, text="Browse", command=self.browse_output_file, bg="#3f3f46", fg="#ffffff", relief="flat", bd=0, padx=8, font=("Segoe UI", 8, "bold"), cursor="hand2")
        browse_btn.grid(row=0, column=1, padx=(5, 0))


        metrics_card = tk.Frame(container, bg="#1a1a1e", bd=1, relief="solid")
        metrics_card.pack(fill=tk.X, pady=(0, 15))

        metrics_grid = tk.Frame(metrics_card, bg="#1a1a1e", padx=15, pady=12)
        metrics_grid.pack(fill=tk.X)
        for i in range(4):
            metrics_grid.columnconfigure(i, weight=1)


        def make_metric(col, label, var_ref, color):
            frame = tk.Frame(metrics_grid, bg="#1a1a1e")
            frame.grid(row=0, column=col, sticky="nsew")
            lbl = tk.Label(frame, text=label, font=("Segoe UI", 8, "bold"), fg="#88888e", bg="#1a1a1e")
            lbl.pack(anchor="center")
            val = tk.Label(frame, textvariable=var_ref, font=("Segoe UI", 16, "bold"), fg=color, bg="#1a1a1e")
            val.pack(anchor="center", pady=(2, 0))

        make_metric(0, "TOTAL MESSAGES", self.total_count_var, "#ffffff")
        make_metric(1, "SPAM MESSAGES", self.spam_count_var, "#f38ba8")
        make_metric(2, "HAM MESSAGES", self.ham_count_var, "#a6e3a1")
        make_metric(3, "CURRENT SPEED", self.speed_var, "#89b4fa")


        progress_control_frame = tk.Frame(container, bg="#121214")
        progress_control_frame.pack(fill=tk.X, pady=(0, 15))


        self.progress_bar = ModernProgressBar(progress_control_frame, bg="#27272a", fg="#6366f1", height=10)
        self.progress_bar.pack(fill=tk.X, pady=(0, 12))


        btn_frame = tk.Frame(progress_control_frame, bg="#121214")
        btn_frame.pack(fill=tk.X)

        self.start_btn = tk.Button(btn_frame, text="START GENERATION", command=self.start_generation)
        self.style_button(self.start_btn, bg_color="#10b981", hover_color="#059669")
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(btn_frame, text="STOP / PAUSE", command=self.stop_generation, state=tk.DISABLED)
        self.style_button(self.stop_btn, bg_color="#3f3f46", hover_color="#52525b")
        self.stop_btn.pack(side=tk.LEFT)


        self.status_lbl = tk.Label(btn_frame, textvariable=self.status_var, font=("Segoe UI", 9, "italic"), fg="#a0a0ab", bg="#121214", anchor="e")
        self.status_lbl.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))


        preview_card = tk.LabelFrame(
            container,
            text=" LIVE GENERATION PREVIEW & LOG ",
            font=("Segoe UI", 9, "bold"),
            fg="#a78bfa",
            bg="#1a1a1e",
            bd=1,
            relief="solid",
            padx=15,
            pady=12
        )
        preview_card.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            preview_card,
            bg="#0f0f11",
            fg="#f4f4f5",
            insertbackground="white",
            font=("Consolas", 9),
            bd=0,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)


        scrollbar = tk.Scrollbar(self.log_text, bg="#0f0f11", bd=0, width=8)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)

    def set_status_dot_color(self, color):
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 10, 10, fill=color, outline="")

    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Choose Output Dataset File"
        )
        if filename:
            self.output_file_var.set(filename)
            self.update_existing_metrics()

    def log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


    def find_file(self, filename):

        if os.path.exists(filename):
            return filename

        alt_path = os.path.join(self.script_dir, filename)
        if os.path.exists(alt_path):
            return alt_path
        material_path = os.path.join(self.script_dir, "..", "materialy", filename)
        if os.path.exists(material_path):
            return material_path
        return None

    def load_reference_files(self):
        csv_path = self.find_file(INPUT_EXAMPLES_FILE)
        json_path = self.find_file(WORDS_DICTIONARY_FILE)


        if csv_path:
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        lbl = row.get('label', '').strip().lower()
                        msg = row.get('message', '').strip()
                        if lbl == 'spam':
                            self.spam_examples.append(msg)
                        elif lbl == 'ham':
                            self.ham_examples.append(msg)

                self.log_message(f"[INFO] Loaded examples: {len(self.spam_examples)} SPAM, {len(self.ham_examples)} HAM from '{csv_path}'.")
            except Exception as e:
                self.log_message(f"[ERROR] Failed to load '{INPUT_EXAMPLES_FILE}': {e}")
        else:
            self.log_message(f"[WARNING] Example CSV '{INPUT_EXAMPLES_FILE}' not found. Cannot run generation without examples!")
            messagebox.showerror("Error", f"Could not find example file '{INPUT_EXAMPLES_FILE}'. Plik jest wymagany w katalogu roboczym lub katalogu materiałów.")


        if json_path:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.dictionary_words = list(data.keys())
                self.log_message(f"[INFO] Loaded {len(self.dictionary_words)} seed words from '{json_path}'.")
            except Exception as e:
                self.log_message(f"[ERROR] Failed to load dictionary '{WORDS_DICTIONARY_FILE}': {e}")
        else:
            self.log_message(f"[WARNING] Dictionary '{WORDS_DICTIONARY_FILE}' not found. Seed words will be generic.")
            messagebox.showerror("Error", f"Could not find dictionary file '{WORDS_DICTIONARY_FILE}'. Plik jest wymagany w katalogu roboczym lub katalogu materiałów.")



    def update_existing_metrics(self):
        output_file = self.output_file_var.get()
        if not output_file:
            return

        spam_count = 0
        ham_count = 0

        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        lbl = row.get('label', '').strip().lower()
                        if lbl == 'spam':
                            spam_count += 1
                        elif lbl == 'ham':
                            ham_count += 1
            except Exception as e:
                self.log_message(f"[ERROR] Failed to read existing '{output_file}': {e}")

        total_count = spam_count + ham_count
        self.spam_count_var.set(str(spam_count))
        self.ham_count_var.set(str(ham_count))
        self.total_count_var.set(str(total_count))


        try:
            goal = int(self.goal_var.get())
            if goal > 0:
                self.progress_bar.set_progress(total_count / goal)
        except ValueError:
            pass

        self.status_var.set(f"Loaded output. Progress: {total_count} messages.")


    def start_generation(self):

        if not self.spam_examples or not self.ham_examples:
            messagebox.showerror("Error", "Example dataset (spam/ham messages) not loaded. Wymagany jest plik CSV z referencjami.")
            return
        if not self.dictionary_words:
            messagebox.showerror("Error", "Words dictionary not loaded. Wymagany jest słownik JSON.")
            return

        try:
            goal = int(self.goal_var.get())
            if goal <= 0:
                raise ValueError("Goal must be greater than zero.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid target goal: {e}. Wymagana jest dodatnia liczba całkowita.")
            return


        current_total = int(self.total_count_var.get())
        if current_total >= goal:
            messagebox.showinfo("Goal Reached", f"Current total ({current_total}) is already equal to or larger than the target count ({goal}).")
            return


        self.is_generating = True
        self.set_status_dot_color("#10b981")
        self.status_var.set("Starting generation thread...")
        self.log_message("[START] Generation started.")


        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.style_button(self.start_btn, bg_color="#3f3f46", hover_color="#3f3f46")
        self.style_button(self.stop_btn, bg_color="#ef4444", hover_color="#dc2626")


        self.session_generated_count = 0
        self.session_start_time = time.time()


        self.worker_thread = threading.Thread(
            target=self.generation_worker,
            args=(
                self.server_url_var.get(),
                self.model_var.get(),
                self.output_file_var.get(),
                goal
            ),
            daemon=True
        )
        self.worker_thread.start()

    def stop_generation(self):
        if self.is_generating:
            self.is_generating = False
            self.status_var.set("Stopping background worker...")
            self.log_message("[STOP] Stop requested. Finishing current request...")

    def generation_worker(self, server_url, model_name, output_file, goal):
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


        spam_count = 0
        ham_count = 0

        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        lbl = row.get('label', '').strip().lower()
                        if lbl == 'spam':
                            spam_count += 1
                        elif lbl == 'ham':
                            ham_count += 1
            except Exception as e:
                self.ui_queue.put(("error", f"Failed to read output file progress: {e}"))
                return

        total_count = spam_count + ham_count
        write_lock = threading.Lock()


        consecutive_timeouts = 0
        timeouts_lock = threading.Lock()


        def send_single_request(target_class):
            nonlocal consecutive_timeouts
            if not self.is_generating:
                return None


            seed_words = random.sample(self.dictionary_words, min(10, len(self.dictionary_words)))
            seed_words_str = ", ".join(seed_words)
            seed_instruction = f"random seed words, generally ignore: {seed_words_str}"


            if target_class == "spam":
                examples = random.sample(self.spam_examples, min(7, len(self.spam_examples)))
                examples_str = "\n".join([f"Example {idx}: {msg}" for idx, msg in enumerate(examples, 1)])

                user_instruction = load_instruction('spam', examples_str, seed_instruction)
            else:
                examples = random.sample(self.ham_examples, min(7, len(self.ham_examples)))
                examples_str = "\n".join([f"Example {idx}: {msg}" for idx, msg in enumerate(examples, 1)])

                user_instruction = load_instruction('ham', examples_str, seed_instruction)

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_instruction}
                ],
                "temperature": 1.1,
                "max_tokens": 3000,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dataset_response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "dataset": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": { "type": "string", "enum": ["spam", "ham"] },
                                            "message": { "type": "string" }
                                        },
                                        "required": ["label", "message"]
                                    }
                                }
                            },
                            "required": ["dataset"]
                        }
                    }
                },
                "chat_template_kwargs": {"enable_thinking": False}
            }

            while self.is_generating:
                try:

                    response = requests.post(server_url, json=payload, timeout=200)
                    response.raise_for_status()

                    raw_response = response.json()['choices'][0]['message']['content'].strip()
                    parsed_data = json.loads(raw_response)
                    items = parsed_data.get('dataset', [])

                    valid_items = []
                    for item in items:
                        lbl = item.get('label', '').strip().lower()
                        msg = item.get('message', '').strip()
                        if lbl == target_class and msg:
                            valid_items.append((lbl, msg))


                    with timeouts_lock:
                        consecutive_timeouts = 0

                    return "success", valid_items

                except requests.exceptions.Timeout as e:
                    with timeouts_lock:
                        consecutive_timeouts += 1
                        current_timeouts = consecutive_timeouts


                    self.ui_queue.put(("success", {
                        "label": "warning",
                        "message": f"Request for {target_class.upper()} timed out ({current_timeouts}/5). Retrying...",
                        "spam_count": spam_count,
                        "ham_count": ham_count,
                        "total_count": total_count,
                        "goal": goal
                    }))

                    if current_timeouts >= 5:
                        return "error", "5 consecutive requests timed out. Terminating generation."

                    time.sleep(2)
                    continue

                except Exception as e:
                    return "error", str(e)


        spam_submitted = spam_count
        ham_submitted = ham_count

        def get_next_class():
            nonlocal spam_submitted, ham_submitted
            if spam_submitted <= ham_submitted:
                spam_submitted += 25
                return "spam"
            else:
                ham_submitted += 25
                return "ham"

        with ThreadPoolExecutor(max_workers=MESSAGE_LIMIT) as executor:
            futures = {}


            remaining_needed = goal - total_count
            needed_requests = (remaining_needed + 24) // 25
            while len(futures) < min(MESSAGE_LIMIT, needed_requests) and self.is_generating:
                c = get_next_class()
                futures[executor.submit(send_single_request, c)] = c

            error_msg = None

            while futures and self.is_generating:

                done, not_done = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    completed_class = futures.pop(future)

                    res = future.result()
                    if res is None:
                        continue

                    res_type, res_data = res
                    if res_type == "error":
                        error_msg = res_data

                        for f in not_done:
                            f.cancel()
                        break


                    with write_lock:
                        for item_label, item_msg in res_data:
                            if total_count >= goal:
                                break

                            file_exists = os.path.exists(output_file)
                            dir_name = os.path.dirname(output_file)
                            if dir_name and not os.path.exists(dir_name):
                                os.makedirs(dir_name)

                            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                if not file_exists or os.path.getsize(output_file) == 0:
                                    writer.writerow(['label', 'message'])
                                writer.writerow([item_label, item_msg])


                            if item_label == 'spam':
                                spam_count += 1
                            else:
                                ham_count += 1
                            total_count = spam_count + ham_count


                            self.ui_queue.put(("success", {
                                "label": item_label,
                                "message": item_msg,
                                "spam_count": spam_count,
                                "ham_count": ham_count,
                                "total_count": total_count,
                                "goal": goal
                            }))

                if error_msg:
                    self.ui_queue.put(("error", f"API Request Failed: {error_msg}"))
                    break


                remaining_needed = goal - total_count
                active_running = len(futures)
                needed_requests = (remaining_needed + 24) // 25
                to_submit = min(MESSAGE_LIMIT, needed_requests) - active_running

                for _ in range(to_submit):
                    if self.is_generating:
                        c = get_next_class()
                        futures[executor.submit(send_single_request, c)] = c


                time.sleep(0.05)


        self.ui_queue.put(("finished", None))


    def poll_ui_queue(self):
        try:
            while True:
                msg_type, data = self.ui_queue.get_nowait()
                if msg_type == "success":

                    self.spam_count_var.set(str(data["spam_count"]))
                    self.ham_count_var.set(str(data["ham_count"]))
                    self.total_count_var.set(str(data["total_count"]))


                    timestamp = time.strftime("%H:%M:%S")
                    self.log_message(f"[{timestamp}] [{data['label'].upper()}] {data['message']}")


                    self.session_generated_count += 1
                    elapsed = time.time() - self.session_start_time
                    if elapsed > 0:
                        msgs_per_min = (self.session_generated_count / elapsed) * 60
                        self.speed_var.set(f"{msgs_per_min:.1f} msg/min")


                    progress = data["total_count"] / data["goal"]
                    self.progress_bar.set_progress(progress)
                    self.status_var.set(f"Generating... {data['total_count']}/{data['goal']} messages.")

                elif msg_type == "error":
                    self.log_message(f"[ERROR] {data}")
                    messagebox.showerror("Generation Error", f"An error occurred during generation:\n{data}")
                    self.is_generating = False

                elif msg_type == "finished":
                    self.is_generating = False
                    self.status_var.set("Idle - Generation Stopped")
                    self.set_status_dot_color("#f59e0b")
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.style_button(self.start_btn, bg_color="#10b981", hover_color="#059669")
                    self.style_button(self.stop_btn, bg_color="#3f3f46", hover_color="#52525b")
                    self.log_message("[FINISHED] Generation loop exited.")


                    goal = int(self.goal_var.get())
                    total = int(self.total_count_var.get())
                    if total >= goal:
                        self.status_var.set("Success - Goal Reached!")
                        self.set_status_dot_color("#10b981")
                        messagebox.showinfo("Success", f"Target goal of {goal} messages reached successfully!")

        except queue.Empty:
            pass


        self.root.after(100, self.poll_ui_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = SpamHamGeneratorApp(root)
    root.mainloop()
