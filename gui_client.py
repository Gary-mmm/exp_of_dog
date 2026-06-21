"""CKKS MLaaS GUI — supports Tiny & HE-CNN models, custom image upload.

Usage:
    python gui_client.py
"""

import base64
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import requests
import tenseal as ts
import torch
from PIL import Image, ImageTk
from torch import nn
from torchvision import transforms

from he_context import create_ckks_context, public_context_bytes
from train_catdog import build_loaders

CLASSES = {0: "cat", 1: "dog"}
IMG_SIZE = (320, 320)  # display size
CIFAR_SIZE = 32


# ============================================================
#  Model adapters
# ============================================================

class TinyModelAdapter:
    name = "Tiny (AvgPool→FC→□²→FC)"
    default_port = 5000

    def __init__(self):
        from he_tiny_model import load_torch_tiny_model
        self.model = load_torch_tiny_model()

    def plain_predict(self, image_3ch_32):
        with torch.no_grad():
            logits = self.model(image_3ch_32.unsqueeze(0))
        return logits.squeeze(0).numpy()

    def encrypt(self, context, image_3ch_32):
        from he_client import encrypt_feature_vector
        enc_x = encrypt_feature_vector(context, self.model, image_3ch_32)
        return {
            "context": b64encode(public_context_bytes(context)),
            "ciphertext": b64encode(enc_x.serialize()),
        }

    def preprocess_display(self, image_3ch_32):
        return image_3ch_32  # 32×32


class CNNModelAdapter:
    name = "HE-CNN (Conv→□²→FC→□²→FC)"
    default_port = 5001

    def __init__(self):
        from he_cnn_model import load_torch_he_cnn
        self.model = load_torch_he_cnn("artifacts/he_cnn_weights.npz")

    def plain_predict(self, image_3ch_32):
        img_16 = nn.functional.avg_pool2d(image_3ch_32.unsqueeze(0), 2)
        with torch.no_grad():
            logits = self.model(img_16)
        return logits.squeeze(0).numpy()

    def encrypt(self, context, image_3ch_32):
        from he_cnn_ops import im2col_encode_channel
        img_16 = nn.functional.avg_pool2d(image_3ch_32.unsqueeze(0), 2).squeeze(0).numpy()
        enc_channels = []
        windows_nb = None
        for c in range(3):
            enc, wn = im2col_encode_channel(context, img_16[c], 3, 3, 1, padding=1)
            enc_channels.append(enc)
            if windows_nb is None:
                windows_nb = wn
        return {
            "context": b64encode(public_context_bytes(context)),
            "ciphertexts": [b64encode(ct.serialize()) for ct in enc_channels],
            "windows_nb": windows_nb,
        }

    def preprocess_display(self, image_3ch_32):
        return nn.functional.avg_pool2d(image_3ch_32.unsqueeze(0), 2).squeeze(0)  # 16×16


# ============================================================
#  Helpers
# ============================================================

def b64encode(value):
    return base64.b64encode(value).decode("ascii")


def load_image_from_disk(path):
    """Load any image from disk → 3×32×32 tensor in [0,1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((CIFAR_SIZE, CIFAR_SIZE), Image.Resampling.BILINEAR)
    tensor = transforms.ToTensor()(img)
    return tensor


# ============================================================
#  GUI Application
# ============================================================

class MLaaSClientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CKKS MLaaS — Encrypted Cat/Dog Inference")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")

        # State
        self.samples = list(build_loaders("./data", batch_size=1, num_workers=0)[1])
        self.context = create_ckks_context()
        self.sample_idx = tk.IntVar(value=0)
        self.status_text = tk.StringVar(value="Ready. Select a sample or upload an image.")
        self.batch_status = tk.StringVar(value="")
        self.batch_progress = tk.IntVar(value=0)

        # Custom image storage
        self.custom_image = None  # 3×32×32 tensor or None

        # Model adapters
        self.adapters = {
            "Tiny": TinyModelAdapter(),
            "HE-CNN": CNNModelAdapter(),
        }
        self.current = tk.StringVar(value="HE-CNN")
        self.adapter = self.adapters[self.current.get()]

        self._build_ui()
        self._on_model_change()
        self.show_sample()

    # ================================================================
    #  UI
    # ================================================================

    def _build_ui(self):
        root = self.root
        p = {"padx": 6, "pady": 4}

        # --- Image panel (left) ---
        img_frame = ttk.LabelFrame(root, text="Preview", padding=10)
        img_frame.grid(row=0, column=0, rowspan=8, padx=10, pady=10, sticky="n")

        self.img_orig_label = ttk.Label(img_frame)
        self.img_orig_label.grid(row=0, column=0)
        ttk.Label(img_frame, text="Original (32×32)", font=("", 8)).grid(
            row=1, column=0, pady=(0, 10))

        self.img_pp_label = ttk.Label(img_frame)
        self.img_pp_label.grid(row=2, column=0)
        ttk.Label(img_frame, text="Preprocessed (sent to server)", font=("", 8)).grid(
            row=3, column=0)

        # --- Control panel (right) ---
        ctrl = ttk.LabelFrame(root, text="Settings", padding=10)
        ctrl.grid(row=0, column=1, padx=10, pady=10, sticky="new")

        row = 0
        ttk.Label(ctrl, text="Model", font=("", 9, "bold")).grid(row=row, column=0, sticky="w", **p); row += 1
        cb = ttk.Combobox(ctrl, textvariable=self.current,
                          values=list(self.adapters.keys()), state="readonly", width=20)
        cb.grid(row=row, column=0, sticky="ew", **p)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())
        row += 1

        ttk.Label(ctrl, text="Server URL").grid(row=row, column=0, sticky="w", **p); row += 1
        self.url_var = tk.StringVar(value="http://127.0.0.1:5001")
        ttk.Entry(ctrl, textvariable=self.url_var, width=30).grid(row=row, column=0, sticky="ew", **p)
        row += 1

        ttk.Label(ctrl, text="CIFAR-10 Sample").grid(row=row, column=0, sticky="w", **p); row += 1
        nav = ttk.Frame(ctrl)
        ttk.Button(nav, text="◀", width=2, command=self._prev_sample).pack(side="left")
        self.sample_spin = ttk.Spinbox(nav, textvariable=self.sample_idx, from_=0,
                                       to=len(self.samples) - 1, width=6,
                                       command=self.show_sample)
        self.sample_spin.pack(side="left", padx=4)
        ttk.Button(nav, text="▶", width=2, command=self._next_sample).pack(side="left")
        ttk.Label(nav, text=f"/ {len(self.samples)-1}").pack(side="left", padx=4)
        nav.grid(row=row, column=0, sticky="w", **p)
        row += 1

        ttk.Label(ctrl, text="— or —", font=("", 8)).grid(row=row, column=0, **p); row += 1
        ttk.Button(ctrl, text="Load Custom Image...", command=self._load_custom).grid(
            row=row, column=0, sticky="ew", **p)
        row += 1

        self.custom_label = ttk.Label(ctrl, text="No custom image loaded", foreground="gray")
        self.custom_label.grid(row=row, column=0, sticky="w", **p)
        row += 1

        ttk.Separator(ctrl, orient="horizontal").grid(row=row, column=0, sticky="ew", pady=8)
        row += 1

        ttk.Button(ctrl, text="Encrypted Infer", command=self._start_single).grid(
            row=row, column=0, sticky="ew", **p)
        row += 1
        ttk.Button(ctrl, text="Batch Test (5 samples)", command=self._start_batch).grid(
            row=row, column=0, sticky="ew", **p)

        # --- Results panel (bottom) ---
        res = ttk.LabelFrame(root, text="Results", padding=10)
        res.grid(row=8, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        self.result_label = ttk.Label(res, textvariable=self.status_text,
                                      font=("Consolas", 10), justify="left")
        self.result_label.grid(row=0, column=0, sticky="w", **p)

        self.batch_bar = ttk.Progressbar(res, variable=self.batch_progress,
                                         maximum=100, length=420)
        self.batch_bar.grid(row=1, column=0, sticky="ew", **p)
        ttk.Label(res, textvariable=self.batch_status, font=("", 8)).grid(
            row=2, column=0, sticky="w", **p)

        root.columnconfigure(1, weight=1)

    # ================================================================
    #  Navigation & custom image
    # ================================================================

    def _on_model_change(self):
        self.adapter = self.adapters[self.current.get()]
        self.url_var.set(f"http://127.0.0.1:{self.adapter.default_port}")
        self.show_sample()

    def _prev_sample(self):
        if self.sample_idx.get() > 0:
            self.sample_idx.set(self.sample_idx.get() - 1)
            self.custom_image = None
            self.custom_label.config(text="No custom image loaded", foreground="gray")
            self.show_sample()

    def _next_sample(self):
        if self.sample_idx.get() < len(self.samples) - 1:
            self.sample_idx.set(self.sample_idx.get() + 1)
            self.custom_image = None
            self.custom_label.config(text="No custom image loaded", foreground="gray")
            self.show_sample()

    def _load_custom(self):
        path = filedialog.askopenfilename(
            title="Select a cat or dog image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.custom_image = load_image_from_disk(path)
            self.custom_label.config(text=f"Loaded: {path.split('/')[-1].split(chr(92))[-1][:40]}",
                                     foreground="green")
            self._display_image(self.custom_image)
        except Exception as e:
            messagebox.showerror("Load Failed", str(e))

    def show_sample(self):
        if self.custom_image is not None:
            self._display_image(self.custom_image)
        else:
            image, label = self.samples[self.sample_idx.get()]
            self._display_image(image.squeeze(0))
        self.status_text.set("Ready for inference")

    def _display_image(self, tensor_3ch):
        # Original
        pil = self._tensor_to_pil(tensor_3ch)
        self._tk_orig = ImageTk.PhotoImage(pil.resize(IMG_SIZE, Image.Resampling.NEAREST))
        self.img_orig_label.configure(image=self._tk_orig)
        # Preprocessed
        try:
            pp = self.adapter.preprocess_display(tensor_3ch)
            pil_pp = self._tensor_to_pil(pp)
            self._tk_pp = ImageTk.PhotoImage(pil_pp.resize(IMG_SIZE, Image.Resampling.NEAREST))
            self.img_pp_label.configure(image=self._tk_pp)
        except Exception:
            pass

    def _tensor_to_pil(self, tensor):
        if tensor.ndim == 2:
            arr = (tensor.numpy() * 255).clip(0, 255).astype("uint8")
            return Image.fromarray(arr, mode="L").convert("RGB")
        arr = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)

    def _get_image(self):
        """Return the current image as a 3×32×32 tensor."""
        if self.custom_image is not None:
            return self.custom_image
        image, _ = self.samples[self.sample_idx.get()]
        return image.squeeze(0)

    # ================================================================
    #  Single inference
    # ================================================================

    def _start_single(self):
        threading.Thread(target=self._run_single, daemon=True).start()

    def _run_single(self):
        try:
            self.status_text.set("Encrypting and sending...")
            self.batch_status.set("")
            self.batch_progress.set(0)

            img_3ch = self._get_image()
            plain_logits = self.adapter.plain_predict(img_3ch)
            plain_pred = int(plain_logits.argmax())

            payload = self.adapter.encrypt(self.context, img_3ch)
            req_bytes = len(json.dumps(payload).encode("utf-8"))

            resp = requests.post(
                f"{self.url_var.get().rstrip('/')}/infer",
                json=payload, timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()
            resp_bytes = len(json.dumps(result).encode("utf-8"))

            enc_logits = ts.ckks_vector_from(
                self.context, base64.b64decode(result["ciphertext"])
            )
            he_logits = enc_logits.decrypt()
            he_pred = int(max(range(len(he_logits)), key=lambda idx: he_logits[idx]))
            diff = sum(abs(float(a) - float(b)) for a, b in zip(plain_logits, he_logits))

            lines = [
                f"Plain pred: {CLASSES[plain_pred]}   HE pred: {CLASSES[he_pred]}",
                f"Match: {'PASS' if plain_pred == he_pred else 'MISMATCH'}",
                f"",
                f"Server time: {result['server_inference_seconds']:.4f}s",
                f"Traffic: {(req_bytes + resp_bytes) / (1024*1024):.2f} MB",
                f"Logit diff: {diff:.8f}",
                f"Plain logits: [{plain_logits[0]:.6f}, {plain_logits[1]:.6f}]",
                f"HE logits:    [{he_logits[0]:.6f}, {he_logits[1]:.6f}]",
            ]
            self.status_text.set("\n".join(lines))

        except requests.ConnectionError:
            messagebox.showerror("Connection Error",
                                 f"Cannot reach {self.url_var.get()}\n\n"
                                 "Start the server first:\n"
                                 "  python he_server.py      (Tiny,  port 5000)\n"
                                 "  python he_cnn_server.py  (CNN,   port 5001)")
            self.status_text.set("Connection failed")
        except Exception as exc:
            messagebox.showerror("Inference Failed", str(exc))
            self.status_text.set(f"Failed: {exc}")

    # ================================================================
    #  Batch test
    # ================================================================

    def _start_batch(self):
        threading.Thread(target=self._run_batch, daemon=True).start()

    def _run_batch(self, n=5):
        try:
            self.batch_progress.set(0)
            self.custom_image = None
            correct = total = 0
            for i, (image, label) in enumerate(self.samples):
                if i >= n:
                    break
                self.batch_status.set(f"Testing {i+1}/{n}...")
                self.batch_progress.set(int((i + 1) / n * 100))

                img_3ch = image.squeeze(0)
                label_val = int(label.item())
                try:
                    payload = self.adapter.encrypt(self.context, img_3ch)
                    resp = requests.post(f"{self.url_var.get().rstrip('/')}/infer",
                                         json=payload, timeout=300)
                    resp.raise_for_status()
                    result = resp.json()
                    enc_logits = ts.ckks_vector_from(
                        self.context, base64.b64decode(result["ciphertext"]))
                    he_logits = enc_logits.decrypt()
                    he_pred = int(max(range(len(he_logits)),
                                      key=lambda idx: he_logits[idx]))
                    correct += int(he_pred == label_val)
                except Exception:
                    pass
                total += 1
            acc = correct / total * 100 if total else 0
            self.batch_status.set(f"Done — {correct}/{total} correct ({acc:.1f}%)")
            self.status_text.set(f"Batch accuracy: {acc:.1f}% ({correct}/{total})")
        except requests.ConnectionError:
            messagebox.showerror("Connection Error", f"Cannot reach {self.url_var.get()}")
        except Exception as exc:
            messagebox.showerror("Batch Failed", str(exc))
        finally:
            self.batch_progress.set(0)


if __name__ == "__main__":
    root = tk.Tk()
    MLaaSClientApp(root)
    root.mainloop()
