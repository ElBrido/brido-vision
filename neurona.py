"""
==========================================================================
  N.E.U.R.O.N.A.  —  GENERATIVE RESONANT CORE  v10.1 (ULTRA LITE)
==========================================================================
"""
import os, sys, math, time, ctypes, subprocess, json, re, threading
from pathlib import Path
from copy import deepcopy

try:
    import onnx
    import onnxscript
except ImportError:
    print("[v10.2] Instalando librerías ONNX requeridas en Kaggle...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxscript", "-q"])

import numpy as np
import numpy.random as npr

def _pip(pkg, imp):
    try: exec(imp, globals())
    except ImportError:
        os.system(f"pip install -q '{pkg}'")
        exec(imp, globals())

_pip("Pillow",          "from PIL import Image, ImageDraw, ImageFilter")
_pip("huggingface_hub", "from huggingface_hub import HfApi, hf_hub_download")
_pip("torch",           "import torch; import torch.nn as nn; import torch.optim as optim")
_pip("torchvision",     "from torchvision import transforms")
_pip("datasets<3.0.0",  "import datasets")
_pip("sentence_transformers", "from sentence_transformers import SentenceTransformer")

import torch
import datasets
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import HfApi, hf_hub_download

try:
    from tensorflow.keras.datasets import cifar10, cifar100
except ImportError:
    os.system("pip install -q tensorflow-cpu")
    from tensorflow.keras.datasets import cifar10, cifar100

# CONFIG
HF_TOKEN     = os.environ.get("HF_TOKEN")
USERNAME     = "Bridoxd"
V8_REPO      = "brido-vision"
V9_REPO      = "brido-vision-v10"
if os.path.exists("/kaggle/working"):
    CKPT_DIR = Path("/kaggle/working/v10_ckpt")
else:
    CKPT_DIR = Path("./v10_ckpt")

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
D            = 16384
D_BYTES      = D // 8
SCALES       = [32, 16, 8, 4]
MAX_GRID     = 128
PATCH_OUT    = 16

OUT_SIZE        = 128
SMOOTH_EPOCHS   = 10  # Reducido porque OneCycle converge mas rapido
SMOOTH_LR       = 1e-3
ESRGAN_EPOCHS   = 15  # Reducido porque ESPCN converge muy rapido
ESRGAN_LR       = 1e-3
BATCH_SIZE      = 16
SAVE_EVERY_HF   = 20
SAVE_EVERY      = 1
RESET_ESRGAN    = True
FORCE_RESTART_SMOOTH = True
torch.set_num_threads(os.cpu_count() or 4)

# C NATIVE (Re-usado)
_HDC_C_SOURCE = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static inline int _hd(const uint8_t* a, const uint8_t* b, int nb) {
    int d = 0;
    const uint64_t* a64 = (const uint64_t*)a;
    const uint64_t* b64 = (const uint64_t*)b;
    for (int i = 0; i < nb/8; i++) d += __builtin_popcountll(a64[i]^b64[i]);
    return d;
}

void batch_nn_search(
    const uint8_t* proto_lsh, const uint8_t* proto_mat,
    int N, int LSH_B, int D_B,
    const uint8_t* query_lsh, const uint8_t* query_mat,
    int M, int lsh_slack,
    int32_t* out_idx, int32_t* out_ham
) {
    #pragma omp parallel for schedule(dynamic,4)
    for (int m = 0; m < M; m++) {
        const uint8_t* ql = query_lsh + (size_t)m * LSH_B;
        const uint8_t* qv = query_mat + (size_t)m * D_B;
        int min_lsh = LSH_B * 8;
        for (int n = 0; n < N; n++) {
            int d = _hd(ql, proto_lsh + (size_t)n*LSH_B, LSH_B);
            if (d < min_lsh) min_lsh = d;
        }
        int lsh_th = min_lsh + lsh_slack;
        int best_ham = D_B*8+1, best_idx = 0;
        for (int n = 0; n < N; n++) {
            int ld = _hd(ql, proto_lsh + (size_t)n*LSH_B, LSH_B);
            if (ld <= lsh_th) {
                int hd = _hd(qv, proto_mat + (size_t)n*D_B, D_B);
                if (hd < best_ham) { best_ham = hd; best_idx = n; }
            }
        }
        out_idx[m] = best_idx; out_ham[m] = best_ham;
    }
}

void batch_encode_scale(
    const uint8_t* ic,
    const uint8_t* in_patch,
    const uint8_t* global_grid,
    int H, int W, int s, int D_B,
    uint8_t* out_packed
) {
    int rows = H / s;
    int cols = W / s;
    int half = (s * s) / 2;
    int MAX_GRID = 128;

    #pragma omp parallel
    {
        int* sums = (int*)malloc((size_t)D_B * 8 * sizeof(int));

        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                memset(sums, 0, (size_t)D_B * 8 * sizeof(int));

                for (int py = 0; py < s; py++) {
                    for (int px = 0; px < s; px++) {
                        int img_y = r * s + py;
                        int img_x = c * s + px;
                        const uint8_t* ic_ptr = ic + (size_t)(img_y * W + img_x) * D_B;
                        const uint8_t* p_ptr  = in_patch + (size_t)(py * s + px) * D_B;

                        for (int b = 0; b < D_B; b++) {
                            uint8_t val = ic_ptr[b] ^ p_ptr[b];
                            sums[b*8 + 0] += (val >> 7) & 1;
                            sums[b*8 + 1] += (val >> 6) & 1;
                            sums[b*8 + 2] += (val >> 5) & 1;
                            sums[b*8 + 3] += (val >> 4) & 1;
                            sums[b*8 + 4] += (val >> 3) & 1;
                            sums[b*8 + 5] += (val >> 2) & 1;
                            sums[b*8 + 6] += (val >> 1) & 1;
                            sums[b*8 + 7] += (val >> 0) & 1;
                        }
                    }
                }

                uint8_t* out_ptr = out_packed + (size_t)(r * cols + c) * D_B;
                const uint8_t* g_ptr = global_grid + (size_t)(r * MAX_GRID + c) * D_B;

                for (int b = 0; b < D_B; b++) {
                    uint8_t res = 0;
                    if (sums[b*8 + 0] > half) res |= (1 << 7);
                    if (sums[b*8 + 1] > half) res |= (1 << 6);
                    if (sums[b*8 + 2] > half) res |= (1 << 5);
                    if (sums[b*8 + 3] > half) res |= (1 << 4);
                    if (sums[b*8 + 4] > half) res |= (1 << 3);
                    if (sums[b*8 + 5] > half) res |= (1 << 2);
                    if (sums[b*8 + 6] > half) res |= (1 << 1);
                    if (sums[b*8 + 7] > half) res |= 1;

                    out_ptr[b] = res ^ g_ptr[b];
                }
            }
        }
        free(sums);
    }
}

int get_num_threads(void) { return omp_get_max_threads(); }
"""

def _compile_c():
    src = Path("/tmp/hdc_v9.c"); so = Path("/tmp/hdc_v9.so")
    src.write_text(_HDC_C_SOURCE)
    ret = subprocess.run(["gcc","-O3","-march=native","-fopenmp",
                          "-shared","-fPIC","-o",str(so),str(src)],
                         capture_output=True, text=True)
    if ret.returncode != 0:
        ret = subprocess.run(["gcc","-O3","-march=native",
                              "-shared","-fPIC","-o",str(so),str(src)],
                             capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"[C] Error compilando: {ret.stderr}"); return None
    lib = ctypes.CDLL(str(so))
    lib.batch_nn_search.restype  = None
    lib.batch_nn_search.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.batch_encode_scale.restype = None
    lib.batch_encode_scale.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p
    ]
    lib.get_num_threads.restype = ctypes.c_int
    lib.get_num_threads.argtypes = []
    return lib

_CLIB = _compile_c()

def hamming_sim(a, b):
    bc = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    return 1.0 - bc[a ^ b[np.newaxis,:]].sum(axis=-1) / D

class BasisBank:
    def __init__(self, seed=42):
        rng = npr.default_rng(seed)
        self.r_b = self._gray(256, 1)
        self.g_b = self._gray(256, 2)
        self.b_b = self._gray(256, 3)
        self.in_patch = {}
        for s in SCALES:
            mat = np.zeros((s,s,D_BYTES), dtype=np.uint8)
            for y in range(s):
                for x in range(s):
                    v = np.zeros(D,dtype=np.uint8); v[:D//2]=1; rng.shuffle(v)
                    mat[y,x] = np.packbits(v)
            self.in_patch[s] = mat
        pos_rng = npr.default_rng(1337)
        self.global_grid = np.zeros((MAX_GRID,MAX_GRID,D_BYTES),dtype=np.uint8)
        for r in range(MAX_GRID):
            for c in range(MAX_GRID):
                v = np.zeros(D,dtype=np.uint8); v[:D//2]=1; pos_rng.shuffle(v)
                self.global_grid[r,c] = np.packbits(v)

    @staticmethod
    def _gray(levels, seed):
        rng = npr.default_rng(seed)
        v = np.zeros(D,dtype=np.uint8); v[:D//2]=1; rng.shuffle(v)
        bases = [np.packbits(v.copy())]
        idx = np.arange(D); rng.shuffle(idx)
        step = max(1, D//(2*(levels-1))); ptr = 0
        for _ in range(1,levels):
            if ptr+step <= D:
                v[idx[ptr:ptr+step]] = 1 - v[idx[ptr:ptr+step]]; ptr+=step
            bases.append(np.packbits(v.copy()))
        return np.array(bases)

    def get_global_pos(self, r, c):
        return self.global_grid[min(r,MAX_GRID-1), min(c,MAX_GRID-1)]

class VectorizedEncoder:
    def __init__(self, basis):
        self.basis = basis

    def encode_image(self, img_rgb, scales=None):
        if scales is None: scales = SCALES
        H,W = img_rgb.shape[:2]
        img_i = img_rgb.astype(np.int32)
        ic_bytes = self.basis.r_b[img_i[:,:,0]] ^ self.basis.g_b[img_i[:,:,1]] ^ self.basis.b_b[img_i[:,:,2]]
        ic_c = np.ascontiguousarray(ic_bytes)

        results = {}
        for s in scales:
            rows,cols = H//s, W//s
            out_packed = np.zeros((rows, cols, D_BYTES), dtype=np.uint8)
            if _CLIB:
                _CLIB.batch_encode_scale(
                    ic_c.ctypes.data_as(ctypes.c_void_p),
                    np.ascontiguousarray(self.basis.in_patch[s]).ctypes.data_as(ctypes.c_void_p),
                    np.ascontiguousarray(self.basis.global_grid).ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(H), ctypes.c_int(W), ctypes.c_int(s), ctypes.c_int(D_BYTES),
                    out_packed.ctypes.data_as(ctypes.c_void_p)
                )
            results[s] = out_packed
        return results

    def extract_patches_rgb(self, img, s, patch_out=16):
        H,W = img.shape[:2]
        rows,cols = H//s, W//s
        target_H, target_W = rows * patch_out, cols * patch_out

        if target_H != H or target_W != W:
            img_res = np.array(Image.fromarray(img).resize((target_W, target_H), Image.BILINEAR))
        else:
            img_res = img

        return img_res.reshape(rows, patch_out, cols, patch_out, 3).transpose(0,2,1,3,4).reshape(-1, patch_out, patch_out, 3)

class ResonantMemory:
    LSH_B = 32
    def __init__(self, max_n=120_000):
        self._n = 0; self.max_n = max_n
        self.proto_mat = np.zeros((max_n,D_BYTES), dtype=np.uint8)
        self.proto_rgb = np.zeros((max_n,PATCH_OUT,PATCH_OUT,3), dtype=np.uint8)
        rng = npr.default_rng(9999)
        self._lsh_proj = rng.integers(0, D, size=self.LSH_B, dtype=np.int32)
        self.hash_table = {}

    def get_count(self): return self._n

    def _lsh(self, v):
        unpacked = np.unpackbits(v, axis=-1)
        return unpacked[..., self._lsh_proj]

    def load(self, path):
        d = np.load(path, allow_pickle=True)
        n = 0
        if "n" in d:
            try:
                val = d["n"]
                if hasattr(val, 'item'):
                    try: val = val.item()
                    except: pass
                if isinstance(val, (list, np.ndarray)) and len(val) > 0: val = val[0]
                n = int(val)
            except:
                pass
        elif "meta" in d:
            try:
                meta = d["meta"]
                if hasattr(meta, 'item'):
                    try: meta = meta.item()
                    except: pass
                if isinstance(meta, dict) and "n" in meta:
                    n = int(meta["n"])
                elif isinstance(meta, (list, np.ndarray)) and len(meta) > 0:
                    first = meta[0]
                    if hasattr(first, 'item'): first = first.item()
                    if isinstance(first, dict) and "n" in first:
                        n = int(first["n"])
            except:
                pass

        self._n = min(n, self.max_n)
        if "proto_mat" in d: self.proto_mat[:self._n] = d["proto_mat"][:self._n]
        if "proto_rgb" in d: self.proto_rgb[:self._n] = d["proto_rgb"][:self._n]

        self.hash_table = {}
        if self._n > 0:
            lsh_bits = self._lsh(self.proto_mat[:self._n])
            keys = np.ascontiguousarray(np.packbits(lsh_bits, axis=-1)).view(np.uint32).flatten()
            for i in range(self._n):
                self.hash_table.setdefault(keys[i], []).append(i)

    def add(self, vec: np.ndarray, patch_rgb: np.ndarray = None):
        if self._n < self.max_n:
            idx = self._n
            self._n += 1
        else:
            idx = np.random.randint(0, self.max_n)
        self.proto_mat[idx] = vec

        key = np.ascontiguousarray(np.packbits(self._lsh(vec))).view(np.uint32)[0]
        self.hash_table.setdefault(key, []).append(idx)

        if patch_rgb is not None:
            self.proto_rgb[idx] = patch_rgb

    def add_batch(self, vecs: np.ndarray, patches_rgb: np.ndarray):
        num = vecs.shape[0]
        if self._n >= self.max_n: return

        allowed = min(num, self.max_n - self._n)
        start, end = self._n, self._n + allowed

        self.proto_mat[start:end] = vecs[:allowed]
        self.proto_rgb[start:end] = patches_rgb[:allowed]

        lsh_bits = self._lsh(vecs[:allowed])
        keys = np.ascontiguousarray(np.packbits(lsh_bits, axis=-1)).view(np.uint32).flatten()
        for i, k in enumerate(keys):
            self.hash_table.setdefault(k, []).append(start + i)

        self._n += allowed

    def query_batch(self, q_mat, lsh_slack=256):
        M = q_mat.shape[0]; N = self._n
        out_idx = np.zeros(M, dtype=np.int32)
        if N == 0: return out_idx

        q_lsh = self._lsh(q_mat)
        q_keys = np.ascontiguousarray(np.packbits(q_lsh, axis=-1)).view(np.uint32).flatten()

        for i in range(M):
            key = q_keys[i]
            candidates = self.hash_table.get(key)

            if not candidates:
                for bit in range(self.LSH_B):
                    neighbor_key = key ^ (1 << bit)
                    candidates = self.hash_table.get(neighbor_key)
                    if candidates: break

            if not candidates:
                out_idx[i] = np.random.randint(0, N)
            else:
                c_idx = np.array(candidates)
                cand_vecs = self.proto_mat[c_idx]
                sims = hamming_sim(cand_vecs, q_mat[i])
                best = np.argmax(sims)
                out_idx[i] = c_idx[best]

        return out_idx

def _build_recon_lut():
    lut = {}
    for s in SCALES:
        if s == PATCH_OUT: lut[s] = lambda p: p
        elif s > PATCH_OUT:
            f = s//PATCH_OUT
            lut[s] = lambda p,f=f: np.repeat(np.repeat(p,f,axis=1),f,axis=2)
        else:
            f = PATCH_OUT//s
            lut[s] = lambda p,s=s,f=f: p.reshape(p.shape[0],s,f,s,f,3).mean(axis=(2,4)).astype(np.uint8)
    return lut

_RECON_LUT = _build_recon_lut()

def reconstruct(encoded, memory_bank, basis, out_h=32, out_w=32):
    canvas = np.zeros((out_h,out_w,3), dtype=np.float32)
    weight = np.zeros((out_h,out_w), dtype=np.float32)
    recon_scales = {s:v for s,v in encoded.items() if s <= 8}
    for s, vec_mat in sorted(recon_scales.items()):
        mem = memory_bank.get(s)
        if mem is None or mem.get_count() == 0: continue
        rows,cols = vec_mat.shape[:2]; M = rows*cols
        gpos = basis.global_grid[:rows,:cols]
        q_mat = np.ascontiguousarray((vec_mat^gpos).reshape(M,-1))
        b_idx = mem.query_batch(q_mat).reshape(rows,cols)
        w_s = 1.0/s
        for pr in range(rows):
            for pc in range(cols):
                p8 = mem.proto_rgb[b_idx[pr,pc]]
                if s != PATCH_OUT: p8 = _RECON_LUT[s](p8[np.newaxis])[0]
                y0,y1 = pr*s, min((pr+1)*s,out_h)
                x0,x1 = pc*s, min((pc+1)*s,out_w)
                ph,pw = y1-y0,x1-x0
                canvas[y0:y1,x0:x1] += p8[:ph,:pw].astype(np.float32)*w_s
                weight[y0:y1,x0:x1] += w_s
    return np.clip(canvas/(weight[...,np.newaxis]+1e-8),0,255).astype(np.uint8)

# ══════════════════════════════════════════════════════════════════════════
# REDES NEURONALES LIGERAS (MobileNet Style + ESPCN)
# ══════════════════════════════════════════════════════════════════════════
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SmoothDecoder(nn.Module):
    """
    Decodificador súper ligero (90% menos parámetros) usando Depthwise Convs
    """
    def __init__(self, ch=32, n_res=5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        layers = []
        for _ in range(n_res):
            layers.append(nn.Sequential(
                DepthwiseSeparableConv(ch, ch),
                nn.InstanceNorm2d(ch),
                nn.ReLU(inplace=True)
            ))
        self.body = nn.ModuleList(layers)

        self.tail = nn.Sequential(
            DepthwiseSeparableConv(ch, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(self,x):
        h = self.head(x)
        for layer in self.body:
            h = h + layer(h)
        delta = torch.tanh(self.tail(h)) * 0.3
        return torch.clamp(x + delta, 0, 1)

class TinyESPCN(nn.Module):
    """
    Reemplazo de ESRGAN por ESPCN: NO sufre de Mode Collapse,
    es ultra rápido, usa PixelShuffle.
    Super-resolución x4: 32×32 → 128×128.
    """
    def __init__(self, scale_factor=4, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=5, padding=2),
            nn.Tanh(),
            DepthwiseSeparableConv(ch, ch),
            nn.Tanh(),
            DepthwiseSeparableConv(ch, ch),
            nn.Tanh(),
            nn.Conv2d(ch, 3 * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self,x):
        # Generar el output directo con residual
        x_up = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)
        out = self.net(x)
        return torch.clamp(x_up + out, 0, 1)

# ══════════════════════════════════════════════════════════════════════════
# TRADUCTOR SEMÁNTICO LIGERO (MINI-LM)
# ══════════════════════════════════════════════════════════════════════════
class MiniLM2HDC:
    """
    Reemplazo ultra-ligero para CLIP usando MiniLM.
    Ideal para gama baja: 20x más rápido, consume muy poca RAM.
    """
    def __init__(self, basis: BasisBank):
        self.basis = basis
        print("[v10.1] Cargando modelo MiniLM ultra ligero...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

        # Proyecta 384-dim (MiniLM) a 16384-dim (HDC)
        rng = npr.default_rng(777)
        self.proj = rng.standard_normal((384, D), dtype=np.float32)

    def encode(self, text: str) -> np.ndarray:
        if not text.strip(): return np.zeros(D_BYTES, dtype=np.uint8)
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list) -> np.ndarray:
        valid = [t if t and t.strip() else "image" for t in texts]
        results = []
        BS = 256 # Batch size muy alto porque MiniLM no pesa nada
        for i in range(0, len(valid), BS):
            chunk = valid[i:i+BS]
            with torch.no_grad():
                feats = self.model.encode(chunk, convert_to_numpy=True)
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            feats = feats / norms
            hdc_floats = feats @ self.proj
            hdc_bits   = (hdc_floats > 0).astype(np.uint8)
            packed = np.packbits(hdc_bits, axis=1)
            results.append(packed)
        return np.concatenate(results, axis=0)

    def encode_weighted(self, text: str) -> np.ndarray:
        return self.encode(text)

class ReferenceEditor:
    def __init__(self, encoder, memory_bank, basis,
                 smooth: SmoothDecoder, esrgan: TinyESPCN,
                 text_enc: MiniLM2HDC):
        self.encoder    = encoder
        self.memory     = memory_bank
        self.basis      = basis
        self.smooth     = smooth.eval()
        self.esrgan     = esrgan.eval()
        self.text_enc   = text_enc
        self._to_tensor = transforms.ToTensor()
        self._to_pil    = transforms.ToPILImage()

    def _img_to_np(self, img) -> np.ndarray:
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
            return np.array(img)
        return np.array(img)

    def _decode_pipeline(self, encoded: dict, hw=(32,32)) -> np.ndarray:
        recon = reconstruct(encoded, self.memory, self.basis, *hw)
        x = torch.tensor(recon/255.0).float().permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            smooth_out = self.smooth(x)
            if self.esrgan is not None:
                out = self.esrgan(smooth_out)
            else:
                out = F.interpolate(smooth_out, scale_factor=4, mode="bilinear", align_corners=False)
        out_np = (out.squeeze(0).permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)
        return out_np

    def edit_with_references(self, original, ref1, ref2, alpha=0.65,
                              text_hint="") -> np.ndarray:
        orig_np  = self._img_to_np(original)
        ref1_np  = self._img_to_np(ref1)
        ref2_np  = self._img_to_np(ref2)

        enc_orig = self.encoder.encode_image(orig_np)
        enc_ref1 = self.encoder.encode_image(ref1_np)
        enc_ref2 = self.encoder.encode_image(ref2_np)

        text_vec = self.text_enc.encode(text_hint) if text_hint else None

        edited = {}
        rng = npr.default_rng(42)
        for s in SCALES:
            if s not in enc_orig: continue
            orig_s = enc_orig[s]
            ref1_s = enc_ref1[s] if enc_ref1[s].shape == orig_s.shape else orig_s
            ref2_s = enc_ref2[s] if enc_ref2[s].shape == orig_s.shape else orig_s

            edit_vec = ref2_s ^ ref1_s

            mask = (rng.random(orig_s.shape[:-1] + (1,)) < alpha).astype(np.uint8)
            mask = np.broadcast_to(mask, orig_s.shape)

            edited_s = orig_s ^ (edit_vec & mask)

            if text_vec is not None:
                txt_mask = (rng.random(orig_s.shape[:-1]+(1,)) < 0.2).astype(np.uint8)
                txt_mask = np.broadcast_to(txt_mask, orig_s.shape)
                edited_s = edited_s ^ (text_vec[np.newaxis,np.newaxis,:] & txt_mask)

            edited[s] = edited_s

        return self._decode_pipeline(edited, orig_np.shape[:2])

    def edit_with_text(self, original, text: str, strength=0.25) -> np.ndarray:
        orig_np = self._img_to_np(original)
        enc_orig = self.encoder.encode_image(orig_np)
        text_vec = self.text_enc.encode_weighted(text)

        edited = {}
        rng = npr.default_rng(77)
        for s, vec_mat in enc_orig.items():
            mask = (rng.random(vec_mat.shape[:-1]+(1,)) < strength).astype(np.uint8)
            mask = np.broadcast_to(mask, vec_mat.shape)
            edited[s] = vec_mat ^ (text_vec[np.newaxis,np.newaxis,:] & mask)

        return self._decode_pipeline(edited, orig_np.shape[:2])

    def generate_from_text(self, text: str, seed=None) -> np.ndarray:
        if seed is None: seed = int(time.time()) % 10000
        rng = npr.default_rng(seed)

        text_vec = self.text_enc.encode_weighted(text)

        encoded = {}
        for s in [4, 8]:
            mem = self.memory.get(s)
            if mem is None or mem.get_count() == 0: continue
            rows,cols = 128//s, 128//s; M = rows*cols

            noise_mask = (rng.random(D_BYTES) < 0.05).astype(np.uint8)
            varied_vec = text_vec ^ noise_mask

            q_mat = np.tile(varied_vec[np.newaxis,:], (M,1))
            pos_noise = (rng.random((M,D_BYTES)) < 0.02).astype(np.uint8)
            q_mat = q_mat ^ pos_noise

            idx = mem.query_batch(np.ascontiguousarray(q_mat))
            rows_mat = np.zeros((rows,cols,D_BYTES), dtype=np.uint8)
            for i,(pr,pc) in enumerate([(r,c) for r in range(rows) for c in range(cols)]):
                rows_mat[pr,pc] = mem.proto_mat[idx[i]]
            gpos = self.basis.global_grid[:rows,:cols]
            encoded[s] = rows_mat ^ gpos

        if not encoded:
            return np.full((512,512,3), 128, dtype=np.uint8)

        return self._decode_pipeline(encoded, hw=(128, 128))

@torch.jit.script
def simulate_hdc_noise_jit(img_t: torch.Tensor) -> torch.Tensor:
    H, W = img_t.shape[1], img_t.shape[2]
    block = 8 if torch.rand(1).item() > 0.5 else 4

    img_down = img_t[:, ::block, ::block]
    img_up = img_down.repeat_interleave(block, dim=1).repeat_interleave(block, dim=2)
    img_up = img_up[:, :H, :W]

    noise_mask = torch.rand(1, H, W) < 0.05
    noise_vals = torch.rand(3, H, W)
    img_up = torch.where(noise_mask, noise_vals, img_up)

    img_up = torch.round(img_up * 16.0) / 16.0
    return img_up

class SyntheticHDCDataset(Dataset):
    def __init__(self, images, augment=True, max_samples=None):
        self.augment = augment
        self._rng = npr.default_rng(42)
        n = max_samples if max_samples else len(images)
        indices = np.random.choice(len(images), min(n, len(images)), replace=False)

        self.cache_orig = [images[i] for i in indices]
        self.to_t = transforms.ToTensor()

    def __len__(self): return len(self.cache_orig)

    def simulate_hdc_noise(self, img_t):
        return simulate_hdc_noise_jit(img_t)

    def __getitem__(self, idx):
        img_128 = self.cache_orig[idx]
        img_32 = np.array(Image.fromarray(img_128).resize((32,32), Image.BILINEAR))
        x_real = self.to_t(Image.fromarray(img_32))

        if self.augment and self._rng.random() > 0.5:
            x_real = x_real.flip(-1)
        if self.augment:
            delta = float(self._rng.integers(-15, 16)) / 255.0
            x_real = (x_real + delta).clamp(0, 1)

        x_hdc = self.simulate_hdc_noise(x_real)
        return x_hdc, x_real

class SRDataset(Dataset):
    def __init__(self, images, smooth_model, memory_bank, encoder, basis,
                 out_size=128, max_samples=None):
        self.out_sz  = out_size
        self.to_t    = transforms.ToTensor()
        self._rng    = npr.default_rng(99)

        n = max_samples if max_samples else len(images)
        indices = np.random.choice(len(images), min(n, len(images)), replace=False)

        from concurrent.futures import ThreadPoolExecutor
        recons_list = [None] * len(indices)
        self.cache_hi = [None] * len(indices)

        def _job(idx_k):
            img_native = images[indices[idx_k]]
            img_32 = np.array(Image.fromarray(img_native).resize((32,32), Image.BILINEAR))
            img_128 = np.array(Image.fromarray(img_native).resize((128,128), Image.BILINEAR))
            enc = encoder.encode_image(img_32)
            recon = reconstruct(enc, memory_bank, basis, 32, 32)
            return idx_k, recon, img_128

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            results = executor.map(_job, range(len(indices)))
            for idx_k, recon, img_128 in results:
                recons_list[idx_k] = recon
                self.cache_hi[idx_k] = self.to_t(img_128)

        self.cache_smooth = [None] * len(indices)
        smooth_model.eval()

        BS = 16
        for i in range(0, len(indices), BS):
            chunk = recons_list[i : i+BS]
            tensors = [self.to_t(r) for r in chunk]
            batch_x = torch.stack(tensors).to(DEVICE)
            with torch.no_grad():
                batch_out = smooth_model(batch_x)
            for j in range(len(chunk)):
                self.cache_smooth[i + j] = batch_out[j].cpu()

    def __len__(self): return len(self.cache_smooth)

    def __getitem__(self, idx):
        x_lo = self.cache_smooth[idx]
        x_hi = self.cache_hi[idx]
        if self._rng.random() > 0.5:
            x_lo = x_lo.flip(-1)
            x_hi = x_hi.flip(-1)
        return x_lo, x_hi

class SSIMLoss(nn.Module):
    def __init__(self, win=7):
        super().__init__(); self.win = win

    def forward(self, x, y):
        w = self.win; p = w//2
        mu_x = F.avg_pool2d(x,w,1,p)
        mu_y = F.avg_pool2d(y,w,1,p)
        sx = F.avg_pool2d(x*x,w,1,p) - mu_x**2
        sy = F.avg_pool2d(y*y,w,1,p) - mu_y**2
        sxy= F.avg_pool2d(x*y,w,1,p) - mu_x*mu_y
        C1,C2 = 0.01**2, 0.03**2
        ssim = ((2*mu_x*mu_y+C1)*(2*sxy+C2)) / ((mu_x**2+mu_y**2+C1)*(sx+sy+C2))
        return 1 - ssim.mean()

class CombinedLoss(nn.Module):
    def __init__(self, l1_w=0.7, ssim_w=0.3):
        super().__init__()
        self.l1   = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.l1_w = l1_w; self.ssim_w = ssim_w

    def forward(self, pred, target):
        return self.l1_w*self.l1(pred,target) + self.ssim_w*self.ssim(pred,target)

# ENTRENAMIENTO ONE-CYCLE
def train_smooth_decoder(model, dataset, epochs=SMOOTH_EPOCHS,
                          lr=SMOOTH_LR, save_path=None,
                          hf_save_fn=None, save_every_hf=SAVE_EVERY_HF,
                          preview_fn=None,
                          grad_accum=2):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)
    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ⚡ ONE-CYCLE LEARNING RATE: Convergencia x3 mas rapida
    steps = (len(loader) + grad_accum - 1) // grad_accum
    if steps == 0: steps = 1
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr*5, steps_per_epoch=steps, epochs=epochs)

    loss_fn = CombinedLoss()
    model.train()

    for epoch in range(1, epochs+1):
        t0 = time.time(); total = 0.0
        opt.zero_grad()
        for b_idx, (x_hdc, x_real) in enumerate(loader):
            x_hdc = x_hdc.to(DEVICE)
            x_real = x_real.to(DEVICE)

            pred = model(x_hdc)
            loss = loss_fn(pred, x_real) / grad_accum

            loss.backward()

            if (b_idx + 1) % grad_accum == 0 or (b_idx + 1) == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

            total += loss.item() * grad_accum

        avg = total/len(loader)
        print(f"[SmoothDecoder] Epoch {epoch:3d}/{epochs} | Loss={avg:.4f} | t={time.time()-t0:.1f}s")
        if preview_fn: preview_fn(f"smooth_epoch_{epoch}")

    return model._orig_mod if hasattr(model, '_orig_mod') else model


def train_esrgan(model, dataset, epochs=ESRGAN_EPOCHS,
                  lr=ESRGAN_LR, save_path=None,
                  hf_save_fn=None, save_every_hf=SAVE_EVERY_HF,
                  preview_fn=None,
                  grad_accum=4):
    loader = DataLoader(dataset, batch_size=max(4, BATCH_SIZE//4), shuffle=True,
                        num_workers=0, pin_memory=False)
    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = (len(loader) + grad_accum - 1) // grad_accum
    if steps == 0: steps = 1
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr*3, steps_per_epoch=steps, epochs=epochs)
    loss_fn = CombinedLoss(l1_w=0.8, ssim_w=0.2)
    model.train()

    for epoch in range(1, epochs+1):
        t0 = time.time(); total = 0.0
        opt.zero_grad()
        for b_idx, (x_lo, x_hi) in enumerate(loader):
            x_lo = x_lo.to(DEVICE)
            x_hi = x_hi.to(DEVICE)
            pred = model(x_lo)
            loss = loss_fn(pred, x_hi) / grad_accum

            loss.backward()

            if (b_idx + 1) % grad_accum == 0 or (b_idx + 1) == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

            total += loss.item() * grad_accum

        avg = total/len(loader)
        print(f"[ESPCN] Epoch {epoch:3d}/{epochs} | Loss={avg:.4f} | t={time.time()-t0:.1f}s")
        if preview_fn: preview_fn(f"esrgan_epoch_{epoch}")

    return model._orig_mod if hasattr(model, '_orig_mod') else model

def load_v8_checkpoint(memory_bank, hf_token, username, repo):
    api = HfApi(token=hf_token)
    repo_id = f"{username}/{repo}"
    print(f"[v9] Cargando checkpoint v8.0 desde {repo_id}...")
    for s, mem in memory_bank.items():
        fname = f"mem_s{s}.npz"
        try:
            path = hf_hub_download(repo_id=repo_id, filename=fname,
                                   token=hf_token, local_dir=str(CKPT_DIR))
            mem.load(path)
        except Exception as e:
            print(f"  [HF] mem_s{s}: {e}")

def save_v9_checkpoint(smooth, esrgan, hf_token, username, repo):
    sp = CKPT_DIR/"smooth_decoder.pt"
    ep = CKPT_DIR/"tiny_esrgan.pt"
    save_checkpoint_async(smooth, esrgan, hf_token, username, repo,
                          upload_hf=True, save_path_smooth=str(sp), save_path_esrgan=str(ep))

def save_checkpoint_async(smooth, esrgan, hf_token, username, repo, upload_hf=False, save_path_smooth=None, save_path_esrgan=None):
    sd_smooth = {k: v.cpu().clone() for k, v in smooth.state_dict().items()} if smooth else None
    sd_esrgan = {k: v.cpu().clone() for k, v in esrgan.state_dict().items()} if esrgan else None
    def task():
        try:
            CKPT_DIR.mkdir(exist_ok=True)
            if sd_smooth and save_path_smooth:
                torch.save(sd_smooth, save_path_smooth)
            if sd_esrgan and save_path_esrgan:
                torch.save(sd_esrgan, save_path_esrgan)
            if upload_hf:
                api = HfApi(token=hf_token)
                repo_id = f"{username}/{repo}"
                try: api.create_repo(repo_id, exist_ok=True, token=hf_token)
                except: pass
                if sd_smooth and save_path_smooth:
                    try: api.upload_file(path_or_fileobj=save_path_smooth, path_in_repo=Path(save_path_smooth).name, repo_id=repo_id, token=hf_token)
                    except: pass
                if sd_esrgan and save_path_esrgan:
                    try: api.upload_file(path_or_fileobj=save_path_esrgan, path_in_repo=Path(save_path_esrgan).name, repo_id=repo_id, token=hf_token)
                    except: pass
        except: pass
    threading.Thread(target=task, daemon=True).start()

def load_v9_checkpoint(smooth, esrgan, hf_token, username, repo, force_restart_smooth=False):
    api = HfApi(token=hf_token)
    repo_id = f"{username}/{repo}"
    models_to_load = []
    if smooth and not force_restart_smooth: models_to_load.append((smooth, "smooth_decoder.pt"))
    if esrgan: models_to_load.append((esrgan, "tiny_esrgan.pt"))
    for model, fname in models_to_load:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=fname, token=hf_token, local_dir=str(CKPT_DIR))
            model.load_state_dict(torch.load(path, map_location="cpu"))
        except: pass

def save_training_preview(smooth, esrgan, sample_imgs, memory_bank, encoder, basis, step, out_dir=CKPT_DIR):
    out_dir.mkdir(exist_ok=True)
    smooth.eval(); esrgan.eval()
    to_t = transforms.ToTensor()
    panels = []
    for img_arr in sample_imgs[:4]:
        img_32 = np.array(Image.fromarray(img_arr).resize((32,32), Image.BILINEAR))
        enc   = encoder.encode_image(img_32)
        recon = reconstruct(enc, memory_bank, basis, 32, 32)
        x = to_t(recon).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            sm  = smooth(x)
            esr = esrgan(sm) if esrgan else sm
        out_np = (esr.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
        sz = 128
        orig_r  = Image.fromarray(img_arr).resize((sz,sz), Image.NEAREST)
        sm_r    = Image.fromarray((sm.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)).resize((sz,sz), Image.NEAREST)
        esr_r   = Image.fromarray(out_np)
        panel   = Image.new("RGB",(sz*3+8,sz),(20,20,20))
        panel.paste(orig_r,(0,0)); panel.paste(sm_r,(sz+4,0)); panel.paste(esr_r,(sz*2+8,0))
        panels.append(panel)
    if not panels:
        return
    pw,ph = panels[0].size
    grid  = Image.new("RGB",(pw*2+4,ph*2+24),(15,15,15))
    for i,(p,pos) in enumerate(zip(panels,[(0,24),(pw+4,24),(0,ph+28),(pw+4,ph+28)])): grid.paste(p,pos)
    ImageDraw.Draw(grid).text((4,4), f"v10.1 LITE | step={step} | orig | smooth | esrgan", fill=(200,200,200))
    grid.save(str(out_dir/f"preview_v10_{step}.png"))
    smooth.train(); esrgan.train()


def main():
    CKPT_DIR.mkdir(exist_ok=True)

    print("\n[v10.1] ══ NEURON-A v10.1 Generative Resonant Core (LITE) ══")
    basis   = BasisBank(seed=42)
    encoder = VectorizedEncoder(basis)

    memory_bank = {
        32: ResonantMemory(max_n=50000),
        16: ResonantMemory(max_n=50000),
        8:  ResonantMemory(max_n=50000),
        4:  ResonantMemory(max_n=50000),
    }

    all_imgs = []
    all_prompts = []

    # Crear modelos
    smooth = SmoothDecoder(ch=32, n_res=5).to(DEVICE)
    esrgan = TinyESPCN(scale_factor=4, ch=32).to(DEVICE)
    text_enc = MiniLM2HDC(basis)

    n_params_smooth = sum(p.numel() for p in smooth.parameters())
    n_params_esrgan = sum(p.numel() for p in esrgan.parameters())
    print(f"  SmoothDecoder: {n_params_smooth/1e6:.2f}M parámetros (vs 1.8M original)")
    print(f"  TinyESPCN:     {n_params_esrgan/1e6:.2f}M parámetros (vs 2.8M original)")

    print("\n[v10.1] Descargando CIFAR-10 y CIFAR-100...")
    try:
        datasets.disable_progress_bar()
        ds10 = datasets.load_dataset("cifar10")
        ds100 = datasets.load_dataset("cifar100")

        cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        for split in ['train', 'test']:
            for item in ds10[split]:
                if len(all_imgs) < 20000:
                    all_imgs.append(np.array(item['img'].convert('RGB').resize((128,128), Image.BILINEAR)))
                    all_prompts.append(cifar10_labels[item['label']])

            for item in ds100[split]:
                if len(all_imgs) < 20000:
                    all_imgs.append(np.array(item['img'].convert('RGB').resize((128,128), Image.BILINEAR)))
                    all_prompts.append("object")
    except Exception as e:
        print(f"  [Error] Falló descarga de CIFAR: {e}")

    def resize_to_np(img):
        if img.mode != 'RGB': img = img.convert('RGB')
        return np.array(img.resize((128,128), Image.BILINEAR))

    print("[v10.1] Descargando Maysee/tiny-imagenet...")
    try:
        datasets.disable_progress_bar()
        tiny_ds = datasets.load_dataset("Maysee/tiny-imagenet", split="train")
        for img in tiny_ds['image']:
            if len(all_imgs) >= 40000: break
            all_imgs.append(resize_to_np(img))
            all_prompts.append("photo")
    except Exception as e:
        print(f"[v10.1] Error cargando Tiny-ImageNet: {e}")


    print(f"[v10.1] DATASET SEMÁNTICO MASIVO LISTO: {len(all_imgs):,} imágenes")

    if not all_imgs:
        print("[v10.1] (Simulación) Cargando datasets semánticos masivos (Fall back a random)...")
        # Generar data dummy local para evitar fallos si no hay datasets
        all_imgs = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(16)]
        all_prompts = ["photo"] * 16

    sample_imgs = [all_imgs[i].copy() for i in [0, min(50000, len(all_imgs)-1), min(100000, len(all_imgs)-1), len(all_imgs)-1]]

    # ── 5. Fase 0: Fusión Semántica en Memoria ────────────────────────
    print("\n[v10.1] ══ FASE 0: Fusión Semántica en Memoria (TURBO) ══")
    unique_texts = list(set(all_prompts[:20000]))
    print(f"  → Ejecutando CLIP batch sobre {len(unique_texts):,} textos únicos...")
    unique_vecs = text_enc.encode_batch(unique_texts)
    txt_vec_map = {t: unique_vecs[k] for k, t in enumerate(unique_texts)}

    t0_mem = time.time()
    n_cores = os.cpu_count() or 4

    from concurrent.futures import ThreadPoolExecutor
    def _encode_img(args):
        img_128, txt_vec = args
        img_32 = np.array(Image.fromarray(img_128).resize((32,32), Image.BILINEAR))
        enc = encoder.encode_image(img_32)
        result = {}
        for s, vec_mat in enc.items():
            fused_vecs = (vec_mat ^ txt_vec).reshape(-1, D_BYTES)
            patches_rgb = encoder.extract_patches_rgb(img_32, s, PATCH_OUT)
            result[s] = (fused_vecs, patches_rgb)
        return result

    limit = min(20000, len(all_imgs))
    batch_args = [(all_imgs[k], txt_vec_map[all_prompts[k]]) for k in range(limit)]

    with ThreadPoolExecutor(max_workers=n_cores) as ex:
        for result in ex.map(_encode_img, batch_args):
            for s, (fused_vecs, patches_rgb) in result.items():
                if memory_bank[s]._n < memory_bank[s].max_n:
                    memory_bank[s].add_batch(fused_vecs, patches_rgb)

    print(f"[v10.1] ¡Memoria Construida! Prototipos totales: {sum(m.get_count() for m in memory_bank.values()):,}")

    print("\n[v10.1] ══ FASE 1: Entrenando SmoothDecoder (LITE) ══")

    smooth_ds = SyntheticHDCDataset(all_imgs, augment=True, max_samples=1500)

    def hf_save_now():
        save_v9_checkpoint(smooth, esrgan, HF_TOKEN, USERNAME, V9_REPO)

    def preview_and_upload(step_name):
        save_training_preview(smooth, esrgan, sample_imgs, memory_bank,
                               encoder, basis, step=step_name)

    smooth = train_smooth_decoder(
        smooth, smooth_ds, epochs=SMOOTH_EPOCHS, lr=SMOOTH_LR,
        save_path=str(CKPT_DIR/"smooth_decoder.pt"),
        hf_save_fn=hf_save_now, save_every_hf=SAVE_EVERY_HF,
        preview_fn=preview_and_upload
    )

    print("\n[v10.1] ══ FASE 2: Entrenando TinyESPCN (LITE) ══")
    sr_ds = SRDataset(all_imgs, smooth, memory_bank, encoder, basis,
                       out_size=OUT_SIZE, max_samples=1500)
    esrgan = train_esrgan(
        esrgan, sr_ds, epochs=ESRGAN_EPOCHS, lr=ESRGAN_LR,
        save_path=str(CKPT_DIR/"tiny_esrgan.pt"),
        hf_save_fn=hf_save_now, save_every_hf=SAVE_EVERY_HF,
        preview_fn=preview_and_upload
    )

    print("\n[v10.1] ══ DEMO DE INFERENCIA ══")
    editor = ReferenceEditor(encoder, memory_bank, basis, smooth, esrgan, text_enc)
    gen1 = editor.generate_from_text("dog running forest", seed=42)
    Image.fromarray(gen1).save(str(CKPT_DIR/"demo_gen_dog.png"))

if __name__ == "__main__":
    main()
