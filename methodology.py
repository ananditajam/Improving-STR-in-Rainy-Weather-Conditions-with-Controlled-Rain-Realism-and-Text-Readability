"""
MULTI-IMAGE GLOBAL RAIN SYNTHESIS (Non-patch) — Pure Bilinear Version (ROBUST BATCH)

- Pairs files across three folders by stripping role-specific suffixes/tokens
  LR rainy  : *_r (optional) — your files may have no suffix
  HR clean  : *_c (optional) — your files may have no suffix
  HR mask   : any of *_m, *_mask, *_seg, *_sliding, *_text, *_bin
- Downsample/upsample: bilinear everywhere
- Readability:          M ← M * (1 - alpha * T)
- Realism (nudging):    M ← M + w(x) * Up( M_gen_lr - Down(M) )
- Per-image logs:       delta_norm, lr_residual, text_energy, delta_rel, lr_improve → convergence.csv
- Memory safe:          processes one triplet at a time
- Batch log (CSV) with summary stats + diagnostics + converged flag
"""

import os
import glob
import csv
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import time
# =============================
# User config (EDIT THESE)
# =============================
LR_DIR   = "/DATA/rohit/anandita/2nd/rdtx_rainy_test"
HRC_DIR  = "/DATA/rohit/anandita/2nd/extracted_frames"
MASK_DIR = "/DATA/rohit/anandita/2nd/segmentation_masks_test/new_test_output_images"
OUT_ROOT = "/DATA/rohit/anandita/2nd/gen2_without_0.5_test_1000_new1"      

# Algorithm params (safe OCR-first defaults; tune as needed)
NUM_ITERS            = 30 # try 28–32 if using smaller realism weight
ALPHA_READABILITY    = 0.5#er = more text protection
REALISM_WEIGHT       =  0.028   #ze for realism nudge (0.02..0.06 typical)
CLAMP_MASK           = 1.0#ip range for M_hr (try 0.85–0.9 to tame overshoot)
PROTECT_TEXT_IN_REALISM = True     # reduce realism strength on text
EPS_STOP_ABS         = 1e-6        # absolute delta stop
EPS_STOP_REL         = 1e-4      # relative delta stop
IMPROVE_MIN          = 1e-3    # <0.1% LR residual improvement considered "plateau"
PATIENCE             = 3  # consecutive iters that must satisfy the stop criteria

# Optional realism step decay (stabilizes late iterations)
USE_STEP_DECAY       = True        # set False to disable
REALISM_DECAY_GAMMA  = 0.90#per-iter decay factor, e.g., 0.90 or 0.95

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pairing tokens (trailing). Edit if your naming changes.
LR_STRIPS   = ["_r"]
HRC_STRIPS  = ["_c"]
MASK_STRIPS = ["_m", "_mask", "_seg", "_sliding", "_text", "_bin"]

# =============================
# I/O helpers
# =============================

def load_rgb(path: str) -> torch.Tensor:
    """Load image as float tensor in [0,1], shape [1,3,H,W] (RGB)."""
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t


def load_mask(path: str, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Load segmentation mask (RGB or grayscale) → [1,1,H,W] in [0,1], resized to target_hw."""
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('RGB')
        arr = np.array(img).astype(np.float32)
        arr = arr.max(axis=2, keepdims=True) / 255.0  # strongest channel → [H,W,1]
    else:
        arr = np.array(img).astype(np.float32)[..., None] / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    H, W = target_hw
    t = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False).clamp(0, 1)
    return t


def save_rgb_tensor(t: torch.Tensor, path: str):
    t = t.clamp(0.0, 1.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(t, path)

# =============================
# Core bilinear resize ops
# =============================

def downsample_bilinear(img: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = size_hw
    return F.interpolate(img, size=(h, w), mode='bilinear', align_corners=False)


def upsample_bilinear(img: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    H, W = size_hw
    return F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)

# =============================
# Alternating Projections (with improved convergence)
# =============================

@torch.no_grad()
def refine_rain_mask_fullimage(
    I_lr_rainy: torch.Tensor,    # [1,3,h,w]
    I_hr_clean: torch.Tensor,    # [1,3,H,W]
    text_mask_hr: torch.Tensor,  # [1,1,H,W] in [0,1] (1=text)
    num_iters: int = NUM_ITERS,
    alpha_readability: float = ALPHA_READABILITY,
    realism_weight: float = REALISM_WEIGHT,
    clamp_mask: float = CLAMP_MASK,
    protect_text_in_realism: bool = PROTECT_TEXT_IN_REALISM,
    eps_stop_abs: float = EPS_STOP_ABS,
    eps_stop_rel: float = EPS_STOP_REL,
    improve_min: float = IMPROVE_MIN,
    patience: int = PATIENCE,
    use_step_decay: bool = USE_STEP_DECAY,
    realism_decay_gamma: float = REALISM_DECAY_GAMMA
):
    """
    Returns:
      M_gen_lr [1,3,h,w], M_hr [1,3,H,W], I_hr_synth [1,3,H,W],
      logs: List[Dict] with
         {'iter','delta_norm','lr_residual','text_energy','delta_rel','lr_improve'},
      iters_used: int,
      converged: bool  (True if early-stopped by plateau criteria)
    """
    H, W = I_hr_clean.shape[-2:]
    h, w = I_lr_rainy.shape[-2:]

    # (1) Clean downsampled + LR-generated mask
    I_lc = downsample_bilinear(I_hr_clean, (h, w))
    M_gen_lr = I_lr_rainy - I_lc

    # (2) Initialize HR mask by bilinear upsample
    M_hr = upsample_bilinear(M_gen_lr, (H, W))

    # (3) Readability map
    T3 = text_mask_hr.repeat(1, 3, 1, 1).clamp(0, 1)
    # T3: [1,3,H,W], values 0/1 or 0..1
    #T3 = text_mask_hr.repeat(1, 3, 1, 1).clamp(0, 1)

# 1) Dilate text mask so protection covers full text area
    #T3 = F.max_pool2d(T3, kernel_size=9, stride=1, padding=4)

# 2) Blur/smooth edges so protection is not dotted
    #T3 = F.avg_pool2d(T3, kernel_size=7, stride=1, padding=3)

    #T3 = T3.clamp(0, 1)
    S_read = (1.0 - alpha_readability * T3).clamp(0, 1)
    S_read = (1.0 - ALPHA_READABILITY * T3).clamp(0, 1)
    
    #S_read_iter = torch.pow(S_read, 1.0 / NUM_ITERS)

    # (4) Realism weight map (weaker corrections on text if enabled)
    if protect_text_in_realism:
    
        #realism_weight_map_base = realism_weight * (1.0 - T3)
        realism_weight_map_base = realism_weight * (0.5+.5*(1.0 - T3))  # 0.5 on text .. 1.0 on bg
    else:
        realism_weight_map_base = torch.full_like(M_hr, fill_value=realism_weight)

    logs: List[Dict[str, float]] = []

    # Convergence helpers
    best_lr = float("inf")
    pat = 0
    converged_flag = False

    for it in range(max(1, int(num_iters))):
        M_prev = M_hr

        # (a) Readability (protect text)
        M_hr = M_hr * S_read
        #M_hr = M_hr * S_read_iter

        # (b) Realism (residual nudge on LR, lifted to HR)
        M_hr_down = downsample_bilinear(M_hr, (h, w))
        resid_lr  = M_gen_lr - M_hr_down
        corr_hr   = upsample_bilinear(resid_lr, (H, W))

        # (b.1) per-iter step decay (optional)
        if use_step_decay:
            decay = realism_decay_gamma ** it
            realism_weight_map = realism_weight_map_base * decay
        else:
            realism_weight_map = realism_weight_map_base

        M_hr = M_hr + realism_weight_map * corr_hr

        # (c) Clamp
        if clamp_mask is not None and clamp_mask > 0:
            M_hr = M_hr.clamp(-clamp_mask, clamp_mask)

        # Metrics
        delta = torch.norm((M_hr - M_prev).reshape(1, -1), p=2).item()
        lr_res = torch.norm(resid_lr.reshape(1, -1), p=2).item()
        text_energy = torch.norm((M_hr * T3).reshape(1, -1), p=2).item()

        den = torch.norm(M_prev.reshape(1, -1), p=2).item() + 1e-12
        delta_rel = delta / den
        improve = 0.0 if best_lr == float('inf') else (best_lr - lr_res) / (best_lr + 1e-12)
        if lr_res < best_lr:
            best_lr = lr_res

        logs.append({
            "iter": it + 1,
            "delta_norm": float(delta),
            "lr_residual": float(lr_res),
            "text_energy": float(text_energy),
            "delta_rel": float(delta_rel),
            "lr_improve": float(improve),
        })

        # Early-stop with patience: small Δ (abs or rel) and plateaued LR residual
        abs_ok = (delta < eps_stop_abs)
        rel_ok = (delta_rel < eps_stop_rel)
        imp_ok = (improve < improve_min)

        if (abs_ok or rel_ok) and imp_ok:
            pat += 1
        else:
            pat = 0

        if pat >= patience:
            converged_flag = True
            break

    I_hr_synth = (I_hr_clean + M_hr).clamp(0.0, 1.0)
    return M_gen_lr.clamp(-1.0, 1.0), M_hr, I_hr_synth, logs, len(logs), converged_flag

# =============================
# Filename pairing (robust)
# =============================

IMG_GLOBS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]

def make_key(path: str, strip_tokens: List[str]) -> str:
    """Return basename (no ext) with one trailing token removed if it matches strip_tokens."""
    base = os.path.splitext(os.path.basename(path))[0]
    lb = base.lower()
    for tok in strip_tokens:
        tok_l = tok.lower()
        if lb.endswith(tok_l):
            base = base[: -len(tok)]
            break
    return base


def index_dir_by_key_role(d: str, strip_tokens: List[str]) -> Dict[str, str]:
    paths = []
    for pat in IMG_GLOBS:
        paths.extend(glob.glob(os.path.join(d, pat)))
    out = {}
    for p in paths:
        k = make_key(p, strip_tokens)
        out[k] = p
    return out


def summarize_missing(label: str, have_keys, want_keys):
    miss = sorted(set(have_keys) - set(want_keys))
    if miss:
        print(f"Missing {label} for {len(miss)} items (showing up to 5): {miss[:5]}")

# =============================
# Batch runner: pairs *_c / *_r / *_m by robust key
# =============================


def run_batch(
    lr_dir: str,
    hrc_dir: str,
    mask_dir: str,
    out_root: str = OUT_ROOT,
    num_iters: int = NUM_ITERS,
    alpha_readability: float = ALPHA_READABILITY,
    realism_weight: float = REALISM_WEIGHT,
    clamp_mask: float = CLAMP_MASK,
    protect_text_in_realism: bool = PROTECT_TEXT_IN_REALISM,
    eps_stop_abs: float = EPS_STOP_ABS,
    eps_stop_rel: float = EPS_STOP_REL,
    improve_min: float = IMPROVE_MIN,
    patience: int = PATIENCE,
    use_step_decay: bool = USE_STEP_DECAY,
    realism_decay_gamma: float = REALISM_DECAY_GAMMA,
    device: str = DEVICE,
):
    os.makedirs(out_root, exist_ok=True)
    torch.set_grad_enabled(False)

    # Index all three folders by pairing key
    lr_paths   = index_dir_by_key_role(lr_dir,   LR_STRIPS)
    hrc_paths  = index_dir_by_key_role(hrc_dir,  HRC_STRIPS)
    mask_paths = index_dir_by_key_role(mask_dir, MASK_STRIPS)

    common = sorted(set(lr_paths) & set(hrc_paths) & set(mask_paths))
    print(f"Found {len(common)} matched triplets.")

    # Diagnostics to help catch naming issues
    summarize_missing("clean", lr_paths.keys(), hrc_paths.keys())
    summarize_missing("rainy", hrc_paths.keys(), lr_paths.keys())
    summarize_missing("mask",  lr_paths.keys(), mask_paths.keys())

    # Show a few examples
    for k in common[:5]:
        print("EXAMPLE KEY:", k)
        print("  HR clean:", hrc_paths[k])
        print("  LR rainy:", lr_paths[k])
        print("  HR mask :", mask_paths[k])

    batch_csv = os.path.join(out_root, "batch_log.csv")
    with open(batch_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "key", "h", "w", "H", "W",
            "num_iters", "alpha_readability", "realism_weight",
            "clamp_mask", "protect_text_in_realism",
            "eps_stop_abs", "eps_stop_rel", "improve_min", "patience",
            "use_step_decay", "realism_decay_gamma",
            "final_delta", "final_lr_residual", "iters_used",
            "converged", "out_dir"
        ])

        for key in common:
            out_dir = os.path.join(out_root, key)
            os.makedirs(out_dir, exist_ok=True)
            try:
                # Load one triplet
                I_lr  = load_rgb(lr_paths[key]).to(device)
                I_hrc = load_rgb(hrc_paths[key]).to(device)
                T_hr  = load_mask(mask_paths[key], target_hw=I_hrc.shape[-2:]).to(device)

                h, w = I_lr.shape[-2:]
                H, W = I_hrc.shape[-2:]

                # Run AP refinement (improved)
                M_lr, M_hr, I_hrs, logs, iters_used, converged = refine_rain_mask_fullimage(
                    I_lr_rainy=I_lr,
                    I_hr_clean=I_hrc,
                    text_mask_hr=T_hr,
                    num_iters=num_iters,
                    alpha_readability=alpha_readability,
                    realism_weight=realism_weight,
                    clamp_mask=clamp_mask,
                    protect_text_in_realism=protect_text_in_realism,
                    eps_stop_abs=eps_stop_abs,
                    eps_stop_rel=eps_stop_rel,
                    improve_min=improve_min,
                    patience=patience,
                    use_step_decay=use_step_decay,
                    realism_decay_gamma=realism_decay_gamma,
                )

                # Save outputs for this image
                save_rgb_tensor((M_lr + 1.0) / 2.0, os.path.join(out_dir, "01_lr_mask_viz.png"))
                save_rgb_tensor((M_hr + 1.0) / 2.0, os.path.join(out_dir, "02_hr_mask_refined_viz.png"))
                save_rgb_tensor(I_hrs,                     os.path.join(out_dir, "03_hr_rainy_synth.png"))

                torch.save(M_lr.cpu(),  os.path.join(out_dir, "01_lr_mask.pt"))
                torch.save(M_hr.cpu(),  os.path.join(out_dir, "2_hr_mask_refined.pt"))   # (kept name short)
                torch.save(I_hrs.cpu(), os.path.join(out_dir, "03_hr_rainy.pt"))

                # Save convergence log with extra metrics
                with open(os.path.join(out_dir, "convergence.csv"), "w", newline="") as fcsv:
                    fields = ["iter", "delta_norm", "lr_residual", "text_energy", "delta_rel", "lr_improve"]
                    writer_c = csv.DictWriter(fcsv, fieldnames=fields)
                    writer_c.writeheader()
                    for row in logs:
                        writer_c.writerow(row)

                final_delta = logs[-1]["delta_norm"] if logs else float("nan")
                final_lr_residual = logs[-1]["lr_residual"] if logs else float("nan")

                writer.writerow([
                    key, h, w, H, W,
                    num_iters, alpha_readability, realism_weight,
                    clamp_mask, int(protect_text_in_realism),
                    eps_stop_abs, eps_stop_rel, improve_min, patience,
                    int(use_step_decay), realism_decay_gamma,
                    f"{final_delta:.3e}", f"{final_lr_residual:.3e}", iters_used,
                    int(converged), out_dir
                ])
                print(f"[{key}] OK — iters_used={iters_used}, final Δ={final_delta:.3e}, "
                      f"LRres={final_lr_residual:.3e}, converged={int(converged)}")

            except Exception as e:
                writer.writerow([
                    key, "", "", "", "",
                    num_iters, alpha_readability, realism_weight, clamp_mask,
                    int(protect_text_in_realism),
                    eps_stop_abs, eps_stop_rel, improve_min, patience,
                    int(use_step_decay), realism_decay_gamma,
                    f"ERROR: {e}", "", "", 0, out_dir
                ])
                print(f"[{key}] FAILED: {e}")

            # Free memory per triplet
            del I_lr, I_hrc, T_hr, M_lr, M_hr, I_hrs
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"Batch complete. Log → {batch_csv}")

# =============================
# Entrypoint
# =============================
if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    run_batch(
        lr_dir=LR_DIR,
        hrc_dir=HRC_DIR,
        mask_dir=MASK_DIR,
        out_root=OUT_ROOT,
        num_iters=NUM_ITERS,
        alpha_readability=ALPHA_READABILITY,
        realism_weight=REALISM_WEIGHT,
        clamp_mask=CLAMP_MASK,
        protect_text_in_realism=PROTECT_TEXT_IN_REALISM,
        eps_stop_abs=EPS_STOP_ABS,
        eps_stop_rel=EPS_STOP_REL,
        improve_min=IMPROVE_MIN,
        patience=PATIENCE,
        use_step_decay=USE_STEP_DECAY,
        realism_decay_gamma=REALISM_DECAY_GAMMA,
        device=DEVICE,
    )

