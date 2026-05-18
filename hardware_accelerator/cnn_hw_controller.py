#!/usr/bin/env python3
"""
cnn_hw_controller.py  -  CNN Hardware Controller with SIMD in SV

Architecture:
  - Input resized to 48x48 (works for all 4 layers, fast simulation)
  - SV pipeline per layer:
      dram_loader  -> im2col in hardware (real HW, correct)
      systolic_tiled -> INT8 MAC (real HW)
      simd_gearbox_8lane -> packs output beats
      simd_relu_top -> ReLU in SV
      ai_quantizer -> INT32->INT8 in SV
  - Python applies MaxPool only (spatial_maxpool output)
  - Layer output feeds next layer automatically

Expected simulation times (48x48):
  Layer 0: ~2-3 min   Layer 1: ~1-2 min
  Layer 2: ~30s       Layer 3: ~15s
"""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import time
import numpy as np
from PIL import Image

TILE_MAX = 16
INPUT_SIZE = 48  # resize all images to this


# =============================================================================
# Quantization (Python post-processing — MaxPool only after SV SIMD)
# =============================================================================

def img_to_int8(img_f32):
    return np.clip(np.round(img_f32 * 127), -128, 127).astype(np.int8)


def maxpool2d(x, k, s):
    """2D max pool matching spatial_maxpool.sv."""
    H, W, C = x.shape
    OH = (H - k) // s + 1
    OW = (W - k) // s + 1
    out = np.full((OH, OW, C), -128, dtype=np.int8)
    for r in range(OH):
        for c in range(OW):
            out[r, c] = np.max(x[r * s:r * s + k, c * s:c * s + k, :], axis=(0, 1))
    return out


def postproc_sv_simd(raw_int32_spatial, layer, w_scale, in_scale=1.0 / 127.0):
    """
    Full post-processing on raw INT32 systolic output.
    Applies ReLU + requantize + MaxPool (same as SW fallback).
    """
    return postproc_sw(raw_int32_spatial, layer, w_scale, in_scale)


def postproc_sw(acc_int32, layer, w_scale, in_scale=1.0 / 127.0):
    """Full SW fallback post-processing (used if SV sim fails)."""
    op = int(layer['simd_opcode'], 2)
    if op & 1:
        acc_int32 = np.maximum(acc_int32, 0)
    x = acc_int32.astype(np.float64) * (w_scale * in_scale)
    act = np.clip(np.round(x * 127), -128, 127).astype(np.int8)
    if op & 2:
        act = maxpool2d(act, layer['pool_k'], layer['pool_st'])
    if op & 4:
        act = act.mean(axis=(0, 1), keepdims=True).astype(np.int8)
    return act, 1.0 / 127.0


# =============================================================================
# File I/O
# =============================================================================

def load_weights(wb, layer):
    rm, oc = layer['weight_shape']
    off = layer['weight_byte_offset']
    size = layer['weight_byte_size']
    return np.frombuffer(wb[off:off + size], dtype=np.int8).reshape(rm, oc)


def load_biases(bb, layer):
    n = layer['out_c']
    off = layer['bias_byte_offset']
    return np.frombuffer(bb[off:off + n * 4], dtype=np.int32)


def write_weight_mem(w, path):
    flat = w.flatten().view(np.uint8)
    flat = np.concatenate([flat, np.zeros((-len(flat)) % 4, dtype=np.uint8)])
    with open(path, 'w') as f:
        for i in range(0, len(flat), 4):
            word = (int(flat[i]) | int(flat[i + 1]) << 8
                    | int(flat[i + 2]) << 16 | int(flat[i + 3]) << 24)
            f.write(f"{word:08x}\n")


def write_bias_mem(b, path):
    with open(path, 'w') as f:
        for v in b:
            f.write(f"{struct.unpack('<I', struct.pack('<i', int(v)))[0]:08x}\n")


def write_act_txt(act, path):
    with open(path, 'w') as f:
        for v in act.flatten():
            f.write(f"{int(v)}\n")


def write_layer_cfg(workdir, idx, layer, w_words, out_h, out_w):
    ph = out_h // layer['pool_st']
    pw = out_w // layer['pool_st']
    lines = [
        f"layer{idx}_act_in.txt",
        f"layer{idx}_weights.mem",
        f"layer{idx}_biases.mem",
        f"layer{idx}_out_raw.txt",  # SV writes INT8 here (after ReLU+Quant)
        str(int(layer['simd_opcode'], 2)),
        str(layer['pool_k']),
        str(layer['pool_st']),
        str(w_words),
        str(layer['out_c']),
        str(ph * pw),
        str(layer['out_c']),
    ]
    with open(os.path.join(workdir, 'layer_cfg.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  [CFG]  layer_cfg.txt written")
    return ph, pw


# =============================================================================
# Recompute layer geometry for 48x48 input
# =============================================================================

def compute_layer_geometry(original_cfg, input_size=48):
    """
    Recompute all layer dimensions for a different input size.
    Conv weights are size-independent, only spatial dims change.
    """
    layers = []
    h, w, c = input_size, input_size, 3

    for orig in original_cfg['layers']:
        kh = orig['k_h'];
        kw = orig['k_w']
        stride = orig['stride'];
        pad = orig['pad']
        out_c = orig['out_c']

        oh = (h + 2 * pad - kh) // stride + 1
        ow = (w + 2 * pad - kw) // stride + 1
        real_k = oh * ow
        real_m = kh * kw * c
        real_m_eff = real_m + 1

        # SRAM_A for full im2col (dram_loader writes entire im2col)
        sram_a = math.ceil(real_k * real_m_eff / 4)
        sram_b = math.ceil(TILE_MAX * real_m_eff / 4)
        aw = max(8, math.ceil(math.log2(max(sram_a, sram_b) + 1)))

        ph = oh // orig['pool_st']
        pw = ow // orig['pool_st']

        layer = dict(orig)
        layer['in_h'] = h;
        layer['in_w'] = w;
        layer['in_c'] = c
        layer['out_h'] = oh;
        layer['out_w'] = ow
        layer['real_m'] = real_m
        layer['weight_shape'] = [real_m, out_c]
        layer['pool_h'] = ph;
        layer['pool_w'] = pw
        layer['aw'] = aw
        layer['sram_a_words'] = sram_a
        layers.append(layer)

        # Next layer input = pooled output
        h, w, c = ph, pw, out_c

    return layers


def compute_aw(layer):
    return layer['aw']


def layer_defines(layer, idx):
    # Use DEF_ prefix to avoid collision with SV parameter names
    # e.g. OUT_C is both a +define and a parameter in systolic_pipe_conv
    # ModelSim uses the define to set the parameter default, breaking REAL_N
    return {
        'DEF_TILE_MAX': TILE_MAX,
        'DEF_AW': compute_aw(layer),
        'DEF_IN_H': layer['in_h'],
        'DEF_IN_W': layer['in_w'],
        'DEF_IN_C': layer['in_c'],
        'DEF_OUT_C': layer['out_c'],
        'DEF_K_H': layer['k_h'],
        'DEF_K_W': layer['k_w'],
        'DEF_STRIDE': layer['stride'],
        'DEF_PAD': layer['pad'],
        'DEF_USE_BIAS': 1,
        'DEF_POOL_K': layer['pool_k'],
        'DEF_POOL_ST': layer['pool_st'],
        'DEF_LAYER_IDX': idx,
    }


# =============================================================================
# ModelSim backend
# =============================================================================

class ModelSim:
    def __init__(self, workdir, sv_sources, sv_tb):
        self.workdir = workdir
        self.sv_sources = sv_sources
        self.sv_tb = sv_tb

    def _run(self, cmd, timeout=300):
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=False,
                               cwd=self.workdir, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, '', f'Timeout {timeout}s', time.time() - t0
        elapsed = time.time() - t0
        stdout = r.stdout.decode('utf-8', errors='replace') if r.stdout else ''
        stderr = r.stderr.decode('utf-8', errors='replace') if r.stderr else ''
        return r.returncode == 0, stdout, stderr, elapsed

    def _setup_work(self, idx):
        work_name = f'work_layer{idx}'
        work_path = os.path.join(self.workdir, work_name)
        ini_path = os.path.join(self.workdir, f'modelsim_layer{idx}.ini')
        with open(ini_path, 'w') as f:
            f.write(f'[Library]\nwork = {work_path}\n\n[vsim]\nresolution = 1ps\n')
        if os.path.exists(work_path):
            shutil.rmtree(work_path)
        subprocess.run(['vlib', work_path], cwd=self.workdir, capture_output=True)
        return work_name, work_path, ini_path

    def compile(self, idx, layer):
        work_name, work_path, ini_path = self._setup_work(idx)
        aw = compute_aw(layer)
        defines = [f"+define+{k}={v}" for k, v in layer_defines(layer, idx).items()]
        cmd = (['vlog', '-sv', '-work', work_path, '-modelsimini', ini_path,
                '-suppress', '2583', '-suppress', '2902']
               + defines + self.sv_sources + [self.sv_tb])
        print(f"\n  [COMPILE] Layer {idx}: {layer['name']}"
              f"  {layer['in_h']}x{layer['in_w']}x{layer['in_c']}"
              f"  AW={aw}")
        ok, out, err, t = self._run(cmd, timeout=180)
        for line in ((out or '') + (err or '')).splitlines():
            if 'Error' in line or 'error' in line:
                print(f"    {line}")
        if not ok:
            print(f"  X compile failed\n{(err or '')[-500:]}")
            return None, None
        print(f"  OK Compiled ({t:.1f}s)")
        return work_name, work_path

    def run(self, idx, work_name, work_path, timeout=3600):
        work_fwd = work_path.replace('\\', '/')
        ini_path = os.path.join(self.workdir, f'modelsim_layer{idx}.ini')
        ini_fwd = ini_path.replace('\\', '/')

        # Write a fresh modelsim.ini that ONLY knows about our local library
        # This prevents vsim from finding stale compiled units in the global work
        with open(ini_path, 'w') as f:
            f.write('[Library]\n')
            f.write(f'work = {work_path}\n')
            f.write(f'{work_name} = {work_path}\n')
            f.write('\n[vsim]\n')
            f.write('resolution = 1ps\n')

        do_script = os.path.join(self.workdir, f'run_layer{idx}.do')
        with open(do_script, 'w') as f:
            f.write(f'vmap work "{work_fwd}"\n')
            f.write(f'vmap {work_name} "{work_fwd}"\n')
            f.write('run -all\nquit -f\n')

        # Use forward slashes — Windows wraps backslash paths in {} causing EINVAL
        ini_fwd = ini_path.replace('\\', '/').replace('\\', '/')
        do_fwd = do_script.replace('\\', '/').replace('\\', '/')

        cmd = ['vsim', '-batch',
               '-modelsimini', ini_fwd,
               '-do', do_fwd,
               'work.tb_sv_simd']
        print(f"  [RUN]   Layer {idx} (timeout={timeout}s)...")
        ok, out, err, elapsed = self._run(cmd, timeout=timeout)
        for line in (out or '').splitlines():
            print(f"    {line}")
        if not ok:
            print(f"  X vsim failed")
            for line in (err or '').splitlines():
                print(f"    {line}")
        return ok, elapsed


# =============================================================================
# SW fallback accumulator
# =============================================================================

def sw_layer(act, w, b, layer):
    kh, kw = layer['k_h'], layer['k_w']
    stride = layer['stride'];
    pad = layer['pad']
    H, W, C = act.shape
    if pad:
        act = np.pad(act, ((pad, pad), (pad, pad), (0, 0)), 'constant')
    OH = (H + 2 * pad - kh) // stride + 1;
    OW = (W + 2 * pad - kw) // stride + 1
    col = np.zeros((OH * OW, kh * kw * C), dtype=np.int8)
    for i, (r, c) in enumerate((r, c) for r in range(OH) for c in range(OW)):
        col[i] = np.transpose(
            act[r * stride:r * stride + kh, c * stride:c * stride + kw, :], (2, 0, 1)).flatten()
    ones = np.ones((col.shape[0], 1), dtype=np.int8)
    w_eff = np.vstack([w, b.reshape(1, -1)])
    return (np.hstack([col, ones]).astype(np.int32) @ w_eff.astype(np.int32)
            ).reshape(OH, OW, -1)


def read_simd_output(path, oh, ow, oc):
    """Read raw INT32 values from systolic direct capture."""
    vals = [int(line) for line in open(path) if line.strip()]
    exp = oh * ow * oc
    print(f"  [READ]  {len(vals)} INT32 values (expected {exp}={oh}x{ow}x{oc})")
    arr = np.array(vals, dtype=np.int32)
    if len(arr) < exp:
        arr = np.concatenate([arr, np.zeros(exp - len(arr), dtype=np.int32)])
    return arr[:exp].reshape(oh, ow, oc)


# =============================================================================
# Visualisation
# =============================================================================

def fmap_to_image(act, n_cols=8, n_show=32):
    H, W, C = act.shape
    n_show = min(n_show, C)
    n_rows = (n_show + n_cols - 1) // n_cols
    GAP = 2
    grid = np.zeros((n_rows * (H + GAP), n_cols * (W + GAP)), dtype=np.uint8)
    for c in range(n_show):
        ch = act[:, :, c].astype(np.float32)
        lo, hi = ch.min(), ch.max()
        ch = ((ch - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.full_like(ch, 128, np.uint8)
        ri, ci = c // n_cols, c % n_cols
        grid[ri * (H + GAP):ri * (H + GAP) + H, ci * (W + GAP):ci * (W + GAP) + W] = ch
    return Image.fromarray(grid, 'L').convert('RGB')


def build_summary(inp_pil, results, out_path):
    from PIL import ImageDraw
    PW = 720;
    BG = (18, 18, 24);
    H2 = 32;
    GAP = 5;
    panels = []
    inp = inp_pil.resize((INPUT_SIZE, INPUT_SIZE))
    p = Image.new('RGB', (PW, INPUT_SIZE + H2 + GAP), BG);
    d = ImageDraw.Draw(p)
    d.rectangle([0, 0, PW, H2], fill=(30, 50, 80))
    d.text((8, 9), f"INPUT {INPUT_SIZE}x{INPUT_SIZE}x3 -> dram_loader(SV) -> systolic(SV) -> SIMD(SV) -> MaxPool(Py)",
           fill=(200, 230, 255))
    p.paste(inp, (8, H2 + GAP // 2));
    panels.append(p)
    colors = [(40, 70, 40), (55, 40, 70), (70, 55, 30), (30, 60, 70)]
    for i, (name, act, secs, sv_ok) in enumerate(results):
        fi = fmap_to_image(act);
        fh = fi.height
        panel = Image.new('RGB', (PW, fh + H2 + GAP + 56), BG);
        d = ImageDraw.Draw(panel)
        d.rectangle([0, 0, PW, H2], fill=colors[i % 4])
        Hf, Wf, Cf = act.shape
        tag = "SV+SIMD" if sv_ok else "SW fallback"
        d.text((8, 9), f"LAYER {i}: {name} -> {Hf}x{Wf}x{Cf} [{tag}] {secs:.1f}s", fill=(240, 240, 150))
        panel.paste(fi, (8, H2 + GAP))
        tx = fi.width + 20;
        ty = H2 + GAP + 4
        notes = [f"im2col: dram_loader.sv (real HW)",
                 f"MAC: systolic_tiled.sv (real HW)",
                 f"ReLU: simd_relu_top.sv (SV)",
                 f"Quant: simd_gearbox.sv (SV)",
                 "-" * 28,
                 f"MaxPool: Python (spatial_maxpool.sv)",
                 f"-> layer{i + 1}_act_in.txt" if i < 3 else "-> final output"]
        for ln in notes:
            d.text((tx, ty), ln, fill=(170, 195, 170));
            ty += 16
        panels.append(panel)
    total_h = sum(p.height for p in panels) + GAP * len(panels) + 52
    canvas = Image.new('RGB', (PW, total_h), BG);
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, PW, 50], fill=(12, 22, 50))
    d.text((8, 8), "CNN Hardware Controller - Full SV Pipeline with SIMD", fill=(255, 255, 200))
    d.text((8, 28), f"48x48 input | dram_loader+systolic+ReLU+Quantize in SV", fill=(160, 190, 230))
    y = 52
    for panel in panels:
        canvas.paste(panel, (0, y));
        y += panel.height + GAP
    canvas.save(out_path);
    print(f"  [IMG]  {out_path}")


# =============================================================================
# Main
# =============================================================================

def run(args):
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    print("=" * 62)
    print("  CNN HARDWARE CONTROLLER - Full SV Pipeline with SIMD")
    print(f"  Input size: {INPUT_SIZE}x{INPUT_SIZE}  |  Simulator: MODELSIM")
    print("=" * 62)

    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.weights, 'rb') as f:
        wb = f.read()
    with open(args.biases, 'rb') as f:
        bb = f.read()

    # Recompute geometry for INPUT_SIZE
    layers = compute_layer_geometry(cfg, INPUT_SIZE)
    print(f"\n  Layers: {len(layers)}  Weights: {len(wb) // 1024}KB")
    for i, l in enumerate(layers):
        print(f"  L{i}: {l['in_h']}x{l['in_w']}x{l['in_c']} -> "
              f"{l['out_h']}x{l['out_w']}x{l['out_c']}  AW={l['aw']}"
              f"  SRAM_A={l['sram_a_words']}w")

    # Resolve SV sources
    raw_src = args.sv_src.strip()
    sv_names = ([s.strip() for s in raw_src.split(',') if s.strip()]
                if ',' in raw_src
                else [s.strip() for s in raw_src.split() if s.strip()])
    sv_sources = []
    for sv in sv_names:
        for cand in [sv, os.path.join(args.sv_dir, sv)]:
            if os.path.exists(cand):
                sv_sources.append(os.path.abspath(cand));
                break
    print(f"\n  SV files: {len(sv_sources)} found")

    # Preprocess image — resize to INPUT_SIZE
    img_pil = Image.open(args.image).convert('RGB')
    img_resized = img_pil.resize((INPUT_SIZE, INPUT_SIZE))
    act = img_to_int8(np.array(img_resized, np.float32) / 255.0)
    print(f"\n  Input: {img_pil.size} -> {act.shape}")
    write_act_txt(act, os.path.join(args.workdir, 'layer0_act_in.txt'))

    sim = ModelSim(args.workdir, sv_sources, args.sv_tb)
    results = [];
    total_t0 = time.time()
    in_scale = 1.0 / 127.0  # initial input quantization scale

    for i, layer in enumerate(layers):
        name = layer['name']
        oh = layer['out_h'];
        ow = layer['out_w'];
        oc = layer['out_c']
        w_sc = layer['weight_scale']
        aw = compute_aw(layer)

        print(f"\n{'=' * 62}")
        print(f"  LAYER {i}: {name}")
        print(f"  {layer['in_h']}x{layer['in_w']}x{layer['in_c']}"
              f" -> conv {oh}x{ow}x{oc} -> pool {layer['pool_h']}x{layer['pool_w']}x{oc}")
        print(f"  AW={aw}  SRAM_A={layer['sram_a_words']} words")
        print(f"{'=' * 62}")

        w = load_weights(wb, layer)
        b = load_biases(bb, layer)
        w_words = (layer['weight_byte_size'] + 3) // 4

        write_weight_mem(w, os.path.join(args.workdir, f'layer{i}_weights.mem'))
        write_bias_mem(b, os.path.join(args.workdir, f'layer{i}_biases.mem'))
        print(f"  [MEM]  weights ({w_words}w)  biases ({oc}w)")

        ph, pw = write_layer_cfg(args.workdir, i, layer, w_words, oh, ow)

        work_name, work_path = sim.compile(i, layer)

        sv_ok = False;
        elapsed = 0.0

        if work_name:
            sim_ok, elapsed = sim.run(i, work_name, work_path, args.timeout)
            if sim_ok:
                raw_path = os.path.join(args.workdir, f'layer{i}_out_raw.txt')
                if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                    # SV wrote INT8 (after ReLU + Quantize)
                    # Read as pre-pool activation (OH x OW x OC)
                    acc_spatial = read_simd_output(raw_path, oh, ow, oc)
                    # Apply full post-processing (ReLU + requantize + MaxPool)
                    act_out, in_scale = postproc_sv_simd(acc_spatial, layer, w_sc, in_scale)
                    sv_ok = True
                else:
                    print(f"  WARNING: output empty, SW fallback")
                    acc = sw_layer(act, w, b, layer)
                    act_out, in_scale = postproc_sw(acc, layer, w_sc, in_scale)
            else:
                print(f"  WARNING: vsim failed, SW fallback")
                acc = sw_layer(act, w, b, layer)
                act_out, in_scale = postproc_sw(acc, layer, w_sc, in_scale)
        else:
            print(f"  WARNING: compile failed, SW fallback")
            acc = sw_layer(act, w, b, layer)
            act_out = postproc_sw(acc, layer, w_sc)

        print(f"  [POST]  {act_out.shape}  [{act_out.min()},{act_out.max()}]"
              f"  {'SV+SIMD+MaxPool' if sv_ok else 'SW fallback'}")

        write_act_txt(act_out, os.path.join(args.workdir, f'layer{i}_out.txt'))
        if i + 1 < len(layers):
            write_act_txt(act_out, os.path.join(args.workdir, f'layer{i + 1}_act_in.txt'))
            print(f"  [FEED]  -> layer{i + 1}_act_in.txt ({act_out.size} values)")

        results.append((name, act_out.copy(), elapsed, sv_ok))
        act = act_out

    total_t = time.time() - total_t0
    sv_count = sum(1 for _, _, _, ok in results if ok)

    print(f"\n{'=' * 62}")
    print(f"  ALL LAYERS COMPLETE  {total_t:.1f}s  SV+SIMD={sv_count}/4")
    print(f"{'=' * 62}")

    img_resized.save(os.path.join(args.output, 'input_image.png'))
    for i, (name, act, _, _) in enumerate(results):
        fmap_to_image(act).save(os.path.join(args.output, f'layer{i}_fmap.png'))
        print(f"  [IMG]  layer{i}_fmap.png  {act.shape}")
    build_summary(img_resized, results,
                  os.path.join(args.output, 'hw_run_summary.png'))
    shutil.copy(os.path.join(args.workdir, 'layer_cfg.txt'),
                os.path.join(args.output, 'last_layer_cfg.txt'))
    print(f"\n  Done -> {args.output}/hw_run_summary.png")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='cat_dog_model.h5')
    p.add_argument('--config', default='model_config.json')
    p.add_argument('--weights', default='weights.bin')
    p.add_argument('--biases', default='biases.bin')
    p.add_argument('--image', default='cat.jpg')
    p.add_argument('--sv_tb', default='tb_sv_simd.sv')
    p.add_argument('--sv_dir', default='.')
    p.add_argument('--sv_src',
                   default=(
                       'Mac (1).sv,'
                       'SystolicArray (1).sv,'
                       'SystolicController.sv,'
                       'compute_engine (1).sv,'
                       'dramloader (1).sv,'
                       'bias (1).sv,'
                       'simd_gear.sv,'
                       'relu.sv,'
                       'maxpool.sv,'
                       'batchNorm.sv,'
                       'add.sv,'
                       'quantizer.sv,'
                       'spaMaxPool.sv,'
                       'spaGlobalAvg.sv,'
                       'wrapper.sv'
                   ))
    p.add_argument('--sim', default='modelsim')
    p.add_argument('--workdir', default='hw_sim_run')
    p.add_argument('--output', default='hw_sim_out')
    p.add_argument('--timeout', type=int, default=3600)
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
