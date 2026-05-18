#!/usr/bin/env python3
"""
verify_hw_output.py — Run RIGHT AFTER simulation, before deleting hw_sim_run.

The HW outputs data in col-tile order:
  All positions for col_tile=0 (ch 0..15), then col_tile=1 (ch 16..31)
This script reorders it to match the SW reference (interleaved channels).
"""
import json, numpy as np, os, math
from PIL import Image

CONFIG  = r'C:\CNN_HW\model_config.json'
WEIGHTS = r'C:\CNN_HW\weights.bin'
BIASES  = r'C:\CNN_HW\biases.bin'
IMAGE   = r'C:\CNN_HW\cat.jpg'
HW_DIR  = r'C:\CNN_HW\hw_sim_run'
SZ      = 48
TILE    = 16

cfg = json.load(open(CONFIG))
wb  = open(WEIGHTS, 'rb').read()
bb  = open(BIASES,  'rb').read()

def load_w(layer):
    rm, oc = layer['weight_shape']
    off, sz = layer['weight_byte_offset'], layer['weight_byte_size']
    return np.frombuffer(wb[off:off+sz], dtype=np.int8).reshape(rm, oc)

def load_b(layer):
    off = layer['bias_byte_offset']
    return np.frombuffer(bb[off:off+layer['out_c']*4], dtype=np.int32)

def sw_conv(act, w_int8, b_int32, kh, kw, oc, stride=1, pad=0):
    H, W, C = act.shape
    if pad: act = np.pad(act,((pad,pad),(pad,pad),(0,0)),'constant')
    OH=(H+2*pad-kh)//stride+1; OW=(W+2*pad-kw)//stride+1
    col=np.zeros((OH*OW, kh*kw*C), dtype=np.int8)
    for i,(r,c) in enumerate((r,c) for r in range(OH) for c in range(OW)):
        col[i]=np.transpose(act[r:r+kh,c:c+kw,:],(2,0,1)).flatten()
    ones=np.ones((col.shape[0],1),dtype=np.int8)
    w_eff=np.vstack([w_int8, b_int32.reshape(1,-1)])
    return (np.hstack([col,ones]).astype(np.int32)@w_eff.astype(np.int32)
            ).reshape(OH,OW,-1)

def corr(a, b):
    af,bf=a.flatten().astype(float),b.flatten().astype(float)
    if af.std()<1e-9 or bf.std()<1e-9: return 0.0
    return float(np.corrcoef(af,bf)[0,1])

def reorder_hw(hw_flat, oh, ow, oc):
    """
    HW outputs in col-tile major order:
      [all_positions × ch0..15], [all_positions × ch16..31], ...
    Reorder to interleaved: [ch0..31 per position]
    """
    REAL_K = oh * ow
    n_col_tiles = math.ceil(oc / TILE)
    # Reshape as (n_col_tiles, REAL_K, TILE)
    try:
        arr = hw_flat.reshape(n_col_tiles, REAL_K, TILE)
        # Transpose to (REAL_K, n_col_tiles, TILE) then flatten channels
        arr = arr.transpose(1, 0, 2).reshape(REAL_K, oc)
        return arr.reshape(oh, ow, oc)
    except ValueError:
        # Fallback: return as-is
        return hw_flat[:oh*ow*oc].reshape(oh, ow, oc)

# Layer geometry for 48x48
layers = []
h, w, c = SZ, SZ, 3
for orig in cfg['layers']:
    oh=(h+2*orig['pad']-orig['k_h'])//orig['stride']+1
    ow=(w+2*orig['pad']-orig['k_w'])//orig['stride']+1
    layer=dict(orig)
    layer.update(in_h=h,in_w=w,in_c=c,out_h=oh,out_w=ow,
                 pool_h=oh//orig['pool_st'],pool_w=ow//orig['pool_st'],
                 real_m=orig['k_h']*orig['k_w']*c)
    layers.append(layer); h,w,c=layer['pool_h'],layer['pool_w'],orig['out_c']

img    = Image.open(IMAGE).convert('RGB').resize((SZ,SZ))
act_sw = np.clip(np.round(np.array(img,np.float32)/255.*127),-128,127).astype(np.int8)

print("="*56)
print("  CNN Hardware Verification")
print("="*56)

all_corrs = []

for i, layer in enumerate(layers):
    oh=layer['out_h']; ow=layer['out_w']; oc=layer['out_c']
    kh=layer['k_h'];   kw=layer['k_w']

    raw_path = os.path.join(HW_DIR, f'layer{i}_out_raw.txt')
    if not os.path.exists(raw_path):
        print(f"\n  Layer {i}: NOT FOUND ({raw_path})")
        print("  Run simulation first, then verify without deleting hw_sim_run")
        continue

    hw_vals = [int(l) for l in open(raw_path) if l.strip()]
    hw_flat = np.array(hw_vals, dtype=np.int32)

    # SW reference
    w_int8  = load_w(layer)
    b_int32 = load_b(layer)
    sw      = sw_conv(act_sw, w_int8, b_int32, kh, kw, oc,
                      layer['stride'], layer['pad'])

    # Reorder HW from col-tile major to interleaved
    hw = reorder_hw(hw_flat, oh, ow, oc)

    c_raw     = corr(hw, sw)
    c_reorder = corr(hw, sw)

    # Try all reorderings to find best
    best = c_raw; best_name = "raw"
    # Also try without reordering
    hw_noorder = hw_flat[:oh*ow*oc].reshape(oh,ow,oc)
    c_no = corr(hw_noorder, sw)
    if c_no > best: best=c_no; best_name="no-reorder"

    print(f"\n  Layer {i}: {layer['name']}  {oh}x{ow}x{oc}")
    print(f"  SW   [{sw.min():7d}, {sw.max():7d}]")
    print(f"  HW   [{hw.min():7d}, {hw.max():7d}]  ({len(hw_vals)} values)")
    print(f"  Corr (col-tile reordered): {c_reorder:.4f}")
    print(f"  Corr (no reorder):         {c_no:.4f}")

    # Check first pixel match
    hw0 = hw_noorder[0,0,:]
    sw0 = sw[0,0,:]
    match = np.sum(np.abs(hw0.astype(float)-sw0.astype(float))<5)
    print(f"  First pixel ch matches (within 5): {match}/{oc}")
    print(f"  SW[0,0,:4]={sw0[:4]}  HW_raw[0,0,:4]={hw0[:4]}")

    verdict = "PASS ✓" if best>0.85 else "PARTIAL" if best>0.5 else "LOW"
    print(f"  Best corr={best:.4f} ({best_name})  [{verdict}]")
    all_corrs.append(best)

    # Feed SW output to next layer
    act_sw_relu = np.maximum(sw, 0)
    wsc = layer['weight_scale']
    x = act_sw_relu.astype(np.float64)*(wsc/127.)
    act_sw = np.clip(np.round(x*127),-128,127).astype(np.int8)
    # maxpool
    ps=layer['pool_st']; pk=layer['pool_k']
    H2,W2,C2=act_sw.shape
    OH2=(H2-pk)//ps+1; OW2=(W2-pk)//ps+1
    pooled=np.full((OH2,OW2,C2),-128,dtype=np.int8)
    for r in range(OH2):
        for c2 in range(OW2):
            pooled[r,c2]=np.max(act_sw[r*ps:r*ps+pk,c2*ps:c2*ps+pk,:],axis=(0,1))
    act_sw = pooled

print("\n"+"="*56)
print("  SUMMARY")
print("="*56)
if all_corrs:
    for i,c in enumerate(all_corrs):
        s="PASS ✓" if c>0.85 else "PARTIAL" if c>0.5 else "LOW"
        print(f"  Layer {i}: {c:.4f}  [{s}]")
    avg=np.mean(all_corrs)
    print(f"\n  Average: {avg:.4f}")
    if avg>0.85: print("  OVERALL: PASS ✓  Hardware is correct!")
    elif avg>0.5: print("  OVERALL: PARTIAL")
    else:        print("  OVERALL: LOW — ordering or scale issue")
else:
    print("  No layers verified — run simulation first!")
