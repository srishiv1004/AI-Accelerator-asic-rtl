"""
convert_model.py
================
Converts a Keras .h5 model into hardware-ready files for the systolic CNN accelerator.

Supported Keras layer types:
  - Conv2D                  -> conv layer entry
  - BatchNormalization      -> folded into preceding Conv2D weights/bias (zero HW cost)
  - ReLU / Activation('relu') -> sets relu bit in SIMD opcode
  - MaxPooling2D            -> sets maxpool bit in SIMD opcode
  - GlobalAveragePooling2D  -> sets gavgpool bit in SIMD opcode
  - Dense, DepthwiseConv2D  -> NOT supported (warning + skip)

SIMD opcode (4-bit, BN and ADD removed - always handled offline):
  bit3=quantize, bit2=gavgpool, bit1=maxpool, bit0=relu

  Hidden conv layer         : 4'b1001
  Hidden conv + maxpool     : 4'b1011
  Last layer with GAP       : 4'b1101

Output files:
  - model_config.json   : layer descriptors, SIMD opcodes, DRAM offsets
  - weights.bin         : all INT8 weights packed back-to-back
  - biases.bin          : all INT32 biases packed back-to-back
  - weights.mem         : hex format for $readmemh in ModelSim
  - biases.mem          : hex format for $readmemh in ModelSim
  - model_summary.txt   : human readable layer info

Quantization scheme (matches quantizer.sv):
  Weights: float32 -> INT8  scale = max(|W|) / 127
  Biases:  float32 -> INT32 (kept wide to match systolic accumulator)

Usage:
  python convert_model.py --model model.h5 --output ./hw_model
"""

import argparse
import json
import math
import struct
import sys
import os
import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py not installed. Run: pip install h5py")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("WARNING: TensorFlow not found. Falling back to h5py direct read.")
    print("         Layer topology will be inferred from weight shapes.")

# =============================================================================
# SIMD opcode bit positions (matches wrapper.sv opcode_t struct packed)
# New 4-bit opcode - BN and ADD removed (always folded offline):
# bit3=quantize, bit2=gavgpool, bit1=maxpool, bit0=relu
#
# Common opcodes:
#   Hidden conv layer          : 4'b1001  (relu + quantize)
#   Hidden conv + maxpool      : 4'b1011  (relu + maxpool + quantize)
#   Last layer with GAP        : 4'b1101  (relu + gavgpool + quantize)
#   Last layer GAP + maxpool   : 4'b1111  (relu + maxpool + gavgpool + quantize)
# =============================================================================
OP_RELU     = (1 << 0)
OP_MAXPOOL  = (1 << 1)
OP_GAVGPOOL = (1 << 2)
OP_QUANTIZE = (1 << 3)

# =============================================================================
# Quantization helpers
# =============================================================================

def quantize_weights_int8(weights_f32):
    """
    Symmetric per-tensor INT8 quantization matching quantizer.sv behaviour.
    scale = max(|W|) / 127
    W_int8 = clip(round(W / scale), -128, 127)
    Returns (w_int8, scale)
    """
    max_val = np.max(np.abs(weights_f32))
    if max_val == 0:
        return np.zeros_like(weights_f32, dtype=np.int8), 1.0
    scale = max_val / 127.0
    w_int8 = np.clip(np.round(weights_f32 / scale), -128, 127).astype(np.int8)
    return w_int8, scale


def quantize_bias_int32(bias_f32, weight_scale, input_scale=1.0):
    """
    Bias quantization for INT32 accumulator.
    bias_scale = weight_scale * input_scale
    b_int32 = round(bias / bias_scale)
    """
    bias_scale = weight_scale * input_scale
    if bias_scale == 0:
        return np.zeros_like(bias_f32, dtype=np.int32)
    return np.clip(np.round(bias_f32 / bias_scale),
                   -(2**31), 2**31 - 1).astype(np.int32)


# =============================================================================
# Weight layout conversion
# =============================================================================

def conv2d_weights_to_hw(keras_weights):
    """
    Keras Conv2D weight shape: (KH, KW, IN_C, OUT_C)
    Hardware expects:          (IN_C, KH, KW, OUT_C)  -> flat row-major

    The systolic array feeds weights column by column (output channel = column).
    For each output position the weight vector is:
      [w[0,0,0,oc], w[0,0,1,oc], ..., w[kh-1,kw-1,ic-1,oc]]  length = KH*KW*IN_C
    This is REAL_M elements per output channel.
    """
    kh, kw, in_c, out_c = keras_weights.shape
    # Transpose from (KH,KW,IN_C,OUT_C) -> (IN_C,KH,KW,OUT_C)
    w = np.transpose(keras_weights, (2, 0, 1, 3))
    # Flatten to (REAL_M, OUT_C) = (IN_C*KH*KW, OUT_C)
    w = w.reshape(in_c * kh * kw, out_c)
    return w  # shape: (REAL_M, OUT_C)


# =============================================================================
# BN folding
# =============================================================================

def fold_batchnorm(conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, bn_eps=1e-3):
    """
    Fold BatchNormalization into Conv2D weights and biases.

    BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
    Collapse to: y = scale * x + shift
      scale = gamma / sqrt(var + eps)   shape: (OUT_C,)
      shift = beta - mean * scale       shape: (OUT_C,)

    Folded conv:
      new_weight[..., oc] = weight[..., oc] * scale[oc]
      new_bias[oc]        = conv_bias[oc] * scale[oc] + shift[oc]
    """
    scale = bn_gamma / np.sqrt(bn_var + bn_eps)  # (OUT_C,)
    shift = bn_beta - bn_mean * scale             # (OUT_C,)

    # keras_weights shape: (KH, KW, IN_C, OUT_C)
    # scale broadcasts over last axis
    new_w = conv_w * scale[np.newaxis, np.newaxis, np.newaxis, :]
    new_b = conv_b * scale + shift

    return new_w, new_b


# =============================================================================
# TensorFlow/Keras model parser
# =============================================================================

class LayerEntry:
    """Represents one hardware layer."""
    def __init__(self):
        self.layer_type   = None   # 'conv2d', 'maxpool', 'gap'
        self.in_h = self.in_w = self.in_c = self.out_c = 0
        self.k_h  = self.k_w  = 1
        self.stride = 1
        self.pad    = 0
        self.pool_k = 2
        self.pool_st = 2
        self.use_bias = True
        self.simd_opcode = 0
        self.weight_byte_offset = 0   # in weights.bin
        self.bias_byte_offset   = 0   # in biases.bin
        self.weight_shape = None
        self.bias_shape   = None
        self.w_int8  = None
        self.b_int32 = None
        self.w_scale = 1.0
        self.name    = ""
        self.is_last = False


def parse_keras_model(model_path, input_scale=1.0):
    """
    Parse a Keras .h5 model and return a list of LayerEntry objects.
    Folds BN into preceding Conv2D automatically.
    """
    if not HAS_TF:
        print("ERROR: TensorFlow required for full model parsing.")
        print("       Install with: pip install tensorflow")
        sys.exit(1)

    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    model.summary()

    layers_out = []
    pending_conv = None   # holds LayerEntry awaiting possible BN fold
    pending_conv_w = None # raw float32 weights
    pending_conv_b = None # raw float32 bias

    current_relu    = False
    current_maxpool = False
    current_gap     = False

    def flush_pending():
        """Finalise pending conv with accumulated SIMD ops."""
        nonlocal pending_conv, pending_conv_w, pending_conv_b
        nonlocal current_relu, current_maxpool, current_gap
        if pending_conv is None:
            return

        opcode = OP_QUANTIZE
        if current_relu:    opcode |= OP_RELU
        if current_maxpool: opcode |= OP_MAXPOOL
        if current_gap:     opcode |= OP_GAVGPOOL
        # Note: BN is always folded offline, ADD not supported - neither appears in opcode

        pending_conv.simd_opcode = opcode

        # Quantize weights
        w_hw = conv2d_weights_to_hw(pending_conv_w)  # (REAL_M, OUT_C)
        w_int8, w_scale = quantize_weights_int8(w_hw.astype(np.float32))
        b_int32 = quantize_bias_int32(pending_conv_b, w_scale, input_scale)

        pending_conv.w_int8  = w_int8
        pending_conv.b_int32 = b_int32
        pending_conv.w_scale = w_scale
        pending_conv.weight_shape = w_int8.shape
        pending_conv.bias_shape   = b_int32.shape

        layers_out.append(pending_conv)

        # Reset
        pending_conv = pending_conv_w = pending_conv_b = None
        current_relu = current_maxpool = current_gap = False

    for lyr in model.layers:
        cfg  = lyr.get_config()
        ltype = lyr.__class__.__name__

        print(f"  [{ltype}] {lyr.name}")

        # ------------------------------------------------------------------
        if ltype == 'Conv2D':
            flush_pending()   # commit previous conv if any

            # Extract weights
            w_f32 = lyr.get_weights()[0]  # (KH,KW,IN_C,OUT_C)
            if lyr.use_bias:
                b_f32 = lyr.get_weights()[1]  # (OUT_C,)
            else:
                b_f32 = np.zeros(w_f32.shape[-1], dtype=np.float32)

            kh, kw, in_c, out_c = w_f32.shape
            strides = cfg['strides']
            padding = cfg['padding']
            pad_val = 0 if padding == 'valid' else kh // 2  # 'same' approx

            # Infer spatial dims from input shape
            in_shape = lyr.input.shape
            if isinstance(in_shape, list):
                in_shape = in_shape[0]
            _, in_h, in_w, _ = in_shape

            entry = LayerEntry()
            entry.layer_type = 'conv2d'
            entry.name   = lyr.name
            entry.in_h   = in_h
            entry.in_w   = in_w
            entry.in_c   = in_c
            entry.out_c  = out_c
            entry.k_h    = kh
            entry.k_w    = kw
            entry.stride = strides[0]
            entry.pad    = pad_val

            pending_conv   = entry
            pending_conv_w = w_f32
            pending_conv_b = b_f32

        # ------------------------------------------------------------------
        elif ltype == 'BatchNormalization':
            if pending_conv is None:
                print(f"    WARNING: BN without preceding Conv2D, skipping")
                continue
            print(f"    -> Folding into {pending_conv.name}")
            bn_gamma = lyr.get_weights()[0]
            bn_beta  = lyr.get_weights()[1]
            bn_mean  = lyr.get_weights()[2]
            bn_var   = lyr.get_weights()[3]
            pending_conv_w, pending_conv_b = fold_batchnorm(
                pending_conv_w, pending_conv_b,
                bn_gamma, bn_beta, bn_mean, bn_var)

        # ------------------------------------------------------------------
        elif ltype in ('ReLU', 'Activation') and (
                ltype == 'ReLU' or cfg.get('activation') == 'relu'):
            current_relu = True

        # ------------------------------------------------------------------
        elif ltype == 'MaxPooling2D':
            current_maxpool = True
            # Store pool params on pending conv for later use
            if pending_conv is not None:
                pending_conv.pool_k  = cfg['pool_size'][0]
                pending_conv.pool_st = cfg['strides'][0]

        # ------------------------------------------------------------------
        elif ltype == 'GlobalAveragePooling2D':
            current_gap = True

        # ------------------------------------------------------------------
        elif ltype in ('Dense', 'DepthwiseConv2D', 'SeparableConv2D'):
            print(f"    WARNING: {ltype} not supported by hardware, skipping")

        # ------------------------------------------------------------------
        elif ltype in ('InputLayer', 'Dropout', 'Flatten', 'ZeroPadding2D',
                       'Reshape', 'Lambda'):
            pass  # purely structural, no hardware action needed

        else:
            print(f"    WARNING: Unknown layer type '{ltype}', skipping")

    # Flush last pending conv
    flush_pending()

    if layers_out:
        layers_out[-1].is_last = True

    return layers_out


# =============================================================================
# Output file writer
# =============================================================================

def write_output(layers, output_dir, input_h, input_w, input_c):
    os.makedirs(output_dir, exist_ok=True)

    weight_offset = 0   # byte offset into weights.bin
    bias_offset   = 0   # byte offset into biases.bin

    config = {
        "input_shape": [input_h, input_w, input_c],
        "num_layers": len(layers),
        "layers": []
    }

    weight_bytes = bytearray()
    bias_bytes   = bytearray()

    for i, lyr in enumerate(layers):
        entry = {
            "index":        i,
            "name":         lyr.name,
            "type":         lyr.layer_type,
            "is_last":      lyr.is_last,
            "simd_opcode":  f"{lyr.simd_opcode:04b}",
            "simd_opcode_sv": f"4'b{lyr.simd_opcode:04b}",
        }

        if lyr.layer_type == 'conv2d':
            real_m = lyr.in_c * lyr.k_h * lyr.k_w
            oh = (lyr.in_h + 2*lyr.pad - lyr.k_h) // lyr.stride + 1
            ow = (lyr.in_w + 2*lyr.pad - lyr.k_w) // lyr.stride + 1

            entry.update({
                "in_h": lyr.in_h, "in_w": lyr.in_w, "in_c": lyr.in_c,
                "out_c": lyr.out_c,
                "k_h":  lyr.k_h,  "k_w":  lyr.k_w,
                "stride": lyr.stride, "pad": lyr.pad,
                "out_h": oh, "out_w": ow,
                "real_m": real_m,
                "pool_k":  lyr.pool_k,
                "pool_st": lyr.pool_st,
                "weight_scale": float(lyr.w_scale),
                "weight_byte_offset": weight_offset,
                "weight_byte_size":   lyr.w_int8.size,   # INT8, 1 byte each
                "bias_byte_offset":   bias_offset,
                "bias_byte_size":     lyr.b_int32.size,  # INT32, 4 bytes each
                "weight_shape": list(lyr.weight_shape),  # (REAL_M, OUT_C)
                "bias_shape":   list(lyr.bias_shape),
            })

            # Pack weights: row-major (REAL_M, OUT_C) INT8
            w_flat = lyr.w_int8.flatten().astype(np.int8)
            weight_bytes.extend(w_flat.tobytes())
            weight_offset += w_flat.size   # 1 byte per weight

            # Pack biases: OUT_C INT32 values, little-endian
            b_flat = lyr.b_int32.flatten().astype(np.int32)
            bias_bytes.extend(b_flat.tobytes())
            bias_offset += b_flat.size * 4  # 4 bytes per bias

        config["layers"].append(entry)

    # Write JSON config
    cfg_path = os.path.join(output_dir, "model_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nWrote: {cfg_path}")

    # Write weights binary
    w_path = os.path.join(output_dir, "weights.bin")
    with open(w_path, "wb") as f:
        f.write(weight_bytes)
    print(f"Wrote: {w_path}  ({len(weight_bytes)} bytes)")

    # Write biases binary
    b_path = os.path.join(output_dir, "biases.bin")
    with open(b_path, "wb") as f:
        f.write(bias_bytes)
    print(f"Wrote: {b_path}  ({len(bias_bytes)} bytes)")

    # Write a human-readable summary
    summary_path = os.path.join(output_dir, "model_summary.txt")
    with open(summary_path, "w") as f:
        f.write("CNN Hardware Model Summary\n")
        f.write("==========================\n\n")
        f.write(f"Input: {input_h}x{input_w}x{input_c}\n\n")
        for lyr in layers:
            if lyr.layer_type == 'conv2d':
                oh = (lyr.in_h + 2*lyr.pad - lyr.k_h) // lyr.stride + 1
                ow = (lyr.in_w + 2*lyr.pad - lyr.k_w) // lyr.stride + 1
                f.write(f"Layer: {lyr.name}\n")
                f.write(f"  Type   : Conv2D\n")
                f.write(f"  Input  : {lyr.in_h}x{lyr.in_w}x{lyr.in_c}\n")
                f.write(f"  Output : {oh}x{ow}x{lyr.out_c}\n")
                f.write(f"  Kernel : {lyr.k_h}x{lyr.k_w}  Stride:{lyr.stride}  Pad:{lyr.pad}\n")
                f.write(f"  Opcode : {lyr.simd_opcode:04b}  (")
                ops = []
                if lyr.simd_opcode & OP_RELU:      ops.append("relu")
                if lyr.simd_opcode & OP_MAXPOOL:   ops.append("maxpool")
                if lyr.simd_opcode & OP_GAVGPOOL:  ops.append("gap")
                if lyr.simd_opcode & OP_QUANTIZE:  ops.append("quantize")
                f.write("+".join(ops) + ")\n")
                f.write(f"  Weights: {lyr.w_int8.size} INT8  scale={lyr.w_scale:.6f}\n")
                f.write(f"  Biases : {lyr.b_int32.size} INT32\n")
                f.write(f"  Last   : {lyr.is_last}\n\n")

    print(f"Wrote: {summary_path}")

    # Also write a flat text file for easy $readmemh in ModelSim
    # weights packed as signed hex bytes, 4 per 32-bit word
    w_mem_path = os.path.join(output_dir, "weights.mem")
    with open(w_mem_path, "w") as f:
        # Pad to multiple of 4 bytes
        padded = list(weight_bytes)
        while len(padded) % 4 != 0:
            padded.append(0)
        for i in range(0, len(padded), 4):
            word = (padded[i+3] << 24) | (padded[i+2] << 16) | \
                   (padded[i+1] << 8)  |  padded[i]
            f.write(f"{word:08x}\n")
    print(f"Wrote: {w_mem_path}  (for $readmemh)")

    b_mem_path = os.path.join(output_dir, "biases.mem")
    with open(b_mem_path, "w") as f:
        for i in range(0, len(bias_bytes), 4):
            word = struct.unpack('<I', bias_bytes[i:i+4])[0]
            f.write(f"{word:08x}\n")
    print(f"Wrote: {b_mem_path}  (for $readmemh)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert Keras .h5 to HW files")
    parser.add_argument("--model",       required=True, help="Path to .h5 model")
    parser.add_argument("--output",      default="./hw_model", help="Output directory")
    parser.add_argument("--input_scale", type=float, default=1.0,
                        help="Input activation scale (default 1.0 for INT8 input)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    layers = parse_keras_model(args.model, args.input_scale)

    if not layers:
        print("ERROR: No supported layers found in model")
        sys.exit(1)

    # Get input shape from first layer
    first = layers[0]
    write_output(layers, args.output, first.in_h, first.in_w, first.in_c)

    print(f"\nDone. {len(layers)} hardware layers extracted.")
    print(f"Output: {args.output}/")
    print(f"  model_config.json  - layer descriptors for testbench")
    print(f"  weights.bin        - INT8 weights, all layers")
    print(f"  biases.bin         - INT32 biases, all layers")
    print(f"  weights.mem        - hex format for $readmemh")
    print(f"  biases.mem         - hex format for $readmemh")
    print(f"  model_summary.txt  - human readable layer info")


if __name__ == "__main__":
    main()