// ============================================================================
// Package
// ============================================================================
package simd_pipeline_pkg;
    typedef enum int {
        STAGE_INPUT    = 0,
        STAGE_RELU     = 1,
        STAGE_MAXPOOL  = 2,
        STAGE_GAVGPOOL = 3,
        STAGE_QUANTIZE = 4,
        STAGE_OUTPUT   = 5
    } stage_id_t;

    // Opcode bit mapping (4 bits, inference-only - no BN or ADD):
    // bit3=quantize, bit2=gavgpool, bit1=maxpool, bit0=relu
    typedef struct packed {
        logic quantize;
        logic gavgpool;
        logic maxpool;
        logic relu;
    } opcode_t;

    localparam int NUM_STAGES = 5;
endpackage

// ============================================================================
// SIMD Pipeline Wrapper  (inference-only, BN and ADD removed)
//
// Pipeline:  ReLU -> MaxPool (spatial, external) -> GAP -> Quantize
//
// Opcodes:
//   Hidden conv layer (relu+quantize)        : 4'b1001
//   Hidden conv + maxpool (relu+mp+quantize) : 4'b1011
//   Last layer with GAP (relu+gap+quantize)  : 4'b1101
//
// MaxPool note: spatial 2D maxpool is handled OUTSIDE this wrapper.
//   The MAXPOOL stage here is a 1-cycle pass-through register only ?
//   it just holds the data for one cycle while the external maxpool
//   processes it. The opcode bit is kept so the testbench knows whether
//   to route through the external maxpool module.
//
// GAP TOTAL_BEATS:
//   With maxpool:    ceil(MP_H * MP_W / 2)   (2px/beat from gearbox)
//   Without maxpool: ceil(IN_H * IN_W / 2)
// ============================================================================
module simd_pipeline_wrapper #(
    parameter int LANES       = 8,
    parameter int DATA_W      = 32,
    parameter int FRAC_BITS   = 4,      // quantizer shift amount
    parameter int OPCODE_W    = 4,      // reduced from 6 to 4
    // GAP
    parameter int TOTAL_BEATS = 98,
    // Spatial dims (used by external maxpool, passed here for reference)
    parameter int IN_H        = 14,
    parameter int IN_W        = 14,
    parameter int POOL_K      = 2,
    parameter int POOL_ST     = 2,
    // Derived
    parameter int MP_H        = (IN_H - POOL_K) / POOL_ST + 1,
    parameter int MP_W        = (IN_W - POOL_K) / POOL_ST + 1
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                                valid_i,
    input  logic [1:0]                          valid_mask_i,
    input  logic [OPCODE_W-1:0]                 opcode_i,
    input  logic [1:0][31:0]                    meta_i,
    input  logic signed [LANES-1:0][DATA_W-1:0] data_i,

    input  logic signed [DATA_W-1:0]            gap_reciprocal,

    output logic                                valid_o,
    output logic [1:0]                          valid_mask_o,
    output logic [1:0][31:0]                    meta_o,
    output logic signed [LANES-1:0][DATA_W-1:0] data_o,

    output logic                                pipeline_busy,
    output logic [2:0]                          current_stage
);
    import simd_pipeline_pkg::*;

    logic signed [NUM_STAGES:0][LANES-1:0][DATA_W-1:0] stage_data;
    logic        [NUM_STAGES:0]                         stage_valid;
    logic        [NUM_STAGES:0][1:0]                    stage_valid_mask;
    logic        [NUM_STAGES:0][OPCODE_W-1:0]           stage_opcode;
    logic        [NUM_STAGES:0][1:0][31:0]              stage_meta;

    assign stage_data[STAGE_INPUT]       = data_i;
    assign stage_valid[STAGE_INPUT]      = valid_i;
    assign stage_valid_mask[STAGE_INPUT] = valid_mask_i;
    assign stage_opcode[STAGE_INPUT]     = opcode_i;
    assign stage_meta[STAGE_INPUT]       = meta_i;

    assign data_o       = stage_data[STAGE_OUTPUT];
    assign valid_o      = stage_valid[STAGE_OUTPUT];
    assign valid_mask_o = stage_valid_mask[STAGE_OUTPUT];
    assign meta_o       = stage_meta[STAGE_OUTPUT];

    opcode_t op_decoded[NUM_STAGES:0];
    generate
        for (genvar s = 0; s <= NUM_STAGES; s++) begin : gen_opcode_decode
            assign op_decoded[s] = opcode_t'(stage_opcode[s]);
        end
    endgenerate

    // ========================================================================
    // STAGE 1: ReLU (1 cycle)
    // ========================================================================
    logic signed [LANES-1:0][DATA_W-1:0] relu_data_o;
    logic relu_enable_d;
    logic signed [LANES-1:0][DATA_W-1:0] relu_data_in_d;

    // ReLU module is combinational ? register input for 1-cycle latency
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            relu_enable_d  <= '0;
            relu_data_in_d <= '0;
        end else begin
            relu_enable_d  <= op_decoded[STAGE_INPUT].relu;
            relu_data_in_d <= stage_data[STAGE_INPUT];
        end
    end

    // ReLU: max(0, x) per lane
    always_comb begin
        for (int i = 0; i < LANES; i++)
            relu_data_o[i] = ($signed(relu_data_in_d[i]) > 0)
                             ? relu_data_in_d[i] : '0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage_data[STAGE_RELU]       <= '0;
            stage_valid[STAGE_RELU]      <= '0;
            stage_opcode[STAGE_RELU]     <= '0;
            stage_meta[STAGE_RELU]       <= '0;
            stage_valid_mask[STAGE_RELU] <= '0;
        end else begin
            stage_data[STAGE_RELU]       <= relu_enable_d ? relu_data_o
                                                          : relu_data_in_d;
            stage_valid[STAGE_RELU]      <= stage_valid[STAGE_INPUT];
            stage_opcode[STAGE_RELU]     <= stage_opcode[STAGE_INPUT];
            stage_meta[STAGE_RELU]       <= stage_meta[STAGE_INPUT];
            stage_valid_mask[STAGE_RELU] <= stage_valid_mask[STAGE_INPUT];
        end
    end

    // ========================================================================
    // STAGE 2: MAXPOOL pass-through (1 cycle register)
    // Spatial 2D maxpool is handled externally. This stage is a pipeline
    // register only ? it keeps latency uniform whether maxpool is on or off.
    // The opcode bit signals the testbench to route through the external SMP.
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage_data[STAGE_MAXPOOL]       <= '0;
            stage_valid[STAGE_MAXPOOL]      <= '0;
            stage_opcode[STAGE_MAXPOOL]     <= '0;
            stage_meta[STAGE_MAXPOOL]       <= '0;
            stage_valid_mask[STAGE_MAXPOOL] <= '0;
        end else begin
            stage_data[STAGE_MAXPOOL]       <= stage_data[STAGE_RELU];
            stage_valid[STAGE_MAXPOOL]      <= stage_valid[STAGE_RELU];
            stage_opcode[STAGE_MAXPOOL]     <= stage_opcode[STAGE_RELU];
            stage_meta[STAGE_MAXPOOL]       <= stage_meta[STAGE_RELU];
            stage_valid_mask[STAGE_MAXPOOL] <= stage_valid_mask[STAGE_RELU];
        end
    end

    // ========================================================================
    // STAGE 3: GLOBAL AVERAGE POOL
    // Fires once after TOTAL_BEATS valid beats, 2 cycles after last beat.
    // ========================================================================
    logic gap_valid_o;
    logic signed [LANES-1:0][DATA_W-1:0] gap_data_o;
    logic [OPCODE_W-1:0] gap_opcode_latch;
    logic [1:0][31:0]    gap_meta_latch;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gap_opcode_latch <= '0;
            gap_meta_latch   <= '0;
        end else if (stage_valid[STAGE_MAXPOOL] & op_decoded[STAGE_MAXPOOL].gavgpool) begin
            gap_opcode_latch <= stage_opcode[STAGE_MAXPOOL];
            gap_meta_latch   <= stage_meta[STAGE_MAXPOOL];
        end
    end

    simd_global_avgpool #(
        .LANES       (LANES),
        .DATA_W      (DATA_W),
        .FRAC_BITS   (DATA_W/2),
        .ACC_W       (48),
        .TOTAL_BEATS (TOTAL_BEATS)
    ) gap_inst (
        .clk        (clk),
        .rst        (~rst_n),
        .valid_i    (stage_valid[STAGE_MAXPOOL] & op_decoded[STAGE_MAXPOOL].gavgpool),
        .data_i     (stage_data[STAGE_MAXPOOL]),
        .reciprocal (gap_reciprocal),
        .valid_o    (gap_valid_o),
        .data_avg   (gap_data_o)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage_data[STAGE_GAVGPOOL]       <= '0;
            stage_valid[STAGE_GAVGPOOL]      <= '0;
            stage_opcode[STAGE_GAVGPOOL]     <= '0;
            stage_meta[STAGE_GAVGPOOL]       <= '0;
            stage_valid_mask[STAGE_GAVGPOOL] <= '0;
        end else begin
            if (op_decoded[STAGE_MAXPOOL].gavgpool) begin
                stage_valid[STAGE_GAVGPOOL]      <= gap_valid_o;
                stage_data[STAGE_GAVGPOOL]       <= gap_valid_o ? gap_data_o
                                                                : stage_data[STAGE_GAVGPOOL];
                stage_opcode[STAGE_GAVGPOOL]     <= gap_valid_o ? gap_opcode_latch
                                                                : stage_opcode[STAGE_GAVGPOOL];
                stage_meta[STAGE_GAVGPOOL]       <= gap_valid_o ? gap_meta_latch
                                                                : stage_meta[STAGE_GAVGPOOL];
                stage_valid_mask[STAGE_GAVGPOOL] <= gap_valid_o ? 2'b01 : 2'b00;
            end else begin
                stage_data[STAGE_GAVGPOOL]       <= stage_data[STAGE_MAXPOOL];
                stage_valid[STAGE_GAVGPOOL]      <= stage_valid[STAGE_MAXPOOL];
                stage_opcode[STAGE_GAVGPOOL]     <= stage_opcode[STAGE_MAXPOOL];
                stage_meta[STAGE_GAVGPOOL]       <= stage_meta[STAGE_MAXPOOL];
                stage_valid_mask[STAGE_GAVGPOOL] <= stage_valid_mask[STAGE_MAXPOOL];
            end
        end
    end

    // ========================================================================
    // STAGE 4: QUANTIZER (combinational logic, registered output)
    // Shift right by FRAC_BITS with round-to-nearest, saturate to DATA_W.
    // ========================================================================
    logic signed [LANES-1:0][DATA_W-1:0] quant_data_o;

    generate
        for (genvar i = 0; i < LANES; i++) begin : gen_quantizers
            ai_quantizer #(
                .IN_WIDTH  (DATA_W),
                .OUT_WIDTH (DATA_W),
                .FRAC_BITS (FRAC_BITS)
            ) quant_inst (
                .data_in  (stage_data[STAGE_GAVGPOOL][i]),
                .data_out (quant_data_o[i])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage_data[STAGE_QUANTIZE]       <= '0;
            stage_valid[STAGE_QUANTIZE]      <= '0;
            stage_opcode[STAGE_QUANTIZE]     <= '0;
            stage_meta[STAGE_QUANTIZE]       <= '0;
            stage_valid_mask[STAGE_QUANTIZE] <= '0;
        end else begin
            stage_data[STAGE_QUANTIZE]       <= op_decoded[STAGE_GAVGPOOL].quantize
                                               ? quant_data_o
                                               : stage_data[STAGE_GAVGPOOL];
            stage_valid[STAGE_QUANTIZE]      <= stage_valid[STAGE_GAVGPOOL];
            stage_opcode[STAGE_QUANTIZE]     <= stage_opcode[STAGE_GAVGPOOL];
            stage_meta[STAGE_QUANTIZE]       <= stage_meta[STAGE_GAVGPOOL];
            stage_valid_mask[STAGE_QUANTIZE] <= stage_valid_mask[STAGE_GAVGPOOL];
        end
    end

    // ========================================================================
    // OUTPUT
    // ========================================================================
    assign stage_data[STAGE_OUTPUT]       = stage_data[STAGE_QUANTIZE];
    assign stage_valid[STAGE_OUTPUT]      = stage_valid[STAGE_QUANTIZE];
    assign stage_opcode[STAGE_OUTPUT]     = stage_opcode[STAGE_QUANTIZE];
    assign stage_meta[STAGE_OUTPUT]       = stage_meta[STAGE_QUANTIZE];
    assign stage_valid_mask[STAGE_OUTPUT] = stage_valid_mask[STAGE_QUANTIZE];

    // ========================================================================
    // Pipeline status
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipeline_busy <= '0;
            current_stage <= '0;
        end else begin
            pipeline_busy <= |stage_valid[NUM_STAGES-1:0];
            if      (stage_valid[STAGE_RELU])     current_stage <= 3'd1;
            else if (stage_valid[STAGE_MAXPOOL])  current_stage <= 3'd2;
            else if (stage_valid[STAGE_GAVGPOOL]) current_stage <= 3'd3;
            else if (stage_valid[STAGE_QUANTIZE]) current_stage <= 3'd4;
            else                                  current_stage <= 3'd0;
        end
    end

endmodule
