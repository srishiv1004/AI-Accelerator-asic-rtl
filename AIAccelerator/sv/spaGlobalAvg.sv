// =============================================================================
// simd_global_avgpool.sv  ?  streaming Global Average Pooling
//
// Mirrors spatial_maxpool: accumulates on the fly, fires ONE output beat.
// Uses an internal beat counter instead of pixel_addr ? works correctly with
// both 2-px packing (gearbox skips odd addresses) and 1-px mode.
//
// TOTAL_BEATS = number of valid_i beats to consume before firing:
//   OC<=4 (2-px/beat): TOTAL_BEATS = ceil(OUT_H*OUT_W / 2)
//   OC>4  (1-px/beat): TOTAL_BEATS = OUT_H*OUT_W
//
// Latency: 2 cycles after the last beat (accumulate ? multiply ? output).
// Resets accumulator automatically at beat 0 of each new pass.
// =============================================================================
`timescale 1ns/1ps

module simd_global_avgpool #(
    parameter int LANES       = 8,
    parameter int DATA_W      = 32,    // Q format signed
    parameter int FRAC_BITS   = 16,
    parameter int ACC_W       = 48,    // must be >= DATA_W + log2(max beats)
    parameter int TOTAL_BEATS = 7938   // ceil(126*126/2) for default config
)(
    input  logic clk,
    input  logic rst,   // active-high

    input  logic                                  valid_i,
    input  logic signed [LANES-1:0][DATA_W-1:0]  data_i,

    // 1/TOTAL_PIX in Q16.16 ? tie to constant at instantiation
    input  logic signed [DATA_W-1:0]              reciprocal,

    // Fires once with the per-lane average, 2 cycles after last beat
    output logic                                  valid_o,
    output logic signed [LANES-1:0][DATA_W-1:0]  data_avg
);

    // =========================================================================
    // Stage 1: Running accumulator + beat counter
    // =========================================================================
    logic signed [LANES-1:0][ACC_W-1:0] accumulator;
    logic signed [LANES-1:0][ACC_W-1:0] s1_final_sum;
    logic                               s1_valid;
    logic [31:0]                        beat_cnt;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            accumulator  <= '0;
            s1_final_sum <= '0;
            s1_valid     <= 1'b0;
            beat_cnt     <= 32'd0;
        end else begin
            s1_valid <= 1'b0;

            if (valid_i) begin
                automatic logic [31:0] next_cnt;
                next_cnt = beat_cnt + 32'd1;

                for (int i = 0; i < LANES; i++) begin
                    automatic logic signed [ACC_W-1:0] next_acc;
                    // Reset on first beat of each pass
                    next_acc = (beat_cnt == 32'd0)
                             ? ACC_W'($signed(data_i[i]))
                             : accumulator[i] + ACC_W'($signed(data_i[i]));
                    accumulator[i] <= next_acc;
                    if (next_cnt == TOTAL_BEATS)
                        s1_final_sum[i] <= next_acc;
                end

                if (next_cnt == TOTAL_BEATS) begin
                    s1_valid <= 1'b1;
                    beat_cnt <= 32'd0;   // auto-reset for next pass
                end else begin
                    beat_cnt <= next_cnt;
                end
            end
        end
    end

    // =========================================================================
    // Stage 2: Multiply sum * reciprocal ? average
    // prod is ACC_W+DATA_W wide; shift right by FRAC_BITS to re-normalise.
    // =========================================================================
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_o  <= 1'b0;
            data_avg <= '0;
        end else begin
            valid_o <= s1_valid;
            if (s1_valid) begin
                for (int i = 0; i < LANES; i++) begin
                    automatic logic signed [ACC_W+DATA_W-1:0] prod;
                    prod        = s1_final_sum[i] * $signed(reciprocal);
                    data_avg[i] <= DATA_W'($signed(prod) >>> FRAC_BITS);
                end
            end
        end
    end

    initial begin
        assert (TOTAL_BEATS >= 1)
            else $fatal(1, "simd_global_avgpool: TOTAL_BEATS must be >= 1");
        assert (ACC_W >= DATA_W + 1)
            else $fatal(1, "simd_global_avgpool: ACC_W too narrow for accumulation");
        $display("[simd_global_avgpool] LANES=%0d TOTAL_BEATS=%0d ACC_W=%0d",
                 LANES, TOTAL_BEATS, ACC_W);
    end

endmodule
