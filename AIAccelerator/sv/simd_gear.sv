`timescale 1ns/1ps

// =============================================================================
// simd_gearbox_8lane
//
// Packs TWO consecutive output pixels into one 8-lane SIMD beat:
//
//   Lane:  [7]    [6]    [5]    [4]    [3]    [2]    [1]    [0]
//          00     px1    px1    px1    00     px0    px0    px0
//                 CH2    CH1    CH0           CH2    CH1    CH0
//
// meta_addr[0] = pixel address of px0 (arrived first / lower index)
// meta_addr[1] = pixel address of px1 (arrived second / higher index)
// valid_mask[0] = px0 slot valid, valid_mask[1] = px1 slot valid
//
// If tile has an odd pixel count, the last beat fires with px0 only:
//   lanes 4-7 = 0, valid_mask = 2'b01, meta_addr[1] = don't-care
//
// OC must be <= 4 for 2-pixel packing. OC 5-8 falls back to 1 pixel/beat.
// OC > 8 uses a second flush beat (legacy path, unlikely in practice).
// =============================================================================

module simd_gearbox_8lane #(
    parameter int OC = 3    // Active output channels per pixel (must be <= 4)
)(
    input  logic               clk,
    input  logic               rst,

    // From compute_engine
    input  logic               acc_valid,
    input  logic [4:0]         out_eff_cols,       // active channels this tile
    input  logic [31:0]        out_byte_addr,      // spatial pixel index
    input  logic signed [31:0] acc_out [16],       // [0..OC-1] valid

    // To SIMD wrapper
    output logic               simd_valid,
    output logic [1:0][31:0]   simd_meta_addr,     // [0]=px0, [1]=px1
    output logic [1:0]         simd_valid_mask,    // [0]=px0 valid, [1]=px1 valid
    output logic signed [7:0][31:0] simd_data,

    output logic               stall_array
);

    // -------------------------------------------------------------------------
    // Pair-buffer: holds the first pixel while waiting for the second
    // -------------------------------------------------------------------------
    logic               has_buffered;
    logic [31:0]        buf_addr;
    logic signed [31:0] buf_ch [4];        // up to 4 channels

    // Flush-phase for OC > 8 overflow (legacy)
    logic               flush_phase;
    logic [31:0]        flush_meta0, flush_meta1;
    logic [1:0]         flush_mask;
    logic signed [31:0] flush_data [8];

    // -------------------------------------------------------------------------
    // Pack two pixels into 8 lanes:
    //   px0 -> lanes [3:0], px1 -> lanes [7:4]
    // -------------------------------------------------------------------------
    function automatic logic signed [7:0][31:0] pack_two(
        input logic signed [31:0] p0 [4],
        input logic signed [31:0] p1 [4],
        input int                 oc
    );
        logic signed [7:0][31:0] d;
        for (int i = 0; i < 4; i++) begin
            d[i]   = (i < oc) ? p0[i] : 32'sd0;
            d[i+4] = (i < oc) ? p1[i] : 32'sd0;
        end
        return d;
    endfunction

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            has_buffered    <= 1'b0;
            buf_addr        <= '0;
            for (int i = 0; i < 4; i++) buf_ch[i] <= '0;
            flush_phase     <= 1'b0;
            flush_meta0     <= '0;
            flush_meta1     <= '0;
            flush_mask      <= '0;
            for (int i = 0; i < 8; i++) flush_data[i] <= '0;
            simd_valid      <= 1'b0;
            simd_meta_addr  <= '0;
            simd_valid_mask <= 2'b00;
            simd_data       <= '0;
            stall_array     <= 1'b0;

        end else begin
            // Default: outputs deasserted
            simd_valid      <= 1'b0;
            simd_valid_mask <= 2'b00;
            stall_array     <= 1'b0;

            // ------------------------------------------------------------------
            // P1: flush overflow beat (OC > 8, legacy)
            // ------------------------------------------------------------------
            if (flush_phase) begin
                simd_valid      <= 1'b1;
                simd_valid_mask <= flush_mask;
                simd_meta_addr  <= {flush_meta1, flush_meta0};
                for (int i = 0; i < 8; i++)
                    simd_data[i] <= flush_data[i];
                flush_phase <= 1'b0;

            // ------------------------------------------------------------------
            // P2: new pixel arrives
            // ------------------------------------------------------------------
            end else if (acc_valid) begin

                if (int'(out_eff_cols) <= 4) begin
                    // === 2-pixel packing mode ===
                    if (!has_buffered) begin
                        // First of a pair: buffer it, no output yet
                        has_buffered <= 1'b1;
                        buf_addr     <= out_byte_addr;
                        for (int i = 0; i < 4; i++)
                            buf_ch[i] <= (i < int'(out_eff_cols))
                                         ? acc_out[i] : 32'sd0;

                    end else begin
                        // Second of a pair: fire both
                        has_buffered <= 1'b0;

                        begin : fire_pair
                            automatic logic signed [31:0] cur [4];
                            for (int i = 0; i < 4; i++)
                                cur[i] = (i < int'(out_eff_cols))
                                         ? acc_out[i] : 32'sd0;

                            simd_valid      <= 1'b1;
                            simd_valid_mask <= 2'b11;
                            simd_meta_addr  <= {out_byte_addr, // [1] px1
                                                buf_addr};     // [0] px0
                            simd_data       <= pack_two(buf_ch, cur,
                                                        int'(out_eff_cols));
                        end
                    end

                end else begin
                    // === Legacy: OC 5-8, one pixel per beat on lanes 0-7 ===
                    has_buffered    <= 1'b0;
                    simd_valid      <= 1'b1;
                    simd_valid_mask <= 2'b01;
                    simd_meta_addr  <= {out_byte_addr, out_byte_addr};
                    for (int i = 0; i < 8; i++)
                        simd_data[i] <= (i < int'(out_eff_cols))
                                        ? acc_out[i] : 32'sd0;

                    // OC > 8: stash overflow for next cycle
                    if (out_eff_cols > 8) begin
                        flush_phase  <= 1'b1;
                        stall_array  <= 1'b1;
                        flush_meta0  <= out_byte_addr;
                        flush_meta1  <= out_byte_addr;
                        flush_mask   <= 2'b01;
                        for (int i = 0; i < 8; i++)
                            flush_data[i] <= ((i + 8) < int'(out_eff_cols))
                                             ? acc_out[i + 8] : 32'sd0;
                    end
                end

            // ------------------------------------------------------------------
            // P3: acc_valid gone but odd pixel still buffered (end of tile)
            // ------------------------------------------------------------------
            end else if (has_buffered) begin
                has_buffered    <= 1'b0;
                simd_valid      <= 1'b1;
                simd_valid_mask <= 2'b01;               // px0 only
                simd_meta_addr  <= {buf_addr, buf_addr};// [1] don't-care
                for (int i = 0; i < 4; i++) begin
                    simd_data[i]   <= buf_ch[i];
                    simd_data[i+4] <= 32'sd0;
                end
            end
        end
    end

endmodule
