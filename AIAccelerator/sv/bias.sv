// =============================================================================
// bias_loader.sv  (v2 - cleaned)
//
// Loads REAL_K signed 32-bit bias values from DRAM B into bias_rf_o[].
// Called once per inference (not per tile) by compute_engine in MS_RESET.
//
// Timing: REAL_K + 2 cycles  (1 IDLE + REAL_K + 1 LOAD cycles)
//   Cycle 0 (IDLE): start_i fires, -> BS_LOAD.  addr[0] issued combinationally.
//   Cycle 1 (ld_cnt=0): no write (pipeline fill). addr[1] issued. ld_cnt->1.
//   Cycle 2 (ld_cnt=1): bias_rf[0] <- data[0]. addr[2] issued. ld_cnt->2.
//   ...
//   Cycle K (ld_cnt=K-1): bias_rf[K-2] <- data[K-2]. addr[K-1] issued. ld_cnt->K-1.
//   Cycle K+1 (ld_cnt=K): bias_rf[K-1] <- data[K-1]. ld_cnt==REAL_K -> done_o=1, -> BS_IDLE.
//
// Changes from v1:
//   - BS_DONE state removed: done_o is now combinational (asserted for 1 cycle when
//     ld_cnt==REAL_K in BS_LOAD), and the FSM returns directly to BS_IDLE.
//     Saves 1 cycle per inference, simplifies state machine.
//   - bias_rf_o port now directly references bias_rf (no copy register file).
//   - ld_cnt width corrected: $clog2(REAL_K+1) bits, not +1 extra bit.
// =============================================================================

module bias_loader #(
    parameter int REAL_K    = 6,
    parameter int BIAS_BASE = 0
)(
    input  logic clk,
    input  logic rst,

    input  logic start_i,
    output logic done_o,
    output logic busy_o,

    output logic [15:0] ext_b_addr_o,
    output logic        ext_b_rd_en_o,
    input  logic [31:0] ext_b_data_i,

    output logic signed [31:0] bias_rf_o [REAL_K]
);

    // -------------------------------------------------------------------------
    // FSM ? two states only (DONE state removed)
    // -------------------------------------------------------------------------
    typedef enum logic {
        BS_IDLE = 1'b0,
        BS_LOAD = 1'b1
    } bs_t;
    bs_t bs_state;

    // ld_cnt: counts 0..REAL_K.
    //   In LOAD: addr[ld_cnt] issued combinationally this cycle.
    //            bias_rf[ld_cnt-1] received from last cycle's read.
    //   When ld_cnt == REAL_K: last write just occurred, assert done_o.
    logic [$clog2(REAL_K+1)-1:0] ld_cnt;

    // Bias register file ? directly wired to output port
    logic signed [31:0] bias_rf [REAL_K];
    assign bias_rf_o = bias_rf;

    // -------------------------------------------------------------------------
    // Sequential FSM
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            bs_state <= BS_IDLE;
            ld_cnt   <= '0;
            busy_o   <= 1'b0;
            for (int i = 0; i < REAL_K; i++) bias_rf[i] <= '0;
        end else begin
            case (bs_state)

                BS_IDLE: begin
                    if (start_i) begin
                        ld_cnt   <= '0;
                        busy_o   <= 1'b1;
                        bs_state <= BS_LOAD;
                    end
                end

                BS_LOAD: begin
                    // Receive: data for address (ld_cnt - 1) has arrived
                    if (ld_cnt >= 1)
                        bias_rf[ld_cnt - 1] <= $signed(ext_b_data_i);

                    if (int'(ld_cnt) == REAL_K) begin
                        // Last bias just written ? return to idle
                        busy_o   <= 1'b0;
                        ld_cnt   <= '0;
                        bs_state <= BS_IDLE;
                    end else begin
                        ld_cnt <= $bits(ld_cnt)'(int'(ld_cnt) + 1);
                    end
                end

                default: bs_state <= BS_IDLE;

            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Combinational: DRAM B read enable and address
    // Issue addr[ld_cnt] whenever we are loading and haven't yet issued all reads.
    // -------------------------------------------------------------------------
    always_comb begin
        ext_b_rd_en_o = 1'b0;
        ext_b_addr_o  = '0;
        if (bs_state == BS_LOAD && int'(ld_cnt) < REAL_K) begin
            ext_b_rd_en_o = 1'b1;
            ext_b_addr_o  = 16'(BIAS_BASE + int'(ld_cnt));
        end
    end

    // -------------------------------------------------------------------------
    // Combinational: done_o ? pulses for exactly 1 cycle when the last bias
    // has been written into bias_rf (ld_cnt == REAL_K at end of BS_LOAD).
    // compute_engine samples this in MS_LOAD_BIAS.
    // -------------------------------------------------------------------------
    assign done_o = (bs_state == BS_LOAD) && (int'(ld_cnt) == REAL_K);

endmodule
