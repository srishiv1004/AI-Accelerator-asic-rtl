// =============================================================================
// dram_loader  ?  im2col activations + weights + optional bias into SRAM A/B
//
// Bias trick:  append constant "1" to every im2col patch row  (SRAM A),
//              append bias[oc] to every filter tap row          (SRAM B).
// The systolic array then computes  x·w + bias  automatically.
//
// USE_BIAS=1 : effective M = REAL_M + 1  (one extra dot-product tap)
// USE_BIAS=0 : effective M = REAL_M      (legacy, unchanged behaviour)
//
// Caller must set the systolic-array K dimension to REAL_M + USE_BIAS.
// =============================================================================
`timescale 1ns/1ps

module dram_loader #(
    parameter int TILE_MAX = 16,
    parameter int REAL_K   = 4,
    parameter int REAL_M   = 9,     // K_H*K_W*IN_C  (WITHOUT bias column)
    parameter int REAL_N   = 1,
    parameter int IN_H     = 128,
    parameter int IN_W     = 128,
    parameter int IN_C     = 3,
    parameter int K_H      = 3,
    parameter int K_W      = 3,
    parameter int STRIDE   = 1,
    parameter int PAD      = 0,
    parameter int AW       = 8,
    parameter int USE_BIAS = 1      // 1 = append bias; 0 = legacy
)(
    input  logic clk,
    input  logic rst,

    input  logic        start_i,
    input  logic        skip_x_i,
    output logic        load_ready_o,
    output logic        busy_o,

    input  logic [$clog2(REAL_K):0]        row_tile_i,
    input  logic [$clog2(REAL_N):0]        col_tile_i,
    input  logic [$clog2(TILE_MAX+1)-1:0]  eff_rows_i,
    input  logic [$clog2(TILE_MAX+1)-1:0]  eff_cols_i,

    input  logic [15:0]  words_a_i,
    input  logic [15:0]  words_b_i,

    input  logic [AW-1:0] sram_a_base_i,

    // ---- activation DRAM port ----
    output logic [15:0] ext_x_addr_o,
    output logic        ext_x_rd_en_o,
    input  logic [31:0] ext_x_data_i,

    // ---- weight DRAM port ----
    output logic [15:0] ext_y_addr_o,
    output logic        ext_y_rd_en_o,
    input  logic [31:0] ext_y_data_i,

    // ---- bias DRAM port (ignored when USE_BIAS=0) ----
    // bias[oc] stored as flat INT8 array, 4-per-word:
    //   word = bias_base_i + oc/4,  byte lane = oc%4
    output logic [15:0] ext_b_addr_o,
    output logic        ext_b_rd_en_o,
    input  logic [31:0] ext_b_data_i,
    input  logic [15:0] bias_base_i,

    // ---- SRAM A (activations) ----
    output logic          sram_a_csb0,
    output logic          sram_a_web0,
    output logic [3:0]    sram_a_wmask0,
    output logic [AW-1:0] sram_a_addr0,
    output logic [31:0]   sram_a_din0,

    // ---- SRAM B (weights) ----
    output logic                     sram_b_csb0,
    output logic                     sram_b_web0,
    output logic [TILE_MAX-1:0]      sram_b_wmask0,
    output logic [AW-1:0]            sram_b_addr0,
    output logic [(TILE_MAX*8)-1:0]  sram_b_din0
);

    // -------------------------------------------------------------------------
    // REAL_M_EFF = actual number of dot-product taps (with or without bias)
    // -------------------------------------------------------------------------
    localparam int REAL_M_EFF = REAL_M + USE_BIAS;
    localparam int OUT_W      = (IN_W + 2*PAD - K_W) / STRIDE + 1;

    // =========================================================================
    // FSM types
    // =========================================================================
    typedef enum logic [2:0] {
        ACT_IDLE=3'd0, ACT_PIPE=3'd1, ACT_DRAIN=3'd2,
        ACT_FLUSH=3'd3, ACT_DONE=3'd4
    } act_t;
    act_t act_state;

    typedef enum logic [2:0] {
        WT_IDLE=3'd0, WT_PIPE=3'd1, WT_DRAIN=3'd2,
        WT_FLUSH=3'd3, WT_DONE=3'd4
    } wt_t;
    wt_t wt_state;

    // =========================================================================
    // Latched tile info
    // =========================================================================
    logic [$clog2(REAL_K):0]       lat_row_tile;
    logic [$clog2(REAL_N):0]       lat_col_tile;
    logic [$clog2(TILE_MAX+1)-1:0] lat_eff_rows;
    logic [$clog2(TILE_MAX+1)-1:0] lat_eff_cols;
    logic [AW-1:0]                 lat_sram_a_base;
    logic [15:0]                   lat_col_base;   // col_tile * TILE_MAX

    // act_total / wt_total use REAL_M_EFF
    logic [15:0] act_total;
    logic [15:0] wt_total;
    always_comb begin
        act_total = 16'(int'(lat_eff_rows) * REAL_M_EFF);
        wt_total  = 16'(REAL_M_EFF * int'(lat_eff_cols));
    end

    // =========================================================================
    // Activation datapath signals
    // =========================================================================
    logic [15:0] act_issue_cnt;
    logic [15:0] act_ic_row;
    logic [15:0] act_ic_col;   // 0 .. REAL_M_EFF-1

    logic        act_s1_valid;
    logic        act_s1_is_pad;
    logic        act_s1_is_bias;
    logic [1:0]  act_s1_byte_lane;

    logic [7:0]  act_pb0, act_pb1, act_pb2, act_pb3;
    logic [1:0]  act_pack_lane;
    logic [AW-1:0] act_sram_wr;

    // =========================================================================
    // Weight datapath signals
    // =========================================================================
    logic [15:0] wt_issue_cnt;
    logic [15:0] wt_tap;        // 0 .. REAL_M_EFF-1
    logic [15:0] wt_col;        // 0 .. eff_cols-1

    logic        wt_s1_valid;
    logic        wt_s1_is_bias;
    logic [1:0]  wt_s1_byte_lane;
    logic [1:0]  wt_s1_bias_lane;

    logic [(TILE_MAX*8)-1:0] wt_pack_reg;
    logic [$clog2(TILE_MAX)-1:0] wt_pack_lane;
    logic [AW-1:0] wt_sram_wr;

    // =========================================================================
    // Helper functions
    // =========================================================================

    // im2col address (ic_c is the kernel-column index, NOT the bias column)
    function automatic void get_im2col(
        input  int row_tile_in, ic_r, ic_c,
        output logic        is_pad_out,
        output logic [15:0] word_addr_out,
        output logic [1:0]  byte_lane_out
    );
        int glob_row, oh, ow, ic_ch, kh_i, kw_i, src_h, src_w, byte_addr;
        glob_row = row_tile_in + ic_r;
        oh       = glob_row / OUT_W;
        ow       = glob_row % OUT_W;
        ic_ch    = ic_c / (K_H * K_W);
        kh_i     = (ic_c % (K_H * K_W)) / K_W;
        kw_i     = ic_c % K_W;
        src_h    = oh * STRIDE + kh_i - PAD;
        src_w    = ow * STRIDE + kw_i - PAD;
        if (src_h < 0 || src_h >= IN_H || src_w < 0 || src_w >= IN_W) begin
            is_pad_out    = 1'b1;
            word_addr_out = '0;
            byte_lane_out = '0;
        end else begin
            byte_addr     = (src_h * IN_W + src_w) * IN_C + ic_ch;
            is_pad_out    = 1'b0;
            word_addr_out = 16'(byte_addr >> 2);
            byte_lane_out = 2'(byte_addr & 2'h3);
        end
    endfunction

    // weight address (tap_in < REAL_M only; bias tap handled separately)
    function automatic void get_weight_addr(
        input  int tap_in, col_base_in, tile_col_in,
        output logic [15:0] word_addr_out,
        output logic [1:0]  byte_lane_out
    );
        int byte_addr;
        byte_addr     = tap_in * REAL_N + col_base_in + tile_col_in;
        word_addr_out = 16'(byte_addr >> 2);
        byte_lane_out = 2'(byte_addr & 2'h3);
    endfunction

    // bias[oc] address: oc = col_base_in + tile_col_in
    function automatic void get_bias_addr(
        input  int col_base_in, tile_col_in,
        input  logic [15:0] base,
        output logic [15:0] word_addr_out,
        output logic [1:0]  byte_lane_out
    );
        int oc;
        oc            = col_base_in + tile_col_in;
        word_addr_out = 16'(int'(base) + (oc >> 2));
        byte_lane_out = 2'(oc & 2'h3);
    endfunction

    // =========================================================================
    // Latch tile info + busy
    // =========================================================================
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            lat_row_tile <= '0; lat_col_tile <= '0;
            lat_eff_rows <= '0; lat_eff_cols <= '0;
            lat_sram_a_base <= '0; lat_col_base <= '0;
            busy_o <= 1'b0;
        end else begin
            if (start_i) begin
                lat_row_tile    <= row_tile_i;
                lat_col_tile    <= col_tile_i;
                lat_eff_rows    <= eff_rows_i;
                lat_eff_cols    <= eff_cols_i;
                lat_sram_a_base <= sram_a_base_i;
                lat_col_base    <= 16'(int'(col_tile_i) * TILE_MAX);
                busy_o          <= 1'b1;
            end
            if (load_ready_o) busy_o <= 1'b0;
        end
    end

    assign load_ready_o = (act_state == ACT_DONE) && (wt_state == WT_DONE);

    // =========================================================================
    // Activation FSM
    // =========================================================================
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            act_state      <= ACT_IDLE;
            act_issue_cnt  <= '0; act_ic_row <= '0; act_ic_col <= '0;
            act_s1_valid   <= 1'b0; act_s1_is_pad <= 1'b0;
            act_s1_is_bias <= 1'b0; act_s1_byte_lane <= '0;
            act_pb0<='0; act_pb1<='0; act_pb2<='0; act_pb3<='0;
            act_pack_lane  <= '0; act_sram_wr <= '0;
        end else begin
            case (act_state)
                // --------------------------------------------------------
                ACT_IDLE: begin
                    act_s1_valid <= 1'b0;
                    if (start_i) begin
                        act_ic_row <= '0; act_ic_col <= '0;
                        act_issue_cnt <= '0;
                        act_pb0<='0; act_pb1<='0; act_pb2<='0; act_pb3<='0;
                        act_pack_lane <= '0; act_sram_wr <= sram_a_base_i;
                        if (skip_x_i) begin
                            act_s1_valid <= 1'b0;
                            act_state    <= ACT_DONE;
                        end else begin
                            act_state <= ACT_PIPE;
                        end
                    end
                end

                // --------------------------------------------------------
                ACT_PIPE: begin
                    // Stage-1 writeback
                    if (act_s1_valid) begin
                        logic [7:0] bval;
                        if (act_s1_is_bias)
                            bval = 8'h01;
                        else
                            bval = act_s1_is_pad ? 8'h00
                                 : ext_x_data_i[int'(act_s1_byte_lane)*8 +: 8];
                        case (act_pack_lane)
                            2'd0: act_pb0 <= bval;
                            2'd1: act_pb1 <= bval;
                            2'd2: act_pb2 <= bval;
                            2'd3: act_pb3 <= bval;
                        endcase
                        if (act_pack_lane == 2'd3) begin
                            act_sram_wr   <= AW'(int'(act_sram_wr) + 1);
                            act_pb0<='0; act_pb1<='0; act_pb2<='0; act_pb3<='0;
                            act_pack_lane <= '0;
                        end else begin
                            act_pack_lane <= act_pack_lane + 2'd1;
                        end
                    end

                    // Issue next element
                    if (int'(act_issue_cnt) < int'(act_total)) begin
                        logic is_b;
                        is_b = (USE_BIAS != 0) && (int'(act_ic_col) == REAL_M);

                        act_s1_valid    <= 1'b1;
                        act_s1_is_bias  <= is_b;
                        if (is_b) begin
                            act_s1_is_pad    <= 1'b0;
                            act_s1_byte_lane <= '0;
                        end else begin
                            logic p; logic [15:0] wa; logic [1:0] bl;
                            get_im2col(int'(lat_row_tile), int'(act_ic_row),
                                       int'(act_ic_col), p, wa, bl);
                            act_s1_is_pad    <= p;
                            act_s1_byte_lane <= bl;
                        end

                        if (int'(act_ic_col) == REAL_M_EFF - 1) begin
                            act_ic_col <= '0;
                            act_ic_row <= 16'(int'(act_ic_row) + 1);
                        end else begin
                            act_ic_col <= 16'(int'(act_ic_col) + 1);
                        end
                        act_issue_cnt <= 16'(int'(act_issue_cnt) + 1);
                        if (int'(act_issue_cnt) == int'(act_total) - 1)
                            act_state <= ACT_DRAIN;
                    end else begin
                        act_s1_valid <= 1'b0;
                    end
                end

                // --------------------------------------------------------
                ACT_DRAIN: begin
                    act_s1_valid <= 1'b0;
                    if (act_s1_valid) begin
                        logic [7:0] bval;
                        if (act_s1_is_bias)
                            bval = 8'h01;
                        else
                            bval = act_s1_is_pad ? 8'h00
                                 : ext_x_data_i[int'(act_s1_byte_lane)*8 +: 8];
                        case (act_pack_lane)
                            2'd0: act_pb0 <= bval;
                            2'd1: act_pb1 <= bval;
                            2'd2: act_pb2 <= bval;
                            2'd3: act_pb3 <= bval;
                        endcase
                        if (act_pack_lane == 2'd3) begin
                            act_sram_wr   <= AW'(int'(act_sram_wr) + 1);
                            act_pb0<='0; act_pb1<='0; act_pb2<='0; act_pb3<='0;
                            act_pack_lane <= '0;
                            act_state     <= ACT_DONE;
                        end else begin
                            act_pack_lane <= act_pack_lane + 2'd1;
                            act_state     <= ACT_FLUSH;
                        end
                    end else begin
                        act_state <= (act_pack_lane != 2'd0) ? ACT_FLUSH : ACT_DONE;
                    end
                end

                // --------------------------------------------------------
                ACT_FLUSH: begin
                    act_sram_wr   <= AW'(int'(act_sram_wr) + 1);
                    act_pb0<='0; act_pb1<='0; act_pb2<='0; act_pb3<='0;
                    act_pack_lane <= '0;
                    act_state     <= ACT_DONE;
                end

                ACT_DONE: begin
                    if (load_ready_o) act_state <= ACT_IDLE;
                end
                default: act_state <= ACT_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Weight FSM
    // =========================================================================
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            wt_state       <= WT_IDLE;
            wt_issue_cnt   <= '0; wt_tap <= '0; wt_col <= '0;
            wt_s1_valid    <= 1'b0; wt_s1_is_bias <= 1'b0;
            wt_s1_byte_lane <= '0; wt_s1_bias_lane <= '0;
            wt_pack_reg    <= '0; wt_pack_lane <= '0; wt_sram_wr <= '0;
        end else begin
            case (wt_state)
                // --------------------------------------------------------
                WT_IDLE: begin
                    wt_s1_valid <= 1'b0;
                    if (start_i) begin
                        wt_tap <= '0; wt_col <= '0; wt_issue_cnt <= '0;
                        wt_pack_reg <= '0; wt_pack_lane <= '0; wt_sram_wr <= '0;
                        wt_state <= WT_PIPE;
                    end
                end

                // --------------------------------------------------------
                WT_PIPE: begin
                    // Stage-1 writeback
                    if (wt_s1_valid) begin
                        logic [7:0] bval;
                        if (wt_s1_is_bias)
                            bval = ext_b_data_i[int'(wt_s1_bias_lane)*8 +: 8];
                        else
                            bval = ext_y_data_i[int'(wt_s1_byte_lane)*8 +: 8];
                        wt_pack_reg[int'(wt_pack_lane)*8 +: 8] <= bval;
                        if (int'(wt_pack_lane) == TILE_MAX - 1) begin
                            wt_sram_wr   <= AW'(int'(wt_sram_wr) + 1);
                            wt_pack_lane <= '0;
                        end else begin
                            wt_pack_lane <= wt_pack_lane + 1;
                        end
                    end

                    // Issue next element
                    if (int'(wt_issue_cnt) < int'(wt_total)) begin
                        logic is_b;
                        is_b = (USE_BIAS != 0) && (int'(wt_tap) == REAL_M);

                        wt_s1_valid   <= 1'b1;
                        wt_s1_is_bias <= is_b;
                        if (is_b) begin
                            logic [15:0] bwa; logic [1:0] bbl;
                            get_bias_addr(int'(lat_col_base), int'(wt_col),
                                          bias_base_i, bwa, bbl);
                            wt_s1_byte_lane <= '0;
                            wt_s1_bias_lane <= bbl;
                        end else begin
                            logic [15:0] wa; logic [1:0] bl;
                            get_weight_addr(int'(wt_tap), int'(lat_col_base),
                                            int'(wt_col), wa, bl);
                            wt_s1_byte_lane <= bl;
                            wt_s1_bias_lane <= '0;
                        end

                        if (int'(wt_col) == int'(lat_eff_cols) - 1) begin
                            wt_col <= '0;
                            wt_tap <= 16'(int'(wt_tap) + 1);
                        end else begin
                            wt_col <= 16'(int'(wt_col) + 1);
                        end
                        wt_issue_cnt <= 16'(int'(wt_issue_cnt) + 1);
                        if (int'(wt_issue_cnt) == int'(wt_total) - 1)
                            wt_state <= WT_DRAIN;
                    end else begin
                        wt_s1_valid <= 1'b0;
                    end
                end

                // --------------------------------------------------------
                WT_DRAIN: begin
                    wt_s1_valid <= 1'b0;
                    if (wt_s1_valid) begin
                        logic [7:0] bval;
                        if (wt_s1_is_bias)
                            bval = ext_b_data_i[int'(wt_s1_bias_lane)*8 +: 8];
                        else
                            bval = ext_y_data_i[int'(wt_s1_byte_lane)*8 +: 8];
                        wt_pack_reg[int'(wt_pack_lane)*8 +: 8] <= bval;
                        if (int'(wt_pack_lane) == TILE_MAX - 1) begin
                            wt_sram_wr   <= AW'(int'(wt_sram_wr) + 1);
                            wt_pack_lane <= '0;
                            wt_state     <= WT_DONE;
                        end else begin
                            wt_pack_lane <= wt_pack_lane + 1;
                            wt_state     <= WT_FLUSH;
                        end
                    end else begin
                        wt_state <= (wt_pack_lane != '0) ? WT_FLUSH : WT_DONE;
                    end
                end

                // --------------------------------------------------------
                WT_FLUSH: begin
                    wt_sram_wr   <= AW'(int'(wt_sram_wr) + 1);
                    wt_pack_lane <= '0;
                    wt_state     <= WT_DONE;
                end

                WT_DONE: begin
                    if (load_ready_o) wt_state <= WT_IDLE;
                end
                default: wt_state <= WT_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Combinational: DRAM read enables
    // =========================================================================

    // Activation read (skip bias column and pad)
    always_comb begin
        ext_x_rd_en_o = 1'b0;
        ext_x_addr_o  = '0;
        if (act_state == ACT_PIPE && int'(act_issue_cnt) < int'(act_total)) begin
            logic is_b;
            is_b = (USE_BIAS != 0) && (int'(act_ic_col) == REAL_M);
            if (!is_b) begin
                logic p; logic [15:0] wa; logic [1:0] bl;
                get_im2col(int'(lat_row_tile), int'(act_ic_row),
                           int'(act_ic_col), p, wa, bl);
                if (!p) begin
                    ext_x_rd_en_o = 1'b1;
                    ext_x_addr_o  = wa;
                end
            end
        end
    end

    // Weight read (skip bias tap)
    always_comb begin
        ext_y_rd_en_o = 1'b0;
        ext_y_addr_o  = '0;
        if (wt_state == WT_PIPE && int'(wt_issue_cnt) < int'(wt_total)) begin
            logic is_b;
            is_b = (USE_BIAS != 0) && (int'(wt_tap) == REAL_M);
            if (!is_b) begin
                logic [15:0] wa; logic [1:0] bl;
                get_weight_addr(int'(wt_tap), int'(lat_col_base),
                                int'(wt_col), wa, bl);
                ext_y_rd_en_o = 1'b1;
                ext_y_addr_o  = wa;
            end
        end
    end

    // Bias read (only on the bias tap when USE_BIAS=1)
    always_comb begin
        ext_b_rd_en_o = 1'b0;
        ext_b_addr_o  = '0;
        if (USE_BIAS && wt_state == WT_PIPE &&
                int'(wt_issue_cnt) < int'(wt_total) &&
                int'(wt_tap) == REAL_M) begin
            logic [15:0] bwa; logic [1:0] bbl;
            get_bias_addr(int'(lat_col_base), int'(wt_col),
                          bias_base_i, bwa, bbl);
            ext_b_rd_en_o = 1'b1;
            ext_b_addr_o  = bwa;
        end
    end

    // =========================================================================
    // Combinational: SRAM A write
    // =========================================================================
    always_comb begin
        sram_a_csb0   = 1'b1; sram_a_web0 = 1'b1;
        sram_a_wmask0 = 4'hF; sram_a_addr0 = '0; sram_a_din0 = '0;

        if (act_state == ACT_FLUSH) begin
            sram_a_csb0   = 1'b0; sram_a_web0 = 1'b0;
            sram_a_wmask0 = 4'hF; sram_a_addr0 = act_sram_wr;
            sram_a_din0   = {act_pb3, act_pb2, act_pb1, act_pb0};

        end else if ((act_state == ACT_PIPE || act_state == ACT_DRAIN) &&
                      act_s1_valid && act_pack_lane == 2'd3) begin
            automatic logic [7:0] bval_c;
            if (act_s1_is_bias)
                bval_c = 8'h01;
            else
                bval_c = act_s1_is_pad ? 8'h00
                       : ext_x_data_i[int'(act_s1_byte_lane)*8 +: 8];
            sram_a_csb0   = 1'b0; sram_a_web0 = 1'b0;
            sram_a_wmask0 = 4'hF; sram_a_addr0 = act_sram_wr;
            sram_a_din0   = {bval_c, act_pb2, act_pb1, act_pb0};
        end
    end

    // =========================================================================
    // Combinational: SRAM B write
    // =========================================================================
    always_comb begin
        sram_b_csb0   = 1'b1; sram_b_web0 = 1'b1;
        sram_b_wmask0 = {TILE_MAX{1'b1}};
        sram_b_addr0  = '0; sram_b_din0 = '0;

        if (wt_state == WT_FLUSH) begin
            sram_b_csb0   = 1'b0; sram_b_web0 = 1'b0;
            sram_b_wmask0 = {TILE_MAX{1'b1}};
            sram_b_addr0  = wt_sram_wr;
            sram_b_din0   = wt_pack_reg;

        end else if ((wt_state == WT_PIPE || wt_state == WT_DRAIN) &&
                      wt_s1_valid && int'(wt_pack_lane) == TILE_MAX - 1) begin
            automatic logic [7:0] bval_c;
            automatic logic [(TILE_MAX*8)-1:0] next_reg;
            if (wt_s1_is_bias)
                bval_c = ext_b_data_i[int'(wt_s1_bias_lane)*8 +: 8];
            else
                bval_c = ext_y_data_i[int'(wt_s1_byte_lane)*8 +: 8];
            next_reg = wt_pack_reg;
            next_reg[int'(wt_pack_lane)*8 +: 8] = bval_c;
            sram_b_csb0   = 1'b0; sram_b_web0 = 1'b0;
            sram_b_wmask0 = {TILE_MAX{1'b1}};
            sram_b_addr0  = wt_sram_wr;
            sram_b_din0   = next_reg;
        end
    end

endmodule
