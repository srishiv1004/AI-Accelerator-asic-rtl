module systolic_pipe_conv #(
    parameter int TILE_MAX  = 16,
    parameter int IN_H      = 64,
    parameter int IN_W      = 64,
    parameter int IN_C      = 3,
    parameter int OUT_C     = 3,
    parameter int K_H       = 3,
    parameter int K_W       = 3,
    parameter int STRIDE    = 1,
    parameter int PAD       = 0,
    parameter int AW        = 8,
    parameter int USE_BIAS  = 1,    // 1 = append bias column; 0 = legacy

    // Derived - do not override
    parameter int OUT_H      = (IN_H + 2*PAD - K_H) / STRIDE + 1,
    parameter int OUT_W      = (IN_W + 2*PAD - K_W) / STRIDE + 1,
    parameter int REAL_K     = OUT_H * OUT_W,
    parameter int REAL_M     = K_H * K_W * IN_C,       // WITHOUT bias
    parameter int REAL_M_EFF = REAL_M + USE_BIAS,       // WITH bias -> passed to compute_engine
    parameter int REAL_N     = OUT_C
)(
    input  logic clk,
    input  logic rst,
    output logic done,

    input  logic is_last_layer,
    input  logic act_sel,
    input  logic out_sel,
    input  logic skip_x,

    // ---- activation DRAM ----
    output logic [15:0] ext_x_addr_o,
    output logic        ext_x_rd_en_o,
    input  logic [31:0] ext_x_data_i,

    // ---- weight DRAM ----
    output logic [15:0] ext_y_addr_o,
    output logic        ext_y_rd_en_o,
    input  logic [31:0] ext_y_data_i,

    // ---- bias DRAM (tie to 0 / unused when USE_BIAS=0) ----
    // bias[oc] stored as flat INT8 array, 4-per-word:
    //   word address = bias_base_i + oc/4,  byte lane = oc%4
    output logic [15:0] ext_b_addr_o,
    output logic        ext_b_rd_en_o,
    input  logic [31:0] ext_b_data_i,
    input  logic [15:0] bias_base_i,    // word address of bias[0] in DRAM

    // ---- output DRAM ----
    output logic [15:0] ext_z_addr_o,
    output logic        ext_z_wr_en_o,
    output logic [31:0] ext_z_data_o,
    output logic [3:0]  ext_z_wmask_o,

    // ---- SRAM A ----
    output logic          sram_a_csb0, sram_a_web0,
    output logic [3:0]    sram_a_wmask0,
    output logic [AW-1:0] sram_a_addr0,
    output logic [31:0]   sram_a_din0,
    input  logic [31:0]   sram_a_dout0,
    output logic          sram_a_csb1,
    output logic [AW-1:0] sram_a_addr1,
    input  logic [31:0]   sram_a_dout1,

    // ---- SRAM B ----
    output logic                    sram_b_csb0, sram_b_web0,
    output logic [TILE_MAX-1:0]     sram_b_wmask0,
    output logic [AW-1:0]           sram_b_addr0,
    output logic [(TILE_MAX*8)-1:0] sram_b_din0,
    input  logic [(TILE_MAX*8)-1:0] sram_b_dout0,
    output logic          sram_b_csb1,
    output logic [AW-1:0] sram_b_addr1,
    input  logic [(TILE_MAX*8)-1:0] sram_b_dout1,

    // ---- SRAM C ----
    output logic          sram_c_csb0, sram_c_web0,
    output logic [3:0]    sram_c_wmask0,
    output logic [AW-1:0] sram_c_addr0,
    output logic [31:0]   sram_c_din0,
    input  logic [31:0]   sram_c_dout0,
    output logic          sram_c_csb1,
    output logic [AW-1:0] sram_c_addr1,
    input  logic [31:0]   sram_c_dout1,

    // ---- systolic outputs ----
    output logic signed [31:0] acc_out [TILE_MAX],
    output logic               acc_valid,
    output logic [31:0]        out_byte_addr,
    output logic [4:0]         out_eff_cols
);

    // -------------------------------------------------------------------------
    // Internal wires between dram_loader and compute_engine
    // -------------------------------------------------------------------------
    logic        dl_start, dl_load_ready;
    logic [$clog2(REAL_K):0]        dl_row_tile;
    logic [$clog2(REAL_N):0]        dl_col_tile;
    logic [$clog2(TILE_MAX+1)-1:0]  dl_eff_rows, dl_eff_cols;
    logic [15:0]   dl_words_a, dl_words_b;
    logic [AW-1:0] dl_sram_a_base;

    logic          dl_sram_a_csb0, dl_sram_a_web0;
    logic [3:0]    dl_sram_a_wmask0;
    logic [AW-1:0] dl_sram_a_addr0;
    logic [31:0]   dl_sram_a_din0;

    logic          dl_sram_b_csb0, dl_sram_b_web0;
    logic [TILE_MAX-1:0]     dl_sram_b_wmask0;
    logic [AW-1:0]           dl_sram_b_addr0;
    logic [(TILE_MAX*8)-1:0] dl_sram_b_din0;

    // -------------------------------------------------------------------------
    // dram_loader
    // Pass REAL_M (without bias); loader uses USE_BIAS internally to add +1 tap.
    // -------------------------------------------------------------------------
    dram_loader #(
        .TILE_MAX  (TILE_MAX),
        .REAL_K    (REAL_K),
        .REAL_M    (REAL_M),    // K_H*K_W*IN_C, loader appends bias internally
        .REAL_N    (REAL_N),
        .IN_H      (IN_H),
        .IN_W      (IN_W),
        .IN_C      (IN_C),
        .K_H       (K_H),
        .K_W       (K_W),
        .STRIDE    (STRIDE),
        .PAD       (PAD),
        .AW        (AW),
        .USE_BIAS  (USE_BIAS)
    ) u_dram_loader (
        .clk           (clk),
        .rst           (rst),
        .start_i       (dl_start),
        .skip_x_i      (skip_x),
        .load_ready_o  (dl_load_ready),
        .busy_o        (),
        .row_tile_i    (dl_row_tile),
        .col_tile_i    (dl_col_tile),
        .eff_rows_i    (dl_eff_rows),
        .eff_cols_i    (dl_eff_cols),
        .words_a_i     (dl_words_a),
        .words_b_i     (dl_words_b),
        .sram_a_base_i (dl_sram_a_base),
        // activation DRAM
        .ext_x_addr_o  (ext_x_addr_o),
        .ext_x_rd_en_o (ext_x_rd_en_o),
        .ext_x_data_i  (ext_x_data_i),
        // weight DRAM
        .ext_y_addr_o  (ext_y_addr_o),
        .ext_y_rd_en_o (ext_y_rd_en_o),
        .ext_y_data_i  (ext_y_data_i),
        // bias DRAM
        .ext_b_addr_o  (ext_b_addr_o),
        .ext_b_rd_en_o (ext_b_rd_en_o),
        .ext_b_data_i  (ext_b_data_i),
        .bias_base_i   (bias_base_i),
        // SRAM A
        .sram_a_csb0   (dl_sram_a_csb0),
        .sram_a_web0   (dl_sram_a_web0),
        .sram_a_wmask0 (dl_sram_a_wmask0),
        .sram_a_addr0  (dl_sram_a_addr0),
        .sram_a_din0   (dl_sram_a_din0),
        // SRAM B
        .sram_b_csb0   (dl_sram_b_csb0),
        .sram_b_web0   (dl_sram_b_web0),
        .sram_b_wmask0 (dl_sram_b_wmask0),
        .sram_b_addr0  (dl_sram_b_addr0),
        .sram_b_din0   (dl_sram_b_din0)
    );

    // -------------------------------------------------------------------------
    // compute_engine
    // REAL_M passed here = REAL_M_EFF = K_H*K_W*IN_C + USE_BIAS
    // so the systolic array accounts for the extra bias tap.
    // -------------------------------------------------------------------------
    compute_engine #(
        .TILE_MAX (TILE_MAX),
        .REAL_K   (REAL_K),
        .REAL_M   (REAL_M_EFF),     // <-- KEY: +1 for bias column
        .REAL_N   (REAL_N),
        .AW       (AW)
    ) u_compute_engine (
        .clk             (clk),
        .rst             (rst),
        .done            (done),
        .is_last_layer   (is_last_layer),
        .act_sel         (act_sel),
        .out_sel         (out_sel),
        .skip_x          (skip_x),

        .dl_start_o      (dl_start),
        .dl_load_ready_i (dl_load_ready),
        .dl_row_tile_o   (dl_row_tile),
        .dl_col_tile_o   (dl_col_tile),
        .dl_eff_rows_o   (dl_eff_rows),
        .dl_eff_cols_o   (dl_eff_cols),
        .dl_words_a_o    (dl_words_a),
        .dl_words_b_o    (dl_words_b),
        .dl_sram_a_base_o(dl_sram_a_base),

        .dl_sram_a_csb0  (dl_sram_a_csb0),
        .dl_sram_a_web0  (dl_sram_a_web0),
        .dl_sram_a_wmask0(dl_sram_a_wmask0),
        .dl_sram_a_addr0 (dl_sram_a_addr0),
        .dl_sram_a_din0  (dl_sram_a_din0),
        .dl_sram_b_csb0  (dl_sram_b_csb0),
        .dl_sram_b_web0  (dl_sram_b_web0),
        .dl_sram_b_wmask0(dl_sram_b_wmask0),
        .dl_sram_b_addr0 (dl_sram_b_addr0),
        .dl_sram_b_din0  (dl_sram_b_din0),

        .sram_a_csb0     (sram_a_csb0),
        .sram_a_web0     (sram_a_web0),
        .sram_a_wmask0   (sram_a_wmask0),
        .sram_a_addr0    (sram_a_addr0),
        .sram_a_din0     (sram_a_din0),
        .sram_b_csb0     (sram_b_csb0),
        .sram_b_web0     (sram_b_web0),
        .sram_b_wmask0   (sram_b_wmask0),
        .sram_b_addr0    (sram_b_addr0),
        .sram_b_din0     (sram_b_din0),
        .sram_c_csb0     (sram_c_csb0),
        .sram_c_web0     (sram_c_web0),
        .sram_c_wmask0   (sram_c_wmask0),
        .sram_c_addr0    (sram_c_addr0),
        .sram_c_din0     (sram_c_din0),

        .sram_a_csb1     (sram_a_csb1),
        .sram_a_addr1    (sram_a_addr1),
        .sram_a_dout1    (sram_a_dout1),
        .sram_b_csb1     (sram_b_csb1),
        .sram_b_addr1    (sram_b_addr1),
        .sram_b_dout1    (sram_b_dout1),
        .sram_c_csb1     (sram_c_csb1),
        .sram_c_addr1    (sram_c_addr1),
        .sram_c_dout1    (sram_c_dout1),

        .acc_out         (acc_out),
        .acc_valid       (acc_valid),
        .out_byte_addr   (out_byte_addr),
        .out_eff_cols    (out_eff_cols)
    );

endmodule