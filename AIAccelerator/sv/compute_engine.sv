module compute_engine #(
    parameter int TILE_MAX  = 16,
    parameter int REAL_K    = 4,
    parameter int REAL_M    = 9,
    parameter int REAL_N    = 1,
    parameter int AW        = 8
)(
    input  logic clk,
    input  logic rst,

    output logic done,

    input  logic is_last_layer,
    input  logic act_sel,
    input  logic out_sel,
    input  logic skip_x,

    output logic        dl_start_o,
    input  logic        dl_load_ready_i,

    output logic [$clog2(REAL_K):0]        dl_row_tile_o,
    output logic [$clog2(REAL_N):0]        dl_col_tile_o,
    output logic [$clog2(TILE_MAX+1)-1:0]  dl_eff_rows_o,
    output logic [$clog2(TILE_MAX+1)-1:0]  dl_eff_cols_o,
    
    output logic [15:0]                    dl_words_a_o, 
    output logic [15:0]                    dl_words_b_o, 
    output logic [AW-1:0]                  dl_sram_a_base_o,

    input  logic         dl_sram_a_csb0, dl_sram_a_web0,
    input  logic [3:0]   dl_sram_a_wmask0,
    input  logic [AW-1:0] dl_sram_a_addr0,
    input  logic [31:0]   dl_sram_a_din0,

    input  logic         dl_sram_b_csb0, dl_sram_b_web0,
    input  logic [TILE_MAX-1:0] dl_sram_b_wmask0,
    input  logic [AW-1:0] dl_sram_b_addr0,
    input  logic [(TILE_MAX*8)-1:0] dl_sram_b_din0,

    output logic         sram_a_csb0, sram_a_web0,
    output logic [3:0]   sram_a_wmask0,
    output logic [AW-1:0] sram_a_addr0,
    output logic [31:0]   sram_a_din0,

    output logic         sram_b_csb0, sram_b_web0,
    output logic [TILE_MAX-1:0] sram_b_wmask0,
    output logic [AW-1:0] sram_b_addr0,
    output logic [(TILE_MAX*8)-1:0] sram_b_din0,

    output logic         sram_c_csb0, sram_c_web0,
    output logic [3:0]   sram_c_wmask0,
    output logic [AW-1:0] sram_c_addr0,
    output logic [31:0]   sram_c_din0,

    output logic         sram_a_csb1,
    output logic [AW-1:0] sram_a_addr1,
    input  logic [31:0]   sram_a_dout1,

    output logic         sram_b_csb1,
    output logic [AW-1:0] sram_b_addr1,
    input  logic [(TILE_MAX*8)-1:0] sram_b_dout1,

    output logic         sram_c_csb1,
    output logic [AW-1:0] sram_c_addr1,
    input  logic [31:0]   sram_c_dout1,

    output logic signed [31:0] acc_out  [TILE_MAX],
    output logic               acc_valid,
    output logic [31:0]        out_byte_addr, // RESTORED: DESCENDING PIXEL INDEX
    output logic [4:0]         out_eff_cols 
);

    localparam int MAC_LATENCY = 1;
    localparam int CAPTURE_T   = REAL_M + 2*(TILE_MAX-1) + MAC_LATENCY;

    typedef enum logic [3:0] {
        MS_RESET       = 4'd0,
        MS_LOAD_DRAM   = 4'd1,
        MS_READ_AB     = 4'd2,
        MS_RESET_ARRAY = 4'd3,
        MS_DEASSERT    = 4'd4,
        MS_RUN         = 4'd5,
        MS_CAPTURE     = 4'd6,
        MS_DRAIN       = 4'd7,
        MS_LIFO_OUT    = 4'd8,   // pop LIFO top-first ? ascending pixel order
        MS_ADVANCE     = 4'd9,
        MS_ADVANCE_CHK = 4'd10,
        MS_DONE        = 4'd11
    } ms_t;
    ms_t ms_state;

    logic [$clog2(REAL_K):0]       ti_row_tile;
    logic [$clog2(REAL_N):0]       ti_col_tile;
    logic [$clog2(TILE_MAX+1)-1:0] ti_eff_rows;
    logic [$clog2(TILE_MAX+1)-1:0] ti_eff_cols;
    logic                          ti_all_done;

    function automatic int imin(input int a, input int b);
        return (a < b) ? a : b;
    endfunction

    function automatic int words_of(input int elements);
        return (elements + 3) / 4;
    endfunction

    logic [$clog2(TILE_MAX+1)-1:0] ti_eff_rows_c, ti_eff_cols_c;
    always_comb begin
        ti_eff_rows_c = $bits(ti_eff_rows)'(imin(REAL_K - int'(ti_row_tile), TILE_MAX));
        ti_eff_cols_c = $bits(ti_eff_cols)'(imin(REAL_N - int'(ti_col_tile), TILE_MAX));
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            ti_row_tile <= '0;
            ti_col_tile <= '0;
            ti_eff_rows <= '0;
            ti_eff_cols <= '0;
            ti_all_done <= 1'b0;
        end else begin
            if (ms_state == MS_RESET) begin
                ti_row_tile <= '0;
                ti_col_tile <= '0;
                ti_eff_rows <= $bits(ti_eff_rows)'(imin(REAL_K, TILE_MAX));
                ti_eff_cols <= $bits(ti_eff_cols)'(imin(REAL_N, TILE_MAX));
                ti_all_done <= 1'b0;
            end else if (ms_state == MS_ADVANCE) begin
                if (int'(ti_col_tile) + int'(ti_eff_cols_c) < REAL_N) begin
                    ti_col_tile <= $bits(ti_col_tile)'(int'(ti_col_tile) + int'(ti_eff_cols_c));
                    ti_eff_cols <= $bits(ti_eff_cols)'(imin(REAL_N - int'(ti_col_tile) - int'(ti_eff_cols_c), TILE_MAX));
                end else if (int'(ti_row_tile) + int'(ti_eff_rows_c) < REAL_K) begin
                    ti_row_tile <= $bits(ti_row_tile)'(int'(ti_row_tile) + int'(ti_eff_rows_c));
                    ti_col_tile <= '0;
                    ti_eff_rows <= $bits(ti_eff_rows)'(imin(REAL_K - int'(ti_row_tile) - int'(ti_eff_rows_c), TILE_MAX));
                    ti_eff_cols <= $bits(ti_eff_cols)'(imin(REAL_N, TILE_MAX));
                end else begin
                    ti_all_done <= 1'b1;
                end
            end
        end
    end

    logic [$clog2(REAL_K):0]       lat_row_tile;
    logic [$clog2(REAL_N):0]       lat_col_tile;
    logic [$clog2(TILE_MAX+1)-1:0] lat_eff_rows;
    logic [$clog2(TILE_MAX+1)-1:0] lat_eff_cols;
    
    logic [15:0]                   lat_words_a; 
    logic [15:0]                   lat_words_b; 
    logic [AW-1:0]                 lat_sram_a_base;
    logic                          lat_act_sel;

    always_ff @(posedge clk) begin
        if (ms_state == MS_LOAD_DRAM) begin
            lat_row_tile    <= ti_row_tile;
            lat_col_tile    <= ti_col_tile;
            lat_eff_rows    <= ti_eff_rows;
            lat_eff_cols    <= ti_eff_cols;
            lat_words_a     <= 16'(words_of(int'(ti_eff_rows) * REAL_M));
            lat_words_b     <= 16'(words_of(int'(ti_eff_cols) * REAL_M)); 
            lat_sram_a_base <= AW'(int'(ti_row_tile) * REAL_M / 4);
            lat_act_sel     <= act_sel;
        end
    end

    assign dl_row_tile_o    = ti_row_tile;
    assign dl_col_tile_o    = ti_col_tile;
    assign dl_eff_rows_o    = ti_eff_rows;
    assign dl_eff_cols_o    = ti_eff_cols;
    assign dl_words_a_o     = 16'(words_of(int'(ti_eff_rows) * REAL_M));
    assign dl_words_b_o     = 16'(words_of(int'(ti_eff_cols) * REAL_M));
    assign dl_sram_a_base_o = AW'(int'(ti_row_tile) * REAL_M / 4);

    logic signed [7:0] x_rf [TILE_MAX][REAL_M];
    logic signed [7:0] y_rf [REAL_M][TILE_MAX];

    logic [15:0] rd_a_cnt, rd_b_cnt; 
    logic        rd_a_done, rd_b_done;
    logic        pipe_a_v, pipe_b_v;
    
    logic [31:0] pipe_a_data;
    logic [(TILE_MAX*8)-1:0] pipe_b_data; 

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            rd_a_cnt  <= '0;
            rd_b_cnt  <= '0;
            rd_a_done <= 1'b0;
            rd_b_done <= 1'b0;
            pipe_a_v  <= 1'b0;
            pipe_b_v  <= 1'b0;
        end else begin
            if (ms_state == MS_LOAD_DRAM) begin
                rd_a_cnt  <= '0;
                rd_b_cnt  <= '0;
                rd_a_done <= skip_x ? 1'b1 : 1'b0;
                rd_b_done <= 1'b0;
            end else if (ms_state == MS_READ_AB) begin
                pipe_a_v <= (rd_a_cnt < lat_words_a);
                pipe_b_v <= (rd_b_cnt < lat_words_b);

                if (rd_a_cnt < lat_words_a) begin
                    rd_a_cnt <= rd_a_cnt + 1;
                end else begin
                    rd_a_done <= 1'b1;
                end

                if (rd_b_cnt < lat_words_b) begin
                    rd_b_cnt <= rd_b_cnt + 1;
                end else begin
                    rd_b_done <= 1'b1;
                end
            end else begin
                pipe_a_v <= 1'b0;
                pipe_b_v <= 1'b0;
            end
        end
    end

    always_comb begin
        pipe_a_data = '0;
        pipe_b_data = '0;
        if (ms_state == MS_READ_AB) begin
            if (!lat_act_sel) pipe_a_data = sram_a_dout1;
            else              pipe_a_data = sram_c_dout1;
            pipe_b_data = sram_b_dout1;
        end
    end

    logic [15:0] unpack_a_idx, unpack_b_idx; 
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            unpack_a_idx <= '0;
            unpack_b_idx <= '0;
        end else begin
            if (ms_state == MS_LOAD_DRAM) begin
                unpack_a_idx <= '0;
                unpack_b_idx <= '0;
            end else if (ms_state == MS_READ_AB) begin
                if (pipe_a_v) begin
                    for (int i = 0; i < 4; i++) begin
                        automatic int tap = int'(unpack_a_idx)*4 + i;
                        automatic int row = tap / REAL_M;
                        automatic int col = tap % REAL_M;
                        if (row < TILE_MAX && col < REAL_M) begin
                            x_rf[row][col] <= pipe_a_data[i*8 +: 8];
                        end
                    end
                    unpack_a_idx <= unpack_a_idx + 1;
                end

                if (pipe_b_v) begin
                    for (int i = 0; i < TILE_MAX; i++) begin
                        automatic int tap = int'(unpack_b_idx)*TILE_MAX + i;
                        automatic int safe_cols = (int'(lat_eff_cols) == 0) ? 1 : int'(lat_eff_cols);
                        automatic int row = tap / safe_cols;
                        automatic int col = tap % safe_cols;
                        if (row < REAL_M && col < TILE_MAX) begin
                            y_rf[row][col] <= pipe_b_data[i*8 +: 8];
                        end
                    end
                    unpack_b_idx <= unpack_b_idx + 1;
                end
            end
        end
    end

    logic signed [7:0] a_feed [TILE_MAX];
    logic signed [7:0] b_feed [TILE_MAX];
    logic signed [31:0] bottom_row [TILE_MAX];

    logic               capture;
    logic               drain_en;
    logic               local_rst;
    logic [$clog2(CAPTURE_T+2)-1:0]    run_cnt;
    logic [$clog2(TILE_MAX+1)-1:0]     drain_cnt;

    // -------------------------------------------------------------------------
    // LIFO: captures rows as they drain bottom-first, then replays top-first
    // so acc_out is emitted in ascending pixel order (row 0 first).
    // Depth = TILE_MAX rows, width = TILE_MAX columns x 32 bits.
    // -------------------------------------------------------------------------
    logic signed [31:0] lifo_mem [TILE_MAX][TILE_MAX]; // [row_slot][col]
    logic [$clog2(TILE_MAX+1)-1:0] lifo_push_ptr;  // next slot to write during drain
    logic [$clog2(TILE_MAX+1)-1:0] lifo_pop_ptr;   // current slot being read during LIFO_OUT
    logic [$clog2(TILE_MAX+1)-1:0] lifo_valid_rows; // how many rows were pushed (= lat_eff_rows)

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            ms_state      <= MS_RESET;
            done          <= 1'b0;
            dl_start_o    <= 1'b0;
            local_rst     <= 1'b0;
            capture       <= 1'b0;
            run_cnt       <= '0;
            drain_cnt     <= '0;
            lifo_push_ptr  <= '0;
            lifo_pop_ptr   <= '0;
            lifo_valid_rows <= '0;
        end else begin
            dl_start_o    <= 1'b0;
            capture       <= 1'b0; 

            case (ms_state)
                MS_RESET: begin
                    done       <= 1'b0;
                    dl_start_o <= 1'b1;
                    ms_state   <= MS_LOAD_DRAM; 
                end
                MS_LOAD_DRAM: if (dl_load_ready_i) ms_state <= MS_READ_AB;
                MS_READ_AB: if (rd_a_done && rd_b_done) ms_state <= MS_RESET_ARRAY;
                MS_RESET_ARRAY: begin
                    local_rst <= 1'b1;
                    ms_state  <= MS_DEASSERT;
                end
                MS_DEASSERT: begin
                    local_rst <= 1'b0;
                    run_cnt   <= '0;
                    ms_state  <= MS_RUN;
                end
                MS_RUN: begin
                    if (int'(run_cnt) == CAPTURE_T) begin
                        capture   <= 1'b1;
                        ms_state  <= MS_CAPTURE;
                    end else begin
                        run_cnt <= run_cnt + 1;
                    end
                end
                MS_CAPTURE: begin
                    drain_cnt      <= '0;
                    lifo_push_ptr  <= '0;
                    lifo_valid_rows <= lat_eff_rows;
                    ms_state       <= MS_DRAIN;
                end
                MS_DRAIN: begin
                    // Systolic shift-register drains bottom-first.
                    // Each cycle bottom_row holds the next row; push it into LIFO.
                    // Only push valid rows (the last lat_eff_rows beats of the drain).
                    if (int'(drain_cnt) < TILE_MAX) begin
                        if (int'(drain_cnt) >= (TILE_MAX - int'(lat_eff_rows))) begin
                            for (int j = 0; j < TILE_MAX; j++)
                                lifo_mem[lifo_push_ptr][j] <= bottom_row[j];
                            lifo_push_ptr <= lifo_push_ptr + 1;
                        end
                        drain_cnt <= drain_cnt + 1;
                        if (int'(drain_cnt) == TILE_MAX - 1) begin
                            // lifo_push_ptr increments this same cycle on the last valid row,
                            // so capture the post-increment value explicitly.
                            lifo_pop_ptr <= lifo_push_ptr + 1;
                            ms_state     <= MS_LIFO_OUT;
                        end
                    end
                end
                MS_LIFO_OUT: begin
                    // Pop from [lifo_valid_rows-1] down to [0].
                    // lifo_mem[lifo_valid_rows-1] = last pushed = top row of tile = pixel lat_row_tile+0
                    // lifo_mem[0]                 = first pushed = bottom row of tile = last pixel
                    // lifo_pop_ptr starts at lifo_valid_rows, decrements each cycle.
                    // acc_out reads lifo_mem[lifo_pop_ptr-1] so first emission is [lifo_valid_rows-1].
                    if (int'(lifo_pop_ptr) > 0) begin
                        lifo_pop_ptr <= lifo_pop_ptr - 1;
                        if (int'(lifo_pop_ptr) == 1)
                            ms_state <= MS_ADVANCE;
                    end
                end
                MS_ADVANCE: ms_state <= MS_ADVANCE_CHK;
                MS_ADVANCE_CHK: begin
                    if (ti_all_done) ms_state <= MS_DONE;
                    else begin
                        dl_start_o <= 1'b1;
                        ms_state   <= MS_LOAD_DRAM;
                    end
                end
                MS_DONE: done <= 1'b1;
                default: ms_state <= MS_RESET;
            endcase
        end
    end

    always_comb begin
        drain_en      = 1'b0;
        acc_valid     = 1'b0;
        out_byte_addr = '0;
        out_eff_cols  = '0;

        // Default: pass bottom_row through (used during drain push phase)
        for (int j = 0; j < TILE_MAX; j++)
            acc_out[j] = bottom_row[j];

        // Drain enable: active while systolic array is shifting out
        if (ms_state == MS_DRAIN)
            drain_en = 1'b1;

        // LIFO output: emit rows in ascending pixel order (row 0 first)
        // lifo_pop_ptr counts from lifo_valid_rows down to 1;
        // the row being read this cycle is [lifo_pop_ptr - 1].
        // lifo_mem[lifo_valid_rows-1] = top row of tile = pixel lat_row_tile+0 (first)
        // lifo_mem[0]                 = bottom row of tile = pixel lat_row_tile+lat_eff_rows-1 (last)
        if (ms_state == MS_LIFO_OUT && int'(lifo_pop_ptr) > 0) begin
            acc_valid     = 1'b1;
            // pixel index = lat_row_tile + (lifo_valid_rows - lifo_pop_ptr)
            // when lifo_pop_ptr = lifo_valid_rows (first pop): pixel = lat_row_tile + 0
            // when lifo_pop_ptr = 1 (last pop):                pixel = lat_row_tile + lifo_valid_rows - 1
            out_byte_addr = 32'(lat_row_tile)
                          + 32'(lifo_valid_rows)
                          - 32'(lifo_pop_ptr);
            out_eff_cols  = 5'(lat_eff_cols);
            for (int j = 0; j < TILE_MAX; j++)
                acc_out[j] = lifo_mem[lifo_pop_ptr - 1][j];
        end
    end

    // -------------------------------------------------------------------------
    // DEBUG: trace every LIFO push and pop so we can verify ordering
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        // Push trace: show which lifo slot gets which drain beat
        if (ms_state == MS_DRAIN &&
            int'(drain_cnt) >= (TILE_MAX - int'(lat_eff_rows)) &&
            int'(drain_cnt) < TILE_MAX) begin
            $display("[CE_LIFO_PUSH] drain_cnt=%0d -> lifo_mem[%0d] ch0=%0d ch1=%0d ch2=%0d",
                     drain_cnt, lifo_push_ptr,
                     bottom_row[0], bottom_row[1], bottom_row[2]);
        end
        // Pop trace: show addr and data emitted to gearbox
        if (ms_state == MS_LIFO_OUT && int'(lifo_pop_ptr) > 0) begin
            $display("[CE_LIFO_POP]  pop_ptr=%0d -> addr=%0d ch0=%0d ch1=%0d ch2=%0d",
                     lifo_pop_ptr,
                     32'(lat_row_tile) + 32'(lifo_valid_rows) - 32'(lifo_pop_ptr),
                     lifo_mem[lifo_pop_ptr-1][0],
                     lifo_mem[lifo_pop_ptr-1][1],
                     lifo_mem[lifo_pop_ptr-1][2]);
        end
    end

    systolic_tiled #(.TILE_MAX(TILE_MAX)) u_systolic (
        .clk(clk),
        .rst(local_rst),
        .capture(capture),
        .drain_en(drain_en),
        .a(a_feed),
        .b(b_feed),
        .bottom_row(bottom_row)
    );

    always_comb begin
        for (int k = 0; k < TILE_MAX; k++) begin
            automatic int  t_minus_k = int'(run_cnt) - k;
            automatic logic active   = (ms_state == MS_RUN || ms_state == MS_DEASSERT);
            automatic logic in_range = (t_minus_k >= 0 && t_minus_k < REAL_M);
            
            a_feed[k] = (active && k < int'(lat_eff_cols) && in_range) ? y_rf[t_minus_k][k] : 8'sd0;
            b_feed[k] = (active && k < int'(lat_eff_rows) && in_range) ? x_rf[k][t_minus_k] : 8'sd0;
        end
    end

    always_comb begin
        sram_a_csb1 = 1'b1; sram_a_addr1 = '0;
        sram_b_csb1 = 1'b1; sram_b_addr1 = '0;
        sram_c_csb1 = 1'b1; sram_c_addr1 = '0;

        if (ms_state == MS_READ_AB) begin
            if (rd_a_cnt < lat_words_a) begin
                if (!lat_act_sel) begin
                    sram_a_csb1  = 1'b0;
                    sram_a_addr1 = AW'(lat_sram_a_base + rd_a_cnt);
                end else begin
                    sram_c_csb1  = 1'b0;
                    sram_c_addr1 = AW'(lat_sram_a_base + rd_a_cnt);
                end
            end
            if (rd_b_cnt < lat_words_b) begin
                sram_b_csb1  = 1'b0;
                sram_b_addr1 = AW'(rd_b_cnt);
            end
        end
    end

    always_comb begin
        sram_a_csb0   = 1'b1; sram_a_web0 = 1'b1; sram_a_wmask0 = 4'hF; sram_a_addr0 = '0; sram_a_din0 = '0;
        sram_b_csb0   = 1'b1; sram_b_web0 = 1'b1; sram_b_wmask0 = {TILE_MAX{1'b1}}; sram_b_addr0 = '0; sram_b_din0 = '0;
        sram_c_csb0   = 1'b1; sram_c_web0 = 1'b1; sram_c_wmask0 = 4'hF; sram_c_addr0 = '0; sram_c_din0 = '0;

        if (ms_state == MS_LOAD_DRAM) begin
            sram_a_csb0   = dl_sram_a_csb0; sram_a_web0   = dl_sram_a_web0;
            sram_a_wmask0 = dl_sram_a_wmask0; sram_a_addr0  = dl_sram_a_addr0; sram_a_din0   = dl_sram_a_din0;
            
            sram_b_csb0   = dl_sram_b_csb0; sram_b_web0   = dl_sram_b_web0;
            sram_b_wmask0 = dl_sram_b_wmask0; sram_b_addr0  = dl_sram_b_addr0; sram_b_din0   = dl_sram_b_din0;
        end 
    end
endmodule
