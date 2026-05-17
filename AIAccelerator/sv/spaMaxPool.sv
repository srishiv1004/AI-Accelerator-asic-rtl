// =============================================================================
// spatial_maxpool.sv  ?  fully parameterised 2-D max-pool (v3, correct)
//
// Core insight for the ring buffer:
//   - When pixel (row R, col C) arrives, window[0..POOL_K-2][C%POOL_K] must
//     hold rows (R-POOL_K+1)..(R-1) from the line buffers.
//   - We write the current pixel into a "staging" slot that becomes the oldest
//     buffered row for the NEXT window, not the current one.
//   - Concretely: maintain a "read_base" pointer that lags write_slot by one
//     rotation, so reads always see fully-committed previous rows.
//
// For POOL_K=2 (N_BUF=1):
//   - One slot holds the previous row.
//   - When row R col C arrives: window[0][C%2] = line_buf[0][C] (prev row).
//   - AFTER reading, write data_i into line_buf[0][C] for next time.
//   - This read-before-write ordering is handled by doing the read in comb
//     logic directly from line_buf (which holds the OLD value from last cycle)
//     and the write in always_ff (takes effect next cycle).
// =============================================================================

module spatial_maxpool #(
    parameter int LANES  = 8,
    parameter int DATA_W = 32,
    parameter int OUT_H  = 62,
    parameter int OUT_W  = 62,
    parameter int POOL_K = 2,
    parameter int STRIDE = 2,
    parameter int PAD    = 0,

    parameter int MP_H = (OUT_H + 2*PAD - POOL_K) / STRIDE + 1,
    parameter int MP_W = (OUT_W + 2*PAD - POOL_K) / STRIDE + 1,

    parameter int N_BUF = POOL_K,   // must be >= POOL_K so write/read slots never alias
    parameter int PTR_W = (N_BUF > 1)       ? $clog2(N_BUF) : 1
)(
    input  logic clk,
    input  logic rst,

    input  logic                                 valid_i,
    input  logic [31:0]                          pixel_addr,
    input  logic signed [LANES-1:0][DATA_W-1:0] data_i,

    output logic                                 valid_o,
    output logic [31:0]                          pixel_addr_o,
    output logic signed [LANES-1:0][DATA_W-1:0] data_o
);

    // =========================================================================
    // Coordinates
    // =========================================================================
    logic [31:0] row, col;
    assign row = pixel_addr / OUT_W;
    assign col = pixel_addr % OUT_W;

    // =========================================================================
    // Line buffers: N_BUF slots, each holding one full row of pixels.
    // Written in always_ff ? the value seen combinationally is always one
    // cycle old, i.e. the previous row's pixel at this column.
    // =========================================================================
    logic signed [DATA_W-1:0] line_buf [N_BUF][OUT_W][LANES];

    // write_slot: which slot to write the CURRENT pixel into.
    // It advances (rotates) at the START of each new row so that:
    //   slot write_slot   ? current row (being written now)
    //   slot (write_slot + N_BUF - 1) % N_BUF  ? oldest row (= row - (POOL_K-1))
    //   ...
    logic [PTR_W-1:0] write_slot;

    // =========================================================================
    // Combinational: build the buffered-row view for the window.
    // line_buf[][] holds registered values from PREVIOUS cycles, so
    // line_buf[slot][col] always contains data written in an earlier cycle.
    // window_buf[r][c] = the pixel at (current_row - POOL_K + 1 + r, col c)
    //                    for r = 0 .. POOL_K-2.
    // =========================================================================
    // read slot for window row r (oldest = 0):
    //   rs(r) = (write_slot + 1 + r) % N_BUF
    // For POOL_K=2, N_BUF=1: rs = (write_slot+1+0)%1 = 0.
    // That gives line_buf[0][col], which was written by row R-1 (previous row). ?

    logic signed [DATA_W-1:0] win_buf [POOL_K-1][POOL_K][LANES]; // buffered rows
    always_comb begin
        for (int r = 0; r < POOL_K-1; r++) begin
            automatic int rs;
            rs = (int'(write_slot) + 1 + r);
            if (rs >= N_BUF) rs = rs - N_BUF;
            for (int k = 0; k < POOL_K; k++)
                for (int c = 0; c < LANES; c++)
                    // We read line_buf[rs][col - (POOL_K-1) + k] ? but we only
                    // keep POOL_K column slots in the window, so we use col%POOL_K
                    // for the current column and reconstructed for others.
                    // Simpler: window refreshes col%POOL_K each cycle so by the
                    // time window_valid fires all POOL_K col-slots are populated.
                    win_buf[r][k][c] = line_buf[rs][(col - (POOL_K-1) + k)
                                                    >= 0 ?
                                                    (col - (POOL_K-1) + k) : 0
                                                   ][c];
        end
    end

    // =========================================================================
    // Window-fire condition
    // =========================================================================
    logic window_valid;
    assign window_valid = valid_i
        && (row  >= (POOL_K - 1))
        && (col  >= (POOL_K - 1))
        && (((row - (POOL_K-1)) % STRIDE) == 0)
        && (((col - (POOL_K-1)) % STRIDE) == 0);

    // =========================================================================
    // Max reduction
    // For the live (bottom) row: pixels at col-(POOL_K-1)..col of current row.
    // We can't index line_buf for those (we're writing them this cycle or
    // they haven't arrived yet). Instead, we track the last POOL_K-1 live
    // pixels in a shift register "live_hist".
    // live_hist[0] = data_i arriving THIS cycle (col)
    // live_hist[1] = data_i from last cycle    (col-1) ? already registered
    // ...
    // live_hist[POOL_K-1] = data from POOL_K-1 cycles ago (col-(POOL_K-1)+1)
    // So the POOL_K live pixels needed are: live_hist[POOL_K-1..0] when firing.
    // =========================================================================
    logic signed [DATA_W-1:0] live_hist [POOL_K][LANES];
    // live_hist[0] = current data_i (combinational bypass)
    // live_hist[1..POOL_K-1] = registered history

    // live_reg: registered shift-register holding the previous POOL_K-1 pixels per lane.
    logic signed [DATA_W-1:0] live_reg [POOL_K-1][LANES];

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int k = 0; k < POOL_K-1; k++)
                for (int c = 0; c < LANES; c++)
                    live_reg[k][c] <= '0;
        end else if (valid_i) begin
            for (int c = 0; c < LANES; c++) begin
                live_reg[0][c] <= data_i[c];
                for (int k = 1; k < POOL_K-1; k++)
                    live_reg[k][c] <= live_reg[k-1][c];
            end
        end
    end

    // Single always_comb drives ALL of live_hist (fixes vsim-7033 multiple-driver error):
    // live_hist[0]      = current data_i (combinational bypass, this cycle)
    // live_hist[1..K-1] = live_reg[0..K-2] (registered history, previous cycles)
    always_comb begin
        for (int c = 0; c < LANES; c++)
            live_hist[0][c] = data_i[c];
        for (int k = 1; k < POOL_K; k++)
            for (int c = 0; c < LANES; c++)
                live_hist[k][c] = live_reg[k-1][c];
    end

    // Pool max: max over buffered rows and live row
    // Buffered row r, col offset k: win_buf[r][k]
    // Live row, col offset k: live_hist[POOL_K-1-k] (oldest = highest index)
    logic signed [DATA_W-1:0] pool_max [LANES];
    always_comb begin
        for (int c = 0; c < LANES; c++) begin
            automatic logic signed [DATA_W-1:0] m;
            m = live_hist[POOL_K-1][c]; // oldest live pixel
            // Rest of live row
            for (int k = 1; k < POOL_K; k++)
                if ($signed(live_hist[POOL_K-1-k][c]) > $signed(m))
                    m = live_hist[POOL_K-1-k][c];
            // Buffered rows
            for (int r = 0; r < POOL_K-1; r++)
                for (int k = 0; k < POOL_K; k++)
                    if ($signed(win_buf[r][k][c]) > $signed(m))
                        m = win_buf[r][k][c];
            pool_max[c] = m;
        end
    end

    // =========================================================================
    // Sequential: write line buffers, advance write_slot, emit output
    // =========================================================================
    logic [31:0] prev_row;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_o      <= 1'b0;
            pixel_addr_o <= '0;
            data_o       <= '0;
            write_slot   <= '0;
            prev_row     <= 32'hFFFF_FFFF;
            for (int s = 0; s < N_BUF;  s++)
                for (int w = 0; w < OUT_W; w++)
                    for (int c = 0; c < LANES; c++)
                        line_buf[s][w][c] <= '0;
        end else begin
            valid_o <= 1'b0;

            if (valid_i) begin

                // Advance write_slot when row changes.
                // IMPORTANT: the line_buf write must use the NEW slot on the
                // same cycle the row transition occurs, because write_slot is
                // a registered value and won't update until next cycle.
                // We compute eff_slot combinationally and use it for both.
                begin : row_advance
                    automatic int eff_slot;
                    eff_slot = int'(write_slot);
                    if (row != prev_row) begin
                        prev_row <= row;
                        if (N_BUF > 1) begin
                            eff_slot = int'(write_slot) + 1;
                            if (eff_slot >= N_BUF) eff_slot = 0;
                            write_slot <= PTR_W'(eff_slot);
                        end
                    end

                    // Write current pixel into the correct slot
                    for (int c = 0; c < LANES; c++)
                        line_buf[eff_slot][col][c] <= data_i[c];
                end

                // Emit when window fires
                if (window_valid) begin
                    for (int c = 0; c < LANES; c++)
                        data_o[c] <= pool_max[c];
                    pixel_addr_o <= (((row - (POOL_K-1)) / STRIDE) * MP_W)
                                  +  ((col - (POOL_K-1)) / STRIDE);
                    valid_o <= 1'b1;
                    // Debug: show window values for first 3 channels
                    $display("[%0t] SMP_WIN: r%0d c%0d -> mp_px%0d | live[0]=%0d,%0d,%0d live[1]=%0d,%0d,%0d | wbuf[0]=%0d,%0d,%0d wbuf[1]=%0d,%0d,%0d | max=%0d,%0d,%0d",
                        $time, row, col,
                        (((row-(POOL_K-1))/STRIDE)*MP_W)+((col-(POOL_K-1))/STRIDE),
                        live_hist[0][0], live_hist[0][1], live_hist[0][2],
                        live_hist[1][0], live_hist[1][1], live_hist[1][2],
                        win_buf[0][0][0], win_buf[0][0][1], win_buf[0][0][2],
                        win_buf[0][1][0], win_buf[0][1][1], win_buf[0][1][2],
                        pool_max[0], pool_max[1], pool_max[2]);
                end

            end
        end
    end

    // =========================================================================
    // Elaboration checks
    // =========================================================================
    initial begin
        assert (POOL_K >= 2)
            else $fatal(1, "spatial_maxpool: POOL_K must be >= 2");
        assert (STRIDE >= 1)
            else $fatal(1, "spatial_maxpool: STRIDE must be >= 1");
        $display("[spatial_maxpool] kernel=%0dx%0d stride=%0d in=%0dx%0d out=%0dx%0d",
                 POOL_K, POOL_K, STRIDE, OUT_H, OUT_W, MP_H, MP_W);
    end

endmodule
