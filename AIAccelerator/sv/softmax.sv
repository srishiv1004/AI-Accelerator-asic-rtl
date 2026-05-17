// =============================================================================
//  softmax_pkg  shared parameters and exp LUT
//
//  CHANGES vs -11 version:
//  1. LUT extended from k=0..11 to k=0..15
//  2. CLAMP_NEG updated to -15 in exp_approx_lane
//  3. k=12..15 yield 0 in Q8.16 (e^-12 = 6.1e-6 * 65536 = 0.4 ? 0)
//     These are kept for interface completeness; they correctly contribute
//     0 to the softmax sum (negligible weight, mathematically sound).
//  4. nr_seed_lut and NR reciprocal logic unchanged
// =============================================================================
package softmax_pkg;

    parameter int N_LANES    = 4;
    parameter int DATA_W     = 32;   // 32-bit signed fixed-point input
    parameter int FRAC_W     = 8;    // fractional bits in input
    parameter int EXP_W      = 24;   // exp output width
    parameter int EXP_FRAC_W = 16;   // fractional bits in exp output

    // -------------------------------------------------------------------------
    //  e^-k in Q8.16 format (value * 2^16), k = 0..15
    //
    //  Exact float values and Q8.16 encodings:
    //    k=0:  e^0   = 1.0          * 65536 = 65536   = 0x010000
    //    k=1:  e^-1  = 0.367879441  * 65536 = 24109.2 = 0x005E2D
    //    k=2:  e^-2  = 0.135335283  * 65536 = 8872.0  = 0x0022A5 (8869 is correct round)
    //    k=3:  e^-3  = 0.049787068  * 65536 = 3262.4  = 0x000CBE
    //    k=4:  e^-4  = 0.018315639  * 65536 = 1200.1  = 0x0004B0
    //    k=5:  e^-5  = 0.006737947  * 65536 = 441.6   = 0x0001BA  (corrected)
    //    k=6:  e^-6  = 0.002478752  * 65536 = 162.4   = 0x0000A2
    //    k=7:  e^-7  = 0.000911882  * 65536 = 59.77   = 0x00003C  (corrected)
    //    k=8:  e^-8  = 0.000335463  * 65536 = 21.99   = 0x000016
    //    k=9:  e^-9  = 0.000123410  * 65536 = 8.09    = 0x000008
    //    k=10: e^-10 = 0.000045400  * 65536 = 2.975   = 0x000003
    //    k=11: e^-11 = 0.000016702  * 65536 = 1.094   = 0x000001
    //    k=12: e^-12 = 0.000006144  * 65536 = 0.403   = 0x000000  (underflows)
    //    k=13: e^-13 = 0.000002260  * 65536 = 0.148   = 0x000000  (underflows)
    //    k=14: e^-14 = 0.000000831  * 65536 = 0.054   = 0x000000  (underflows)
    //    k=15: e^-15 = 0.000000306  * 65536 = 0.020   = 0x000000  (underflows)
    //
    //  k=12..15 are zero in Q8.16. They are valid inputs and correctly produce
    //  zero exp output, contributing nothing to the softmax sum.
    //  To represent them non-zero you would need EXP_FRAC_W >= 21.
    // -------------------------------------------------------------------------
    function automatic logic [EXP_W-1:0] exp_lut(input int k);
        logic [EXP_W-1:0] LUT [0:15];
        LUT[0]  = 24'h010000; // e^0   = 1.0          * 2^16 = 65536
        LUT[1]  = 24'h005E2D; // e^-1  = 0.36788      * 2^16 = 24109
        LUT[2]  = 24'h0022A5; // e^-2  = 0.13534      * 2^16 = 8869
        LUT[3]  = 24'h000CBE; // e^-3  = 0.049787     * 2^16 = 3262
        LUT[4]  = 24'h0004B0; // e^-4  = 0.018316     * 2^16 = 1200
        LUT[5]  = 24'h0001BA; // e^-5  = 0.0067379    * 2^16 = 442
        LUT[6]  = 24'h0000A2; // e^-6  = 0.0024788    * 2^16 = 162
        LUT[7]  = 24'h00003C; // e^-7  = 0.00091188   * 2^16 = 60
        LUT[8]  = 24'h000016; // e^-8  = 0.00033546   * 2^16 = 22
        LUT[9]  = 24'h000008; // e^-9  = 0.00012341   * 2^16 = 8
        LUT[10] = 24'h000003; // e^-10 = 0.000045400  * 2^16 = 3
        LUT[11] = 24'h000001; // e^-11 = 0.000016702  * 2^16 = 1
        return (k < 12) ? LUT[k] : 24'h000000;
    endfunction

    // ---------------------------------------------------------------------------
    //  NR seed LUT  1/esum approximation seeded from leading bits
    // ---------------------------------------------------------------------------
    parameter int SW = EXP_W + 2; // sum width = 26 bits

    function automatic logic [EXP_W-1:0] nr_seed_lut(input logic [SW-1:0] esum);
        logic [EXP_W-1:0] MLUT [0:15];
        int lz, shift;
        logic [SW-1:0] norm;
        logic [3:0]    mant;

        MLUT[0]  = 24'h010000; // 1/1.000 = 1.0000
        MLUT[1]  = 24'h00F0F1; // 1/1.0625= 0.9412
        MLUT[2]  = 24'h00E38F; // 1/1.125 = 0.8889
        MLUT[3]  = 24'h00D794; // 1/1.1875= 0.8421
        MLUT[4]  = 24'h00CCCD; // 1/1.250 = 0.8000
        MLUT[5]  = 24'h00C31C; // 1/1.3125= 0.7619
        MLUT[6]  = 24'h00BA2F; // 1/1.375 = 0.7273
        MLUT[7]  = 24'h00B216; // 1/1.4375= 0.6957
        MLUT[8]  = 24'h00AAAB; // 1/1.500 = 0.6667
        MLUT[9]  = 24'h00A3D7; // 1/1.5625= 0.6400
        MLUT[10] = 24'h009D8A; // 1/1.625 = 0.6154
        MLUT[11] = 24'h009798; // 1/1.6875= 0.5926
        MLUT[12] = 24'h009249; // 1/1.750 = 0.5714
        MLUT[13] = 24'h008D3E; // 1/1.8125= 0.5517
        MLUT[14] = 24'h008889; // 1/1.875 = 0.5333
        MLUT[15] = 24'h008421; // 1/1.9375= 0.5161

        if (esum == 0) return {EXP_W{1'b1}};

        lz = 0;
        for (int i = SW-1; i >= 0; i--) begin
            if (esum[i]) break;
            lz++;
        end

        norm = esum << lz;
        mant = norm[SW-2 -: 4];
        shift = (SW - 1 - EXP_FRAC_W) - lz;

        if (shift >= 0)
            return EXP_W'(MLUT[mant] >> shift);
        else
            return EXP_W'(MLUT[mant] << (-shift));

    endfunction

endpackage : softmax_pkg


// =============================================================================
//  exp_approx_lane  single lane 3-cycle pipelined exp approximation
//
//  CHANGE: CLAMP_NEG updated to -15 (from -11)
//          k range in exp_lut now covers 0..15
// =============================================================================
module exp_approx_lane
    import softmax_pkg::*;
#(parameter int DW=DATA_W, parameter int FW=FRAC_W,
  parameter int OW=EXP_W,  parameter int OFW=EXP_FRAC_W)
(
    input  logic                  clk, rst_n, valid_i,
    input  logic signed [DW-1:0]  x_i,
    output logic                  valid_o,
    output logic [OW-1:0]         exp_o
);
    // Clamp range extended to -15. k=12..15 yield 0 from LUT (Q8.16 underflow).
    localparam logic signed [DW-1:0] CLAMP_NEG = DW'(-15*(1<<FW));
    localparam logic signed [DW-1:0] ZERO      = '0;
    localparam logic [FW-1:0]        HALF      = FW'(1<<(FW-1));
    localparam logic [FW-1:0]        SIXTH     = FW'((1<<FW)/6);
    localparam logic [FW-1:0]        T24       = FW'((1<<FW)/24);
    localparam logic [FW-1:0]        FW_MASK   = {FW{1'b1}};

    // -------------------------------------------------------------------------
    //  Stage 1: clamp x to [-15, 0]
    // -------------------------------------------------------------------------
    logic signed [DW-1:0] xc;
    logic                 vc1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vc1 <= 1'b0;
            xc  <= '0;
        end else begin
            vc1 <= valid_i;
            if      (x_i < CLAMP_NEG) xc <= CLAMP_NEG;
            else if (x_i > ZERO)      xc <= ZERO;
            else                       xc <= x_i;
        end
    end

    // -------------------------------------------------------------------------
    //  Combinational: split into integer k and fractional frac in [0,1)
    //  k = ceil(-xc), frac = k + xc  (both non-negative since xc <= 0)
    // -------------------------------------------------------------------------
    logic signed [DW-1:0] floor_x;
    logic [3:0]            k;       // now needs 4 bits to hold 0..15
    logic [FW-1:0]         frac;

    assign k       = 4'(($unsigned(-xc) + FW_MASK) >> FW);
    assign floor_x = -(DW'(k) << FW);
    assign frac    = FW'($unsigned(xc - floor_x));

    // -------------------------------------------------------------------------
    //  Taylor series: e^frac ? 1 + f + f²/2 + f³/6 + f?/24
    // -------------------------------------------------------------------------
    logic [2*FW-1:0]  f2, f3, f4;
    logic [2*FW+1:0]  t1, t2, t3, t4;

    assign f2 = frac * frac;
    assign f3 = f2[2*FW-1:FW] * frac;
    assign f4 = f3[2*FW-1:FW] * frac;

    assign t1 = {{2{1'b0}}, frac, {FW{1'b0}}};
    assign t2 = {2'b00, (2*FW)'(f2[2*FW-1:FW]) * (2*FW)'(HALF)};
    assign t3 = {2'b00, (2*FW)'(f3[2*FW-1:FW]) * (2*FW)'(SIXTH)};
    assign t4 = {2'b00, (2*FW)'(f4[2*FW-1:FW]) * (2*FW)'(T24)};

    // -------------------------------------------------------------------------
    //  Stage 2: register LUT base and polynomial sum
    // -------------------------------------------------------------------------
    logic [OW-1:0]     base;
    logic [2*FW+3:0]   poly;
    logic              vc2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vc2  <= 1'b0;
            base <= '0;
            poly <= '0;
        end else begin
            vc2  <= vc1;
            base <= exp_lut(int'(k));
            poly <= (2*FW+4)'(1 << (2*FW))
                  + (2*FW+4)'(t1)
                  + (2*FW+4)'(t2)
                  + (2*FW+4)'(t3)
                  + (2*FW+4)'(t4);
        end
    end

    // -------------------------------------------------------------------------
    //  Stage 3: multiply base * poly, shift back to Q8.16
    // -------------------------------------------------------------------------
    logic [OW+2*FW+3:0] prod;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_o <= 1'b0;
            exp_o   <= '0;
        end else begin
            valid_o <= vc2;
            prod     = (OW+2*FW+4)'(base) * (OW+2*FW+4)'(poly);
            exp_o   <= OW'(prod >> (2*FW));
        end
    end

endmodule : exp_approx_lane


// =============================================================================
//  simd_softmax  4-lane softmax, fully pipelined  (unchanged from -11 version)
// =============================================================================
module simd_softmax
    import softmax_pkg::*;
#(parameter int N  = N_LANES,
  parameter int DW = DATA_W,
  parameter int FW = FRAC_W,
  parameter int OW = EXP_W,
  parameter int OFW= EXP_FRAC_W)
(
    input  logic                  clk, rst_n, valid_i,
    input  logic signed [DW-1:0]  i0, i1, i2, i3,
    output logic                  valid_o,
    output logic [OW-1:0]         p0, p1, p2, p3
);

    // =========================================================================
    //  STAGE A: Max reduction - 1 cycle
    // =========================================================================
    logic signed [DW-1:0] m1a, m1b;
    logic signed [DW-1:0] xmax;
    logic                 va;

    assign m1a = (i0 > i1) ? i0 : i1;
    assign m1b = (i2 > i3) ? i2 : i3;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin va <= 1'b0; xmax <= '0; end
        else         begin va <= valid_i; xmax <= (m1a > m1b) ? m1a : m1b; end
    end

    // =========================================================================
    //  Input delay: 1 stage
    // =========================================================================
    logic signed [DW-1:0] d1_0, d1_1, d1_2, d1_3;
    logic                 vd1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vd1  <= 1'b0;
            d1_0 <= '0; d1_1 <= '0; d1_2 <= '0; d1_3 <= '0;
        end else begin
            vd1  <= valid_i;
            d1_0 <= i0; d1_1 <= i1; d1_2 <= i2; d1_3 <= i3;
        end
    end

    // =========================================================================
    //  STAGE B: Subtract max
    // =========================================================================
    logic signed [DW-1:0] xs0, xs1, xs2, xs3;
    logic                 vb;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vb  <= 1'b0;
            xs0 <= '0; xs1 <= '0; xs2 <= '0; xs3 <= '0;
        end else begin
            vb  <= va;
            xs0 <= d1_0 - xmax;
            xs1 <= d1_1 - xmax;
            xs2 <= d1_2 - xmax;
            xs3 <= d1_3 - xmax;
        end
    end

    // =========================================================================
    //  STAGE C: 4 parallel exp lanes (3 cycles)
    // =========================================================================
    logic [OW-1:0] ex0, ex1, ex2, ex3;
    logic          ev0, ev1, ev2, ev3;

    exp_approx_lane #(.DW(DW),.FW(FW),.OW(OW),.OFW(OFW))
        u0 (.clk(clk),.rst_n(rst_n),.valid_i(vb),.x_i(xs0),.valid_o(ev0),.exp_o(ex0));
    exp_approx_lane #(.DW(DW),.FW(FW),.OW(OW),.OFW(OFW))
        u1 (.clk(clk),.rst_n(rst_n),.valid_i(vb),.x_i(xs1),.valid_o(ev1),.exp_o(ex1));
    exp_approx_lane #(.DW(DW),.FW(FW),.OW(OW),.OFW(OFW))
        u2 (.clk(clk),.rst_n(rst_n),.valid_i(vb),.x_i(xs2),.valid_o(ev2),.exp_o(ex2));
    exp_approx_lane #(.DW(DW),.FW(FW),.OW(OW),.OFW(OFW))
        u3 (.clk(clk),.rst_n(rst_n),.valid_i(vb),.x_i(xs3),.valid_o(ev3),.exp_o(ex3));

    logic ec; assign ec = ev0;

    // =========================================================================
    //  STAGE D: Sum reduction - 2 cycles
    // =========================================================================
    localparam int SW = OW + 2;

    logic [SW-1:0] s1a, s1b, esum;
    logic          vd1s, vd2s;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin vd1s <= 1'b0; s1a <= '0; s1b <= '0; end
        else begin
            vd1s <= ec;
            s1a  <= SW'(ex0) + SW'(ex1);
            s1b  <= SW'(ex2) + SW'(ex3);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin vd2s <= 1'b0; esum <= '0; end
        else         begin vd2s <= vd1s; esum <= s1a + s1b; end
    end

    // =========================================================================
    //  STAGE E: Newton-Raphson reciprocal - 3 cycles
    // =========================================================================
    logic [OW-1:0]    nr_seed;
    logic [OW-1:0]    nr_recip1, nr_recip2;
    logic [SW-1:0]    nr_esum1, nr_esum2;
    logic [2*OW-1:0]  nr_sp1, nr_sp2;
    logic [OW-1:0]    nr_2m1, nr_2m2;
    logic             nr_e1, nr_e2, nr_e3;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nr_e1   <= 1'b0;
            nr_seed <= '0;
            nr_esum1<= '0;
        end else begin
            nr_e1    <= vd2s;
            nr_seed  <= nr_seed_lut(esum);
            nr_esum1 <= esum;
        end
    end

    assign nr_sp1 = (2*OW)'(nr_esum1) * (2*OW)'(nr_seed);
    assign nr_2m1 = OW'(2 << OFW) - OW'(nr_sp1 >> OFW);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nr_e2    <= 1'b0;
            nr_recip1<= '0;
            nr_esum2 <= '0;
        end else begin
            nr_e2     <= nr_e1;
            nr_recip1 <= OW'(((2*OW)'(nr_seed) * (2*OW)'(nr_2m1)) >> OFW);
            nr_esum2  <= nr_esum1;
        end
    end

    assign nr_sp2 = (2*OW)'(nr_esum2) * (2*OW)'(nr_recip1);
    assign nr_2m2 = OW'(2 << OFW) - OW'(nr_sp2 >> OFW);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            nr_e3    <= 1'b0;
            nr_recip2<= '0;
        end else begin
            nr_e3     <= nr_e2;
            nr_recip2 <= OW'(((2*OW)'(nr_recip1) * (2*OW)'(nr_2m2)) >> OFW);
        end
    end

    // =========================================================================
    //  Exp delay chain: align ex0-ex3 with nr_recip2 (5-cycle delay)
    // =========================================================================
    logic [OW-1:0] ed_0 [0:4];
    logic [OW-1:0] ed_1 [0:4];
    logic [OW-1:0] ed_2 [0:4];
    logic [OW-1:0] ed_3 [0:4];
    logic          edv   [0:4];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 5; i++) begin
                edv[i]  <= 1'b0;
                ed_0[i] <= '0; ed_1[i] <= '0;
                ed_2[i] <= '0; ed_3[i] <= '0;
            end
        end else begin
            edv[0]  <= ec;   ed_0[0] <= ex0; ed_1[0] <= ex1;
                             ed_2[0] <= ex2; ed_3[0] <= ex3;
            for (int i = 1; i < 5; i++) begin
                edv[i]  <= edv[i-1];
                ed_0[i] <= ed_0[i-1]; ed_1[i] <= ed_1[i-1];
                ed_2[i] <= ed_2[i-1]; ed_3[i] <= ed_3[i-1];
            end
        end
    end

    // =========================================================================
    //  STAGE F: Scale
    // =========================================================================
    logic [2*OW-1:0] sc0, sc1, sc2, sc3;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_o <= 1'b0;
            p0 <= '0; p1 <= '0; p2 <= '0; p3 <= '0;
        end else begin
            valid_o <= nr_e3;
            sc0 = (2*OW)'(ed_0[4]) * (2*OW)'(nr_recip2); p0 <= OW'(sc0 >> OFW);
            sc1 = (2*OW)'(ed_1[4]) * (2*OW)'(nr_recip2); p1 <= OW'(sc1 >> OFW);
            sc2 = (2*OW)'(ed_2[4]) * (2*OW)'(nr_recip2); p2 <= OW'(sc2 >> OFW);
            sc3 = (2*OW)'(ed_3[4]) * (2*OW)'(nr_recip2); p3 <= OW'(sc3 >> OFW);
        end
    end

    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (nr_e3 !== edv[4]) begin
            $display("ERROR: valid misalignment at Stage F: nr_e3=%0b edv[4]=%0b",
                      nr_e3, edv[4]);
        end
    end
    // synthesis translate_on

endmodule : simd_softmax
