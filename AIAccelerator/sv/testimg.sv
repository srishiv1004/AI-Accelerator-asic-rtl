`timescale 1ns/1ps

module tb_image;
    localparam int TILE_MAX = 16, AW = 8;

    localparam int IN_H = 16, IN_W = 16, IN_C = 3;
    localparam int K_H = 3, K_W = 3, ST = 1, PD = 0, OC = 3;
    localparam int OH = 14, OW = 14;   // (16-3)/1+1=14
    localparam int CONV_TOTAL = OH * OW; // 196 conv output pixels

    // GAP directly over conv output - feed 1px/beat so all channels land in lanes 0-2
    localparam int TOTAL_BEATS = CONV_TOTAL;                     // 196 beats, 1 px per beat
    localparam logic signed [31:0] GAP_RECIPROCAL = 32'sd334;   // round(65536/196)

    reg [31:0] dram_x[0:16383];
    reg [31:0] dram_y[0:16383];
    reg [31:0] dram_b[0:16383];
    reg [31:0] dram_z[0:16383];
    initial begin : init_dram_z
        integer j;
        for (j = 0; j < 16384; j++) dram_z[j] = 0;
    end

    logic clk = 0; always #5 clk = ~clk;
    logic rst, done;

    logic [15:0] x_addr, y_addr, b_addr;
    logic x_en, y_en, b_en;
    logic [31:0] x_data, y_data, b_data;
    always_ff @(posedge clk) begin
        x_data <= x_en ? dram_x[x_addr] : '0;
        y_data <= y_en ? dram_y[y_addr] : '0;
        b_data <= b_en ? dram_b[b_addr] : '0;
    end

    logic [31:0] out_byte_addr; logic [4:0] out_eff_cols;
    logic signed [31:0] acc_out [TILE_MAX]; logic acc_valid;

    logic        sa_csb0, sa_web0, sa_csb1; logic [3:0] sa_wm;
    logic [AW-1:0] sa_a0, sa_a1; logic [31:0] sa_di, sa_do0, sa_do1;
    logic        sb_csb0, sb_web0, sb_csb1; logic [15:0] sb_wm;
    logic [AW-1:0] sb_a0, sb_a1; logic [127:0] sb_di, sb_do0, sb_do1;
    logic [31:0] sbb0,sbb1,sbb2,sbb3,sbb4,sbb5,sbb6,sbb7;
    assign sb_do0 = {sbb3,sbb2,sbb1,sbb0};
    assign sb_do1 = {sbb7,sbb6,sbb5,sbb4};

    sky130_sram_1kbyte_1rw1r_32x256_8 #(.VERBOSE(0)) sram_a (
        .clk0(clk),.csb0(sa_csb0),.web0(sa_web0),.wmask0(sa_wm),
        .addr0(sa_a0),.din0(sa_di),.dout0(sa_do0),
        .clk1(clk),.csb1(sa_csb1),.addr1(sa_a1),.dout1(sa_do1));
    sky130_sram_1kbyte_1rw1r_32x256_8 #(.VERBOSE(0)) sram_b0 (
        .clk0(clk),.csb0(sb_csb0),.web0(sb_web0),.wmask0(sb_wm[3:0]),
        .addr0(sb_a0),.din0(sb_di[31:0]),.dout0(sbb0),
        .clk1(clk),.csb1(sb_csb1),.addr1(sb_a1),.dout1(sbb4));
    sky130_sram_1kbyte_1rw1r_32x256_8 #(.VERBOSE(0)) sram_b1 (
        .clk0(clk),.csb0(sb_csb0),.web0(sb_web0),.wmask0(sb_wm[7:4]),
        .addr0(sb_a0),.din0(sb_di[63:32]),.dout0(sbb1),
        .clk1(clk),.csb1(sb_csb1),.addr1(sb_a1),.dout1(sbb5));
    sky130_sram_1kbyte_1rw1r_32x256_8 #(.VERBOSE(0)) sram_b2 (
        .clk0(clk),.csb0(sb_csb0),.web0(sb_web0),.wmask0(sb_wm[11:8]),
        .addr0(sb_a0),.din0(sb_di[95:64]),.dout0(sbb2),
        .clk1(clk),.csb1(sb_csb1),.addr1(sb_a1),.dout1(sbb6));
    sky130_sram_1kbyte_1rw1r_32x256_8 #(.VERBOSE(0)) sram_b3 (
        .clk0(clk),.csb0(sb_csb0),.web0(sb_web0),.wmask0(sb_wm[15:12]),
        .addr0(sb_a0),.din0(sb_di[127:96]),.dout0(sbb3),
        .clk1(clk),.csb1(sb_csb1),.addr1(sb_a1),.dout1(sbb7));

    // ===================================================================
    // 1. SYSTOLIC ARRAY
    // ===================================================================
    systolic_pipe_conv #(
        .TILE_MAX(TILE_MAX),.IN_H(IN_H),.IN_W(IN_W),.IN_C(IN_C),
        .OUT_C(OC),.K_H(K_H),.K_W(K_W),.STRIDE(ST),.PAD(PD),.AW(AW),
        .USE_BIAS(1)
    ) dut (
        .clk(clk),.rst(rst),.done(done),
        .is_last_layer(1'b1),.act_sel(1'b0),.out_sel(1'b0),.skip_x(1'b0),
        .ext_x_addr_o(x_addr),.ext_x_rd_en_o(x_en),.ext_x_data_i(x_data),
        .ext_y_addr_o(y_addr),.ext_y_rd_en_o(y_en),.ext_y_data_i(y_data),
        .ext_b_addr_o(b_addr),.ext_b_rd_en_o(b_en),.ext_b_data_i(b_data),
        .bias_base_i(16'h0000),
        .ext_z_addr_o(),.ext_z_wr_en_o(),.ext_z_data_o(),.ext_z_wmask_o(),
        .sram_a_csb0(sa_csb0),.sram_a_web0(sa_web0),.sram_a_wmask0(sa_wm),
        .sram_a_addr0(sa_a0),.sram_a_din0(sa_di),.sram_a_dout0(sa_do0),
        .sram_a_csb1(sa_csb1),.sram_a_addr1(sa_a1),.sram_a_dout1(sa_do1),
        .sram_b_csb0(sb_csb0),.sram_b_web0(sb_web0),.sram_b_wmask0(sb_wm),
        .sram_b_addr0(sb_a0),.sram_b_din0(sb_di),.sram_b_dout0(sb_do0),
        .sram_b_csb1(sb_csb1),.sram_b_addr1(sb_a1),.sram_b_dout1(sb_do1),
        .sram_c_csb0(),.sram_c_web0(),.sram_c_wmask0(),.sram_c_addr0(),
        .sram_c_din0(),.sram_c_dout0(32'h0),
        .sram_c_csb1(),.sram_c_addr1(),.sram_c_dout1(32'h0),
        .acc_out(acc_out),.acc_valid(acc_valid),
        .out_byte_addr(out_byte_addr),.out_eff_cols(out_eff_cols));

    // ===================================================================
    // 2. GEARBOX
    // ===================================================================
    logic               gb_valid;
    logic [1:0][31:0]   gb_meta;
    logic [1:0]         gb_mask;
    logic signed [7:0][31:0] gb_data;
    logic               gb_stall;

    simd_gearbox_8lane #(.OC(OC)) u_gearbox (
        .clk(clk),.rst(rst),
        .acc_valid(acc_valid),.out_eff_cols(out_eff_cols),
        .out_byte_addr(out_byte_addr),.acc_out(acc_out),
        .simd_valid(gb_valid),.simd_meta_addr(gb_meta),
        .simd_valid_mask(gb_mask),.simd_data(gb_data),
        .stall_array(gb_stall));

    // ===================================================================
    // 2b. GEARBOX UNPACKER
    //
    // Gearbox packs 2 pixels per beat: px0 in lanes[3:0], px1 in lanes[7:4].
    // GAP accumulates each lane independently, so 2-px packing splits the
    // accumulation across two sets of lanes ? the final average would be wrong.
    // Unpack to 1 pixel per beat (channels in lanes 0-2) so the GAP sees
    // all 196 pixels cleanly in the same lanes.
    // ===================================================================
    logic               gap_in_valid;
    logic [1:0][31:0]   gap_in_meta;
    logic [1:0]         gap_in_mask;
    logic signed [7:0][31:0] gap_in_data;

    logic               unpack_has_px1;
    logic [31:0]        unpack_px1_addr;
    logic signed [31:0] unpack_px1_d0, unpack_px1_d1, unpack_px1_d2;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            gap_in_valid    <= 1'b0;
            gap_in_meta     <= '0;
            gap_in_mask     <= '0;
            gap_in_data     <= '0;
            unpack_has_px1  <= 1'b0;
            unpack_px1_addr <= '0;
            unpack_px1_d0   <= '0;
            unpack_px1_d1   <= '0;
            unpack_px1_d2   <= '0;
        end else begin
            gap_in_valid <= 1'b0;
            gap_in_mask  <= 2'b01;  // always 1 px per beat

            if (unpack_has_px1) begin
                // Fire buffered px1
                gap_in_valid   <= 1'b1;
                gap_in_meta[0] <= unpack_px1_addr;
                gap_in_meta[1] <= '0;
                gap_in_data[0] <= unpack_px1_d0;
                gap_in_data[1] <= unpack_px1_d1;
                gap_in_data[2] <= unpack_px1_d2;
                for (int i = 3; i < 8; i++) gap_in_data[i] <= '0;
                unpack_has_px1 <= 1'b0;
            end else if (gb_valid) begin
                // Fire px0 this cycle
                if (gb_mask[0]) begin
                    gap_in_valid   <= 1'b1;
                    gap_in_meta[0] <= gb_meta[0];
                    gap_in_meta[1] <= '0;
                    gap_in_data[0] <= gb_data[0];
                    gap_in_data[1] <= gb_data[1];
                    gap_in_data[2] <= gb_data[2];
                    for (int i = 3; i < 8; i++) gap_in_data[i] <= '0;
                end
                // Buffer px1 for next cycle
                if (gb_mask[1]) begin
                    unpack_has_px1  <= 1'b1;
                    unpack_px1_addr <= gb_meta[1];
                    unpack_px1_d0   <= gb_data[4];
                    unpack_px1_d1   <= gb_data[5];
                    unpack_px1_d2   <= gb_data[6];
                end
            end
        end
    end

    // ===================================================================
    // 3. SIMD PIPELINE (ReLU + GAP + Quantize)
    //    Opcode 6'b110100 -> quantize(5) + gavgpool(4) + relu(2)
    //    1 pixel per beat, 196 beats, channels in lanes 0-2 only.
    // ===================================================================
    logic               simd_valid_o;
    logic [1:0][31:0]   simd_meta_o;
    logic [1:0]         simd_mask_o;
    logic signed [7:0][31:0] simd_data_o;

    simd_pipeline_wrapper #(
        .LANES(8),.DATA_W(32),.FRAC_BITS(4),.OPCODE_W(6),
        .TOTAL_BEATS(TOTAL_BEATS),
        .IN_H(OH),.IN_W(OW),.POOL_K(2),.POOL_ST(2)
    ) u_simd (
        .clk(clk),.rst_n(~rst),
        .valid_i(gap_in_valid),.valid_mask_i(gap_in_mask),
        .opcode_i(6'b110100),           // quantize(5) + gavgpool(4) + relu(2)
        .meta_i(gap_in_meta),.data_i(gap_in_data),
        .operand_b('0),.bn_mean('0),.bn_inv_sqrt('0),.bn_gamma('0),.bn_beta('0),
        .gap_reciprocal(GAP_RECIPROCAL),
        .valid_o(simd_valid_o),.valid_mask_o(simd_mask_o),
        .meta_o(simd_meta_o),.data_o(simd_data_o),
        .pipeline_busy(),.current_stage());

    // ===================================================================
    // 4. WRITE-BACK: GAP output -> dram_z[0]
    // ===================================================================
    function automatic logic [7:0] clamp8(input logic signed [31:0] v);
        return (v < 0) ? 8'h00 : (v > 127) ? 8'h7F : v[7:0];
    endfunction

    always @(posedge clk) begin
        if (simd_valid_o) begin
            $display("[%0t] GAP OUT: CH0=%0d CH1=%0d CH2=%0d",
                     $time,
                     clamp8(simd_data_o[0]),
                     clamp8(simd_data_o[1]),
                     clamp8(simd_data_o[2]));
            dram_z[0] <= {8'h00,
                          clamp8(simd_data_o[2]),
                          clamp8(simd_data_o[1]),
                          clamp8(simd_data_o[0])};
        end
    end

    // ===================================================================
    // 5. STIMULUS & EXPORT
    // ===================================================================
    integer fd, code, idx;
    logic signed [31:0] val;
    real start_time, end_time;

    initial begin
        for (int i = 0; i < 16384; i++) begin
            dram_x[i] = 0; dram_y[i] = 0; dram_b[i] = 0;
        end

        fd = $fopen("img.txt","r");
        if (fd) begin
            idx = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd,"%d\n",val);
                if (code == 1) begin dram_x[idx/4][(idx%4)*8+:8] = val[7:0]; idx++; end
            end
            $fclose(fd);
        end

        fd = $fopen("weights.txt","r");
        if (fd) begin
            idx = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd,"%d\n",val);
                if (code == 1) begin dram_y[idx/4][(idx%4)*8+:8] = val[7:0]; idx++; end
            end
            $fclose(fd);
        end

        fd = $fopen("bias.txt","r");
        if (fd) begin
            idx = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd,"%d\n",val);
                if (code == 1) begin dram_b[idx/4][(idx%4)*8+:8] = val[7:0]; idx++; end
            end
            $fclose(fd);
        end

        rst = 1; @(posedge clk); #1; rst = 0;
        start_time = $realtime;
        $display("Conv %0dx%0d->%0dx%0d | GAP->1x1x%0d",
                 IN_H,IN_W,OH,OW,OC);

        wait(done === 1'b1);
        repeat(500) @(posedge clk);
        end_time = $realtime;

        $display("\nTotal Cycles: %0d", (end_time-start_time)/10);
        $display("=== GAP Output ===");
        $display("  CH0:%0d CH1:%0d CH2:%0d",
                 dram_z[0][7:0], dram_z[0][15:8], dram_z[0][23:16]);

        fd = $fopen("hw_out_python.txt","w");
        if (fd) begin
            for (int c = 0; c < OC; c++)
                $fwrite(fd, "%0d\n", dram_z[0][c*8+:8]);
            $fclose(fd);
            $display("Export -> hw_out_python.txt (GAP: %0d channels)", OC);
        end
        $finish;
    end
endmodule
