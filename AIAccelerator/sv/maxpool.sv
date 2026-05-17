module simd_maxpool #(
    parameter int LANES  = 4,
    parameter int DATA_W = 32    // signed Q16.16
)(
    input  logic                                   clk,
    input  logic                                   rst_n,
    input  logic                                   valid_i,
    input  logic signed [LANES-1:0][DATA_W-1:0]   data_a,  // first set
    input  logic signed [LANES-1:0][DATA_W-1:0]   data_b,  // second set
    output logic                                   valid_o,
    output logic signed [LANES-1:0][DATA_W-1:0]   data_max
);
    // =========================================================================
    // Stage 1 ? compare a vs b per lane
    // =========================================================================
    logic signed [LANES-1:0][DATA_W-1:0] s1_max;
    logic                                 s1_valid;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= '0;
            s1_max   <= '0;
        end else begin
            s1_valid <= valid_i;
            for (int i = 0; i < LANES; i++) begin
                s1_max[i] <= ($signed(data_a[i]) > $signed(data_b[i])) ? 
                             data_a[i] : data_b[i];
            end
        end
    end
    
    // =========================================================================
    // Stage 2 ? register output
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_o  <= '0;
            data_max <= '0;
        end else begin
            valid_o  <= s1_valid;
            data_max <= s1_max;
        end
    end
    
    // Synthesis-time checks
    initial begin
        assert (LANES == 4) else
            $warning("Design optimized for 4 lanes; verify for other values");
    end
    
endmodule : simd_maxpool
