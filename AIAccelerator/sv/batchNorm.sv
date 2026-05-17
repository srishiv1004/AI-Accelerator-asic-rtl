module simd_batchnorm #(
    parameter int LANES     = 4,
    parameter int DATA_W    = 32,
    parameter int PARAM_W   = 32,
    parameter int FRAC_BITS = 16
)(
    input  logic                                    clk,
    input  logic                                    rst_n,
    input  logic                                    valid_i,
    input  logic signed [LANES-1:0][DATA_W-1:0]    data_in,
    input  logic signed [LANES-1:0][PARAM_W-1:0]   mean,
    input  logic signed [LANES-1:0][PARAM_W-1:0]   inv_sqrt,
    input  logic signed [LANES-1:0][PARAM_W-1:0]   gamma,
    input  logic signed [LANES-1:0][PARAM_W-1:0]   beta,
    output logic                                    valid_o,
    output logic signed [LANES-1:0][DATA_W-1:0]    data_out
);
    // =========================================================================
    // Stage 1 ? subtract mean, latch all parameters
    // =========================================================================
    logic signed [LANES-1:0][DATA_W-1:0]    s1_xm;
    logic signed [LANES-1:0][PARAM_W-1:0]   s1_inv_sqrt;
    logic signed [LANES-1:0][PARAM_W-1:0]   s1_gamma;
    logic signed [LANES-1:0][PARAM_W-1:0]   s1_beta;
    logic                                    s1_valid;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid    <= '0;
            s1_xm       <= '0;
            s1_inv_sqrt <= '0;
            s1_gamma    <= '0;
            s1_beta     <= '0;
        end else begin
            s1_valid <= valid_i;
            for (int i = 0; i < LANES; i++) begin
                s1_xm[i]       <= data_in[i] - mean[i];
                s1_inv_sqrt[i] <= inv_sqrt[i];
                s1_gamma[i]    <= gamma[i];
                s1_beta[i]     <= beta[i];
            end
        end
    end
    
    // =========================================================================
    // Stage 2 ? scale by gamma
    // =========================================================================
    logic signed [LANES-1:0][DATA_W-1:0]    s2_scaled_xm;
    logic signed [LANES-1:0][PARAM_W-1:0]   s2_inv_sqrt;
    logic signed [LANES-1:0][PARAM_W-1:0]   s2_beta;
    logic                                    s2_valid;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid      <= '0;
            s2_scaled_xm  <= '0;
            s2_inv_sqrt   <= '0;
            s2_beta       <= '0;
        end else begin
            s2_valid <= s1_valid;
            for (int i = 0; i < LANES; i++) begin
                automatic logic signed [63:0] prod;
                prod = $signed(s1_xm[i]) * $signed(s1_gamma[i]);
                s2_scaled_xm[i] <= DATA_W'($signed(prod) >>> FRAC_BITS);
                
                s2_inv_sqrt[i] <= s1_inv_sqrt[i];
                s2_beta[i]     <= s1_beta[i];
            end
        end
    end
    
    // =========================================================================
    // Stage 3 ? multiply by inv_sqrt, add beta
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_o  <= '0;
            data_out <= '0;
        end else begin
            valid_o <= s2_valid;
            for (int i = 0; i < LANES; i++) begin
                automatic logic signed [63:0] prod;
                prod = $signed(s2_scaled_xm[i]) * $signed(s2_inv_sqrt[i]);
                data_out[i] <= DATA_W'($signed(prod) >>> FRAC_BITS) + s2_beta[i];
            end
        end
    end
    
    // Synthesis-time checks
    initial begin
        assert (FRAC_BITS == DATA_W/2) else
            $fatal("FRAC_BITS must equal DATA_W/2 for Q16.16 multiply-shift to work");
        assert (LANES == 4) else
            $warning("Design optimised for 4 lanes; verify for other values");
    end
    
endmodule : simd_batchnorm
