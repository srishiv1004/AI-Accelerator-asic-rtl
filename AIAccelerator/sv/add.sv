module simd_add #(
    parameter int LANES  = 4,
    parameter int DATA_W = 32
)(
    input  logic                                 clk,
    input  logic                                 rst_n,
    input  logic                                 valid_i,
    input  logic signed [LANES-1:0][DATA_W-1:0]  a,
    input  logic signed [LANES-1:0][DATA_W-1:0]  b,
    output logic                                 valid_o,
    output logic signed [LANES-1:0][DATA_W-1:0]  out
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_o <= '0;
            out     <= '0;
        end else begin
            valid_o <= valid_i;
            
            for (int i = 0; i < LANES; i++) begin
                out[i] <= a[i] + b[i];
            end
        end
    end

endmodule : simd_add
