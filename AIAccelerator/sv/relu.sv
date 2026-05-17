module simd_relu_top #(
    parameter int DATA_WIDTH = 64,      // Configurable data width
    parameter int NUM_LANES  = 4        // Configurable SIMD lanes
)(
    input  logic                                clk,
    input  logic                                rst_n,
    input  logic [NUM_LANES-1:0][DATA_WIDTH-1:0] data_in,
    output logic [NUM_LANES-1:0][DATA_WIDTH-1:0] data_out
);
    // Combinational ReLU logic
    logic [NUM_LANES-1:0][DATA_WIDTH-1:0] relu_results;
    
    genvar i;
    generate
        for (i = 0; i < NUM_LANES; i++) begin : relu_lane
            // ReLU: output = (input < 0) ? 0 : input
            // MSB is sign bit for signed numbers
            assign relu_results[i] = (data_in[i][DATA_WIDTH-1] == 1'b1) ? '0 : data_in[i];
        end
    endgenerate
    
    // Output register stage
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) 
            data_out <= '0;
        else        
            data_out <= relu_results;
    end
endmodule
