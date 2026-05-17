module mac(
    input  logic               clk, 
    input  logic               rst,
    input  logic               capture,
    input  logic               drain_en,
    input  logic signed [31:0] drain_in,
    input  logic signed  [7:0] north, 
    input  logic signed  [7:0] west,
    output logic signed  [7:0] south, 
    output logic signed  [7:0] east,
    output logic signed [31:0] drain_out
);

    logic signed [15:0] multi_reg;  // ? PIPELINE STAGE
    logic signed [31:0] compute_reg;
    logic signed [31:0] drain_reg;

    assign drain_out = drain_reg;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            multi_reg   <= 16'sd0;
            compute_reg <= 32'sd0;
            drain_reg   <= 32'sd0;
            south       <= 8'sd0;
            east        <= 8'sd0;
        end else begin
            // ---------------------------------------------------------------
            // PIPELINE STAGE 0: Multiply
            // ---------------------------------------------------------------
            multi_reg <= north * west;

            // ---------------------------------------------------------------
            // PIPELINE STAGE 1: Accumulate
            // ---------------------------------------------------------------
            if (capture) begin
                drain_reg   <= compute_reg + 32'(signed'(multi_reg));
                compute_reg <= 32'sd0; 
            end else begin
                compute_reg <= compute_reg + 32'(signed'(multi_reg));
            end

            // ---------------------------------------------------------------
            // Drain Shift Logic
            // ---------------------------------------------------------------
            if (drain_en && !capture) begin
                drain_reg <= drain_in;
            end

            // ---------------------------------------------------------------
            // Data Flow
            // ---------------------------------------------------------------
            south <= north;
            east  <= west;
        end
    end
endmodule
