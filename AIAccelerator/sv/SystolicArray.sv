// =============================================================================
// systolicArray_drain.sv - Final Working Version
// =============================================================================

module systolic_tiled #(parameter TILE_MAX = 4) (
    input  logic clk, rst,
    input  logic capture,                  
    input  logic drain_en,                 
    input  logic signed  [7:0] a    [TILE_MAX], // Weights
    input  logic signed  [7:0] b    [TILE_MAX], // Activations
    output logic signed [31:0] bottom_row [TILE_MAX] 
);

    logic signed [7:0]  south [TILE_MAX][TILE_MAX];
    logic signed [7:0]  east  [TILE_MAX][TILE_MAX];
    logic signed [31:0] drain_wires [TILE_MAX][TILE_MAX];

    genvar i, j;
    generate
        for (i=0; i<TILE_MAX; i++) begin : ROW
            for (j=0; j<TILE_MAX; j++) begin : COL
                mac PE (
                    .clk(clk), 
                    .rst(rst),
                    .capture (capture),
                    .drain_en(drain_en),
                    .drain_in ((i==0) ? 32'sd0 : drain_wires[i-1][j]),
                    .drain_out(drain_wires[i][j]),
                    
                    // --- STANDARD WIRING RESTORED ---
                    .north    ((i==0) ? a[j] : south[i-1][j]),
                    .west     ((j==0) ? b[i] : east [i][j-1]),
                    // --------------------------------
                    
                    .east     (east [i][j]),
                    .south    (south[i][j])
                );
            end
        end
    endgenerate

    always_comb begin
        for (int j=0; j<TILE_MAX; j++) begin
            bottom_row[j] = drain_wires[TILE_MAX-1][j];
        end
    end

endmodule
