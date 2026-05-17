module ai_quantizer #(
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 8,
    parameter int FRAC_BITS = 8
)(
    input  logic signed [IN_WIDTH-1:0]  data_in,
    output logic signed [OUT_WIDTH-1:0] data_out
);
    // Parameter validation
    initial begin
        assert (FRAC_BITS <= IN_WIDTH) else 
            $fatal("Invalid params: FRAC_BITS=%0d must be <= IN_WIDTH=%0d", 
                   FRAC_BITS, IN_WIDTH);
        assert (OUT_WIDTH <= IN_WIDTH) else
            $fatal("OUT_WIDTH must be <= IN_WIDTH");
        assert (FRAC_BITS > 0) else
            $fatal("FRAC_BITS must be > 0 (use wire assignment if no quantization needed)");
    end

    // FIXED: Use FRAC_BITS directly (standard for AI quantization)
    localparam int SHIFT_AMT = FRAC_BITS;
    
    // Saturation limits - explicit signed values
    localparam signed [IN_WIDTH-1:0] MAX_LIMIT = (1 << (OUT_WIDTH-1)) - 1;
    localparam signed [IN_WIDTH-1:0] MIN_LIMIT = -(1 << (OUT_WIDTH-1));
    
    always_comb begin
        // FIXED: Use IN_WIDTH+1 bits for ALL intermediate calculations
        // This prevents overflow when adding rounding constant to max values
        logic signed [IN_WIDTH:0] data_in_ext;
        logic signed [IN_WIDTH:0] round_const;
        logic signed [IN_WIDTH:0] rounded_tmp;
        logic signed [IN_WIDTH:0] scaled_val;
        
        // Step 1: Sign-extend input to IN_WIDTH+1 bits
        data_in_ext = $signed(data_in);
        
        // Step 2: Create rounding constant in IN_WIDTH+1 bit space
        // Add 0.5 in fixed-point (2^(SHIFT_AMT-1)) for round-to-nearest
        if (SHIFT_AMT > 0)
            round_const = signed'(1 << (SHIFT_AMT - 1));
        else
            round_const = '0;  // No rounding needed if no fractional bits
        
        // Step 3: Add in extended precision to prevent overflow
        rounded_tmp = data_in_ext + round_const;
        
        // Step 4: Arithmetic right shift (preserves sign)
        scaled_val = rounded_tmp >>> SHIFT_AMT;
        
        // Step 5: Saturate to output range with explicit signed comparison
        if      (scaled_val > $signed(MAX_LIMIT)) 
            data_out = MAX_LIMIT[OUT_WIDTH-1:0];  // Clamp to +127
        else if (scaled_val < $signed(MIN_LIMIT)) 
            data_out = MIN_LIMIT[OUT_WIDTH-1:0];  // Clamp to -128
        else                             
            data_out = scaled_val[OUT_WIDTH-1:0]; // Within range
    end
endmodule
