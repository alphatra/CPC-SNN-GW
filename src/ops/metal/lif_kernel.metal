#include <metal_stdlib>
using namespace metal;

// Fused LIF Kernel
// Parallelized over (Batch * Channels)
// Loops over Time
kernel void fused_lif_update(
    device const float* inputs [[ buffer(0) ]],      // (Batch, Channels, Time) - assuming ChannelsLast or specialized stride? 
                                                     // Ideally flat input or handled via indexing.
                                                     // Simpler: Input is (B, C, T) or (B, T, C). 
                                                     // If we use channels_last (B, T, C), then accessing T+1 is stride C.
                                                     
    device float* spikes [[ buffer(1) ]],            // Output Spikes
    device float* membrane_out [[ buffer(2) ]],      // Optional: Return final membrane or full history? usually full history for training.
    
    constant uint& time_steps [[ buffer(3) ]],
    constant float& beta [[ buffer(4) ]],
    constant float& threshold [[ buffer(5) ]],
    
    // Strides for indexing (assuming contiguous standard or channels_last)
    // To be generic, let's assume we process a flattened array of neurons.
    // Each thread handles ONE neuron sequence.
    // We need to know explicitly the initialization offset and stride per timestep.
    constant uint& batch_size [[ buffer(6) ]],
    constant uint& channels [[ buffer(7) ]],
    constant uint& input_stride_t [[ buffer(8) ]],   // Stride to get to next time step
    constant uint& input_stride_n [[ buffer(9) ]],   // Stride to get to next neuron (channel/batch elem)
    
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= batch_size * channels) return;
    
    // Determine input offset for this neuron
    // Logic: id maps to a specific (b, c) pair.
    // The base pointer for inputs[0, b, c] depends on memory format.
    // We iterate t=0..T-1.
    // Pointer math: base + t * input_stride_t
    
    // Actually, simpler if we enforce a specific layout or pass offset logic.
    // Let's assume standard PyTorch contiguous (B, C, T) -> stride_t=1, stride_c=T, stride_b=C*T
    // Or channels_last (B, T, C) -> stride_c=1, stride_t=C, stride_b=T*C
    
    // To be robust, we calculate base offset for this thread 'id'.
    // If input is flattened as (Neuron, Time) or (Time, Neuron)? 
    // Usually (B, C, T). 
    // If we use default PyTorch contiguous (B, C, T):
    // id 0 -> b=0, c=0. data is at [0, 0, :]
    // id 1 -> b=0, c=1. data is at [0, 1, :]
    // The distance between inputs[b,c,t] and inputs[b,c,t+1] is '1'.
    
    // If channels_last (B, T, C):
    // id 0 -> b=0, c=0.
    // id 1 -> b=0, c=1.
    // Distance between t and t+1 is 'C'.
    
    // We use the passed strides to handle both.
    // But we need the Base Offset for this 'id'.
    // Assumption: 'id' indexes the flattened (Batch, Channels) dimensions.
    // We need to reconstruct (b, c) from id to calculate base offset?
    // Or just pass "stride_between_neurons" (input_stride_n)? 
    // Yes, if uniform step.
    
    uint neuron_idx = id;
    uint base_offset = neuron_idx * input_stride_n; // Correct ONLY if (N, T) flat. 
    // If (B, C, T) (contiguous) -> Neuron 0 is at 0. Neuron 1 is at T. Stride_n = T. Stride_t = 1.
    // If (B, T, C) (channels_last) -> Neuron 0 is at 0. Neuron 1 is at 1. Stride_n = 1. Stride_t = C.
    
    // But wait, (B, T, C): Neuron (0,0) is at 0. (0,1) is at 1.
    // Neuron (0,0) at t=1 is at C.
    // So distinct neurons are interleaved.
    // The "base_offset" logic works if we assume the thread grid is 1D over neurons.
    // We need separate logic for NCHW vs NHWC if we don't pass full strides.
    // Let's rely on the passed strides.
    // Warning: id * input_stride_n only works if neurons are contiguous blocks or simple striding.
    // In NHWC:
    // Neuron 0 (c0) start: 0. Next t: +C.
    // Neuron 1 (c1) start: 1. Next t: +C.
    // input_stride_n = 1. input_stride_t = C.
    // This holds!
    // In NCHW:
    // Neuron 0 (c0) start: 0. Next t: +1.
    // Neuron 1 (c1) start: T. Next t: +1.
    // input_stride_n = T. input_stride_t = 1.
    // This holds too!
    
    // So we proceed with base + t * stride_t.
    
    float mem = 0.0;
    
    for (uint t = 0; t < time_steps; t++) {
        uint input_idx = base_offset + t * input_stride_t;
        
        float i = inputs[input_idx];
        
        // Update
        mem = mem * beta + i;
        
        float s = 0.0;
        if (mem > threshold) {
            s = 1.0;
            mem -= threshold;
        }
        
        // Write Output
        // Output shape is same as input shape usually?
        // spikes[input_idx] = s;
        // Same strides for output
        spikes[input_idx] = s;
        
        // Optional: Save mem
        // membrane_out[input_idx] = mem; 
    }
}
