__kernel void compute_force(
    const int N,
    __global const float4* pos,
    __global float4* acc,
    const float eps2
) {
    int i = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    __local float4 shared_pos[64];  // Adjust size based on workgroup size (max ~64 for Apple GPU)

    if (i >= N) return;

    float4 pi = pos[i];
    float4 ai = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    int num_tiles = (N + local_size - 1) / local_size;

    for (int tile = 0; tile < num_tiles; tile++) {
        int j = tile * local_size + local_id;
        if (j < N) {
            shared_pos[local_id] = pos[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int tile_limit = min(local_size, N - tile * local_size);
        for (int k = 0; k < tile_limit; k++) {
            float4 pj = shared_pos[k];
            float dx = pj.x - pi.x;
            float dy = pj.y - pi.y;
            float dz = pj.z - pi.z;
            float distSqr = dx * dx + dy * dy + dz * dz + eps2;
            float invDist = 1.0f / sqrt(distSqr);
            float invDist3 = invDist * invDist * invDist;

            ai.x += pj.w * dx * invDist3;
            ai.y += pj.w * dy * invDist3;
            ai.z += pj.w * dz * invDist3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    acc[i] = ai;
}
