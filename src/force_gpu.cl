__kernel void compute_force(
    const int N,
    __global const float4* pos,
    __global float4* acc,
    const float eps2
) {
    int i = get_global_id(0);
    if (i >= N) return;

    float4 pi = pos[i];
    float4 ai = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int j = 0; j < N; j++) {
        float4 pj = pos[j];
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

    acc[i] = ai;
}
