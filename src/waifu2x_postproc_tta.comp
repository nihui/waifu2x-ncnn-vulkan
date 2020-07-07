
#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#define sfp float16_t
#else
#define sfp float
#endif

#if NCNN_int8_storage
#extension GL_EXT_shader_8bit_storage: require
#endif

layout (constant_id = 0) const int bgr = 0;

layout (binding = 0) readonly buffer bottom_blob0 { sfp bottom_blob0_data[]; };
layout (binding = 1) readonly buffer bottom_blob1 { sfp bottom_blob1_data[]; };
layout (binding = 2) readonly buffer bottom_blob2 { sfp bottom_blob2_data[]; };
layout (binding = 3) readonly buffer bottom_blob3 { sfp bottom_blob3_data[]; };
layout (binding = 4) readonly buffer bottom_blob4 { sfp bottom_blob4_data[]; };
layout (binding = 5) readonly buffer bottom_blob5 { sfp bottom_blob5_data[]; };
layout (binding = 6) readonly buffer bottom_blob6 { sfp bottom_blob6_data[]; };
layout (binding = 7) readonly buffer bottom_blob7 { sfp bottom_blob7_data[]; };
layout (binding = 8) readonly buffer alpha_blob { sfp alpha_blob_data[]; };
#if NCNN_int8_storage
layout (binding = 9) writeonly buffer top_blob { uint8_t top_blob_data[]; };
#else
layout (binding = 9) writeonly buffer top_blob { float top_blob_data[]; };
#endif

layout (push_constant) uniform parameter
{
    int w;
    int h;
    int cstep;

    int outw;
    int outh;
    int outcstep;

    int offset_x;
    int gx_max;

    int channels;

    int alphaw;
    int alphah;
} p;

void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);

    if (gx >= p.gx_max || gy >= p.outh || gz >= p.channels)
        return;

    float v;

    if (gz == 3)
    {
        v = float(alpha_blob_data[gy * p.alphaw + gx]);
    }
    else
    {
        int gzi = gz * p.cstep;

        float v0 = float(bottom_blob0_data[gzi + gy * p.w + gx]);
        float v1 = float(bottom_blob1_data[gzi + gy * p.w + (p.w - 1 - gx)]);
        float v2 = float(bottom_blob2_data[gzi + (p.h - 1 - gy) * p.w + (p.w - 1 - gx)]);
        float v3 = float(bottom_blob3_data[gzi + (p.h - 1 - gy) * p.w + gx]);
        float v4 = float(bottom_blob4_data[gzi + gx * p.h + gy]);
        float v5 = float(bottom_blob5_data[gzi + gx * p.h + (p.h - 1 - gy)]);
        float v6 = float(bottom_blob6_data[gzi + (p.w - 1 - gx) * p.h + (p.h - 1 - gy)]);
        float v7 = float(bottom_blob7_data[gzi + (p.w - 1 - gx) * p.h + gy]);

        v = (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7) * 0.125f;

        const float denorm_val = 255.f;

        v = v * denorm_val;
    }

    const float clip_eps = 0.5f;

    v = v + clip_eps;

#if NCNN_int8_storage
    int v_offset = gy * p.outw + gx + p.offset_x;

    uint v32 = clamp(uint(floor(v)), 0, 255);

    if (bgr == 1 && gz != 3)
        top_blob_data[v_offset * p.channels + 2 - gz] = uint8_t(v32);
    else
        top_blob_data[v_offset * p.channels + gz] = uint8_t(v32);
#else
    int v_offset = gz * p.outcstep + gy * p.outw + gx + p.offset_x;

    top_blob_data[v_offset] = v;
#endif
}
