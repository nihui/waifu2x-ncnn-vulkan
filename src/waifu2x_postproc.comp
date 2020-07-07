
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

layout (binding = 0) readonly buffer bottom_blob { sfp bottom_blob_data[]; };
layout (binding = 1) readonly buffer alpha_blob { sfp alpha_blob_data[]; };
#if NCNN_int8_storage
layout (binding = 2) writeonly buffer top_blob { uint8_t top_blob_data[]; };
#else
layout (binding = 2) writeonly buffer top_blob { float top_blob_data[]; };
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
        v = float(bottom_blob_data[gz * p.cstep + gy * p.w + gx]);

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
