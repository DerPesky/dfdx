#include "cuda_fp16.h"

struct Conv1DOp {
    size_t kernel;
    size_t stride;
    size_t padding;
    size_t dilation;
    size_t groups;
    size_t batch;
    size_t chan_in;
    size_t chan_out;
    size_t h_in;
    size_t h_out;
};

template<typename T>
__device__ void unfold_input_into_patches(
    const Conv1DOp op,
    const T *image, // 3d (Batch, Groups * Channels, Height)
    const size_t *strides, // 3d image strides
    T *patches // 4d (Batch, Groups * Channels, KernelSize, HeightOut)
) {
    const size_t n = op.batch * op.groups * op.chan_in * op.h_out;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        unsigned int idx = i;
        const size_t oh = idx % op.h_out;
        idx /= op.h_out;
        const size_t c = idx % (op.chan_in * op.groups);
        idx /= (op.chan_in * op.groups);
        const size_t b = idx % op.batch;

        const T *image_i = image + b * strides[0] + c * strides[1];
        T *patches_i = patches + oh;
        patches_i += c * (op.kernel * op.h_out);
        patches_i += b * (op.groups * op.chan_in * op.kernel * op.h_out);

        T zero = 0.0;

        for (int k1 = 0; k1 < op.kernel; k1++) {
            const size_t x = oh * op.stride + op.dilation * k1 - op.padding;
            *patches_i = (x >= op.h_in) ? zero : image[x * strides[2]];
            patches_i += op.h_out;
        }
    }
}
