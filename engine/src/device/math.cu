// Most of the kernels here requires significant optimizations..

extern "C" __global__ void matmul(float* A, float* B, float* C, int width, int C_rows, int C_cols) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < C_rows && COL < C_cols) {
        float tmpSum = 0;
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < width; i++) {
            tmpSum += A[ROW * width + i] * B[i * C_cols + COL];
        }
        C[ROW * C_cols + COL] = tmpSum;
    }
}

extern "C" __global__ void copy_from_slice(float *src, float *dest, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) {
        dest[i] = src[i];
    }
}

extern "C" __global__ void rmsnorm(float *output, float *input, float *weight, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i == 1) {
        float sum = 0.0;
        for (int d = 0; d < N; d++) {
            sum += input[d] * input[d];
        }
        float v = 1.0 / sqrtf((sum / N) + 0.00001);
        for (int k = 0; k < N; k++) {
            output[k] = weight[k] * v * input[k];
        }
    }

}

extern "C" __global__ void apply_position(float *q, float *k, float *pos_real, float *pos_img, int head_size) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < head_size / 2) {
        float fcr = pos_real[i];
        float fci = pos_img[i];
        q[i * 2] = q[i * 2] * fcr - q[i * 2 + 1] * fci;
        q[i * 2 + 1] = q[i * 2] * fci + q[i * 2 + 1] * fcr;
        k[i * 2] = k[i * 2] * fcr - k[i * 2 + 1] * fci;
        k[i * 2 + 1] = k[i * 2] * fci + k[i * 2 + 1] * fcr;
    }
}

extern "C" __global__ void softmax(float *arr, int N) {
    // replace this max with a CUDA reduction function.
    float max = -10.0;
    for (int idx = 0; idx < N; idx++) {
        if (arr[idx] > max) {
            max = arr[idx];
        }
    }
    for (int i = 0; i < N; i++){
        arr[i] = expf(arr[i] - max);
    }
    float sum = 0;
    for (int j = 0; j < N; j++) {
        sum += arr[j];
    }
    for (int j = 0; j < N; j++) {
        arr[j] /= sum;
    }

}

extern "C" __global__ void update_xb(float *xb, float *att, float *q, float *k_cache, float *v_cache, int layer, int dim, int pos, int head_size, int seq_len, int n_heads) {
    int h = blockIdx.y*blockDim.y+threadIdx.y;
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if (h < n_heads && i < head_size) {
        int loff = layer * seq_len * dim;
        float *q_t = q + h * head_size;
        float *att_t = att + h * seq_len;
        float *xb_t = xb + h * head_size;

        xb_t[i] = 0;

        for (int t = 0; t < pos + 1; t++) {
            int koff = loff + t * dim + h * head_size;
            float *v_c = v_cache + koff;
            float a = att_t[t];
            atomicAdd(xb_t + i, a * v_c[i]);
        }

    }
}

extern "C" __global__ void calculate_attention(float *xb, float *att, float *q, float *k_cache, float *v_cache, int layer, int dim, int pos, int head_size, int seq_len, int n_heads) {
    int h = blockIdx.x*blockDim.x+threadIdx.x;
    if (h < n_heads) {
        int loff = layer * seq_len * dim;
        float *q_t = q + h * head_size;
        float *att_t = att + h * seq_len;
        float *xb_t = xb + h * head_size;
        for (int t = 0; t < pos + 1; t++) {
            int koff = loff + t * dim + h * head_size;
            float *k_t = k_cache + koff;

            float v = 0.0;
            for (int i = 0; i < head_size; i++) {
                v += q_t[i] * k_t[i];
            }
            att_t[t] = v / sqrtf(head_size);

        }
        softmax(att_t, pos + 1);
    }
}


extern "C" __global__ void array_add(float *x, float *xb, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) {
        x[i] += xb[i];
    }
}

extern "C" __global__ void array_mult(float *x, float *xb, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) {
        x[i] *= xb[i];
    }
}

extern "C" __global__ void sinu(float *x, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) {
        x[i] = x[i] * (1.0 / (1.0 + expf(-x[i])));
    }
}
