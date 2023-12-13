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

        // output[i] = sum; //input[i];// weight[start + i];
    }

}

extern "C" __global__ void apply_position(float *q, float *k, float *pos_real, float *pos_img, int n_heads, int head_size) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < head_size / 2) {
        float fcr = pos_real[i];
        float fci = pos_img[i];
        q[i * 2] = q[i * 2] * fcr - q[i * 2 + 1] * fci;
        q[i * 2 + 1] = q[i * 2] * fcr + q[i * 2 + 1] * fcr;
        k[i * 2] = k[i * 2] * fcr - k[i * 2 + 1] * fci;
        k[i * 2 + 1] = k[i * 2] * fcr + k[i * 2 + 1] * fcr;
    }
}

extern "C" __device__ void softmax(float *arr, int N) {

    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i != 1) {return;}
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

extern "C" __global__ void multi_head_attention(float *xb, float *att, float *q, float *k_cache, float *v_cache, int layer, int dim, int pos, int head_size, int seq_len, int n_heads) {

    // replace the input config with a config struct. the code below also needs serious refactoring later
    // after correctness checks.
    int hh = blockIdx.x*blockDim.x+threadIdx.x;
    if (hh != 1) {
        return;
    }
    for (int h = 0; h < n_heads; h++) {
        int loff = layer * seq_len * dim;
        float *q_h = q + h * head_size;
        float *att_h = att + h * seq_len;

        for (int t = 0; t < pos + 1; t++) {
            int koff = loff + t * dim + h * head_size;
            float *k = k_cache + koff;

            float sum = 0.0;
            for (int idx = 0; idx < head_size; idx++) {
                sum += q_h[idx] * k[idx];
            }

            sum = sum / sqrtf(head_size);
            att_h[t] = sum;
        }

        softmax(att_h, pos + 1);

        float *xb_tmp = xb + h * head_size;
        for (int k = 0; k < head_size; k++) {
            xb_tmp[k] = 0;
        }
        for (int t = 0; t < pos + 1; t++) {
            int koff = loff + t * dim + h * head_size;
            float *v = v_cache + koff;
            float a = att_h[t];
            for (int xbi = 0; xbi < head_size; xbi++) {
                xb_tmp[xbi] += a * v[xbi];

                ///TDBD
                // xb_tmp[xbi] = v[0];
            }
        }
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
