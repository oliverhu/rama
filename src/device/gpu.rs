#[cfg(feature = "gpu")]
pub mod cuda {
    use std::collections::HashMap;
    use std::sync::Arc;

    use cudarc::driver::result::memcpy_dtoh_sync;
    use cudarc::driver::sys::CUdeviceptr;
    use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig, CudaFunction};
    use cudarc::nvrtc::compile_ptx;

    use super::super::device::*;

    const PTX_SRC: &str = "
extern \"C\" __global__ void matmul(float* A, float* B, float* C, int width, int C_rows, int C_cols) {
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

extern \"C\" __global__ void copy_from_slice(float *src, float *dest, int start, int end) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < end - start) {
        dest[i] = src[start + i];
    }
}

extern \"C\" __global__ void rmsnorm(float *output, float *input, float *weight, int start, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) {
        float sum = 0;
        for (int d = 0; d < N; d++) {
            sum += powf(input[d], 2);// input[d] * input[d];
        }
        int v = 1.0 / sqrt((sum / N) + 0.00001);
        output[i] = weight[start + i] * (v * input[i]);
        // output[i] = sum; //input[i];// weight[start + i];
    }

}

extern \"C\" __global__ void apply_position(float *q, float *k, float *pos_real, float *pos_img, int n_heads, int head_size) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < head_size / 2) {
        int fcr = pos_real[i];
        int fci = pos_img[i];
        q[i * 2] = q[i * 2] * fcr - q[i * 2 + 1] * fci;
        q[i * 2 + 1] = q[i * 2] * fcr + q[i * 2 + 1] * fcr;
        k[i * 2] = k[i * 2] * fcr - k[i * 2 + 1] * fci;
        k[i * 2 + 1] = k[i * 2] * fcr + k[i * 2 + 1] * fcr;
    }
}

extern \"C\" __global__ void multi_head_attention(float *att, float *q, float *xb, int att_size) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;

}
";

    ///
    /// Brief Introduction to CUDA Programming
    /// blockIdx.x, blockIdx.y, blockIdx.z are built-in variables that returns the block ID
    /// in the x-axis, y-axis, and z-axis of the block that is executing the given block of code.
    ///
    /// threadIdx.x, threadIdx.y, threadIdx.z are built-in variables that return the
    /// thread ID in the x-axis, y-axis, and z-axis of the thread that is being executed by this
    /// stream processor in this particular block.
    ///
    /// blockDim.x, blockDim.y, blockDim.z are built-in variables that return the “block
    /// dimension” (i.e., the number of threads in a block in the x-axis, y-axis, and z-axis).
    ///
    /// The full global thread ID in x dimension can be computed by:
    ///  x = blockIdx.x * blockDim.x + threadIdx.x;
    ///
    /// Personally I found this blog post quite easy to follow or as a reference:
    ///     https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/
    ///

    const ROW_TILE_WIDTH: usize = 32;
    const COL_TILE_WIDTH: usize = 32;

    pub struct GPU {
        // Reference to the GPU device.
        pub gpu: Arc<CudaDevice>,
        // A map from string to a loaded function in the device.
        pub cuda_funs: HashMap<String, CudaFunction>,
    }


    ///
    /// Expected APIs:
    /// let g = GPU::new();
    /// g.copy_weights(); // copy transformer weights & state weights into GPU memory.
    /// g.matmul(o, a, b);
    /// let host_o = mut [X; Y];
    /// g.copy_to_host(o, host_o);
    ///
    impl GPU {
        pub fn new() -> Self {
            let dev = CudaDevice::new(0).unwrap();
            let ptx = compile_ptx(PTX_SRC).unwrap();
            dev.load_ptx(ptx, "module", &["matmul", "copy_from_slice", "rmsnorm", "apply_position"]).unwrap();
            // let f: CudaFunction = dev.get_func("matmul", "matmul").unwrap();
            let cf = HashMap::new();
            // cf.insert("matmul".to_string(), f);
            Self {
                gpu: dev,
                cuda_funs: cf,
            }
        }
        pub fn copy_from_slice(&self, src: CUdeviceptr, dest: CUdeviceptr, start: i32, end: i32) {
            let f = self.gpu.get_func("module", "copy_from_slice").unwrap();
            unsafe { f.launch(LaunchConfig::for_num_elems((end - start) as u32), (src, dest, start, end,)) }.unwrap();
        }

        pub fn rmsnorm(&self, o: CUdeviceptr, x: CUdeviceptr, w: CUdeviceptr, start: i32, n: i32) {
            let f = self.gpu.get_func("module", "rmsnorm").unwrap();
            unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (o, x, w, start, n,)) }.unwrap();
        }

        pub fn matmul2(&self, o: CUdeviceptr, a: CUdeviceptr, b: CUdeviceptr, width: usize, o_rows: i32, o_cols: i32) {
            let f = self.gpu.get_func("module", "matmul").unwrap();
            let cfg = LaunchConfig {
                block_dim: (COL_TILE_WIDTH as u32, ROW_TILE_WIDTH as u32, 1),
                grid_dim: ((o_cols/COL_TILE_WIDTH as i32 + 1) as u32, (o_rows/ROW_TILE_WIDTH as i32 + 1) as u32, 1),
                shared_mem_bytes: 0,
            };
            unsafe { f.launch(cfg, (a, b, o, width, o_rows, o_cols)) }.unwrap();
        }

        pub fn apply_position(&self, q: CUdeviceptr, k: CUdeviceptr, pos_real: CUdeviceptr, pos_img: CUdeviceptr, n_heads: i32, head_size: i32) {
            let f = self.gpu.get_func("module", "apply_position").unwrap();
            unsafe { f.launch(LaunchConfig::for_num_elems((head_size / 2 + 1) as u32), (q, k, pos_real, pos_img, n_heads, head_size)) }.unwrap();
        }
        ///
        /// o_buf: the buffer to write the GPU ram into
        pub fn debug(&self, o_buf: &mut Vec<f32>, input: CUdeviceptr) {
            unsafe { let _ = memcpy_dtoh_sync(o_buf, input); };
            println!("--------------------\noutput_buf is: {:?}\n", o_buf)
        }
    }

    impl Device for GPU {
        type Err = DriverError;
        fn matmul(o: &mut [f32], a: &[f32], b: &[f32], width: usize, o_rows: usize, o_cols: usize) -> Result<(), DriverError> {
            let ptx = compile_ptx(PTX_SRC).unwrap();

            let dev = CudaDevice::new(0)?;

            dev.load_ptx(ptx, "matmul", &["matmul"]).unwrap();
            let f = dev.get_func("matmul", "matmul").unwrap();
            let a_dev = dev.htod_sync_copy(&a)?;
            let b_dev: cudarc::driver::CudaSlice<f32> = dev.htod_sync_copy(&b)?;
            let mut o_dev = dev.htod_sync_copy(&o)?;
            // println!("Copied in {:?}", start.elapsed());

            let cfg = LaunchConfig {
                block_dim: (COL_TILE_WIDTH as u32, ROW_TILE_WIDTH as u32, 1),
                grid_dim: ((o_cols/COL_TILE_WIDTH + 1) as u32, (o_rows/ROW_TILE_WIDTH + 1) as u32, 1),
                shared_mem_bytes: 0,
            };

            // let cfg = LaunchConfig {
            //     block_dim: (o_cols as u32, o_rows as u32, 1),
            //     grid_dim: (20, 20, 1),
            //     shared_mem_bytes: 0,
            // };

            unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut o_dev, width, o_rows, o_cols)) }?;
            dev.dtoh_sync_copy_into(&o_dev,  o)?;
            // println!("Found {:?} in {:?}", o, start.elapsed());

            Ok(())

        }
    }

    #[cfg(test)]
    mod tests {

        use super::*;

        use cudarc::driver::{sys::CUdeviceptr, CudaSlice, DevicePtr, DeviceRepr};
        use rand::prelude::*;


        #[test]
        fn test_matrix_mul2() {
            let a_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let b_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let mut c_host = [0.0f32; 4];
            let _ = GPU::matmul(&mut c_host, &a_host, &b_host, 3, 2, 2);

            assert_eq!(c_host, [22.0, 28.0, 49.0, 64.0]);

            let mut rng = thread_rng();

            // Test size larger than 1024 threads
            const SIZE: usize = 288*288;
            let mut arr1 = [0.0f32; SIZE];
            let mut arr2 = [0.0f32; SIZE];
            let mut oo = [0.0f32; 288];
            for i in 0..SIZE {
                arr1[i] = rng.gen::<f32>();
                arr2[i] = rng.gen::<f32>();
            }

            let e = GPU::matmul(&mut oo, &arr1, &arr2, 288, 288, 288);
            match e {
                Ok(_) => (),
                Err(_) => panic!("error!"),
            }

            assert_ne!(oo[0], 0.0f32);
            assert_ne!(oo[287], 0.0f32);
        }

    }
}
