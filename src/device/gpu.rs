#[cfg(feature = "gpu")]
pub mod cuda {
    use std::collections::HashMap;
    use std::sync::Arc;

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

extern \"C\" __global__ void rmsnorm(float *output, float *input, float *weight, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) {
        int sum = 0;
        for (int d = 0; d < N; d++) {
            sum += input[d] * input[d];
        }
        int v = sqrt(1.0 / (sum / N) + 0.00001);
        output[i] = weight[i] * (v * input[i]);
    }

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
            dev.load_ptx(ptx, "module", &["matmul", "copy_from_slice", "rmsnorm"]).unwrap();
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
            println!(" COPY_FROM_SLICE: {}, {}", start, end);
            // assert!(start == 288);
            unsafe { f.launch(LaunchConfig::for_num_elems((end - start) as u32), (src, dest, start, end,)) }.unwrap();
        }

        pub fn rmsnorm(&self, o: CUdeviceptr, x: CUdeviceptr, w: CUdeviceptr) {

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
