use std::sync::Arc;

use cudarc::cublas::{CudaBlas, GemmConfig, Gemm};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig, DeviceRepr, DevicePtr, DevicePtrMut};
use cudarc::nvrtc::compile_ptx;

use crate::transformer::Config;
use crate::transformer::hbm::RunStateGPU;

use super::device::Device;

#[allow(dead_code)]
const TRANS: cublasOperation_t = cublasOperation_t::CUBLAS_OP_T;
#[allow(dead_code)]
const NO_TRANS: cublasOperation_t = cublasOperation_t::CUBLAS_OP_N;

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
    pub blas: Arc<CudaBlas>,
}

impl GPU {
    pub fn new() -> Self {
        let dev = CudaDevice::new(0).unwrap();

        let cu_file = std::fs::read_to_string("./src/device/math.cu").unwrap();
        let ptx = compile_ptx(cu_file).unwrap();
        dev.load_ptx(ptx, "module", &
            ["matmul",
             "copy_from_slice",
             "rmsnorm",
             "apply_position",
             "calculate_attention",
             "array_add",
             "array_mult",
             "update_xb",
             "sinu"]).unwrap();
        let blas = Arc::new(CudaBlas::new(dev.clone()).unwrap());
        Self {
            gpu: dev,
            blas: blas,
        }
    }

    pub fn array_add<T: DeviceRepr>(&self, output: T, inp: T, n: usize) {
        let f = self.gpu.get_func("module", "array_add").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (output, inp, n,)) }.unwrap();
    }

    pub fn array_mult<T: DeviceRepr>(&self, output: T, inp: T, n: i32) {
        let f = self.gpu.get_func("module", "array_mult").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (output, inp, n,)) }.unwrap();
    }

    pub fn sinu<T: DeviceRepr>(&self, output: T, n: i32) {
        let f = self.gpu.get_func("module", "sinu").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (output, n,)) }.unwrap();

    }

    pub fn multi_head_attention(&self, gpu_state: &RunStateGPU, cfg: &Config, layer: usize, pos: usize) {
        let head_size = cfg.dim / cfg.n_heads;
        let f = self.gpu.get_func("module", "calculate_attention").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(cfg.n_heads as u32), (
            &gpu_state.xb,
            &gpu_state.att,
            &gpu_state.q,
            &gpu_state.key_cache,
            &gpu_state.value_cache,
            layer,
            cfg.dim,
            pos,
            head_size,
            cfg.seq_len,
            cfg.n_heads,
        )) }.unwrap();

        let head_size = cfg.dim / cfg.n_heads;
        let f = self.gpu.get_func("module", "update_xb").unwrap();

        let lcfg = LaunchConfig {
            block_dim: (COL_TILE_WIDTH as u32, ROW_TILE_WIDTH as u32, 1),
            grid_dim: ((cfg.n_heads/COL_TILE_WIDTH + 2) as u32, (head_size/ROW_TILE_WIDTH + 2)  as u32, 1),
            shared_mem_bytes: 0,
        };

        unsafe { f.launch(lcfg, (
            &gpu_state.xb,
            &gpu_state.att,
            &gpu_state.q,
            &gpu_state.key_cache,
            &gpu_state.value_cache,
            layer,
            cfg.dim,
            pos,
            head_size,
            cfg.seq_len,
            cfg.n_heads,
        )) }.unwrap();

    }

    pub fn copy_from_slice<S: DeviceRepr, D: DeviceRepr>(&self, src: S, dest: D, n: i32)
    {
        let f = self.gpu.get_func("module", "copy_from_slice").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (src, dest, n,)).unwrap(); };
    }

    pub fn rmsnorm<O: DeviceRepr, X: DeviceRepr, W: DeviceRepr>(&self, o: O, x: X, w: W, n: i32) {
        let f = self.gpu.get_func("module", "rmsnorm").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (o, x, w, n,)) }.unwrap();
    }

    pub fn apply_position<Q: DeviceRepr, K: DeviceRepr, R: DeviceRepr, I: DeviceRepr>(&self, q: Q, k: K, pos_real: R, pos_img: I, n_heads: i32, head_size: i32) {
        let f = self.gpu.get_func("module", "apply_position").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems((head_size / 2 + 1) as u32), (q, k, pos_real, pos_img, n_heads, head_size)) }.unwrap();
    }

}

impl<MT: DeviceRepr, T: DeviceRepr, T2: DeviceRepr> Device<MT, T, T2> for GPU where  {
    fn matmul_1d(&self, o: MT, w: T, x: T2, n: usize) {
        self.matmul(o, w, x, n, n, 1)
    }

    fn matmul(&self, o: MT, a: T, b: T2, width: usize, o_rows: usize, o_cols: usize) {
        let f = self.gpu.get_func("module", "matmul").unwrap();
        let cfg = LaunchConfig {
            block_dim: (COL_TILE_WIDTH as u32, ROW_TILE_WIDTH as u32, 1),
            grid_dim: ((o_cols/COL_TILE_WIDTH + 2) as u32, (o_rows/ROW_TILE_WIDTH + 2) as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (a, b, o, width, o_rows, o_cols)) }.unwrap();
    }

    fn rmsnorm(&self, o: MT, x: T, w: T2, n: usize) {
        let f = self.gpu.get_func("module", "rmsnorm").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (o, x, w, n,)) }.unwrap();
    }

    fn softmax(&self, arr: MT, size: usize) {
        let f = self.gpu.get_func("module", "softmax").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(size as u32), (arr, size)) }.unwrap();
    }
}

impl GPU {

    pub fn matmul_cublas<A: DevicePtrMut<f32>, B: DevicePtr<f32>, C: DevicePtr<f32>>(&self, o: &mut A, a: &B, b: &C, width: usize, o_rows: usize, o_cols: usize) {

        let blas_cfg: GemmConfig<f32> = GemmConfig {
            transa: NO_TRANS,
            transb: NO_TRANS,
            m: o_cols as i32,
            n: o_rows as i32,
            k: width as i32,
            alpha: 1.0,
            lda: o_cols as i32,
            ldb: width as i32,
            beta: 0.0,
            ldc: o_cols as i32,
        };
        unsafe { self.blas.gemm(blas_cfg, b, a,  o).unwrap(); };
    }
}

#[cfg(test)]
mod tests {

    use cudarc::cublas::{GemmConfig, Gemm};

    use super::*;

    #[test]
    fn test_blas() {
        let gpu = GPU::new();
        let blas_cfg: GemmConfig<f32> = GemmConfig {
            transa: NO_TRANS,
            transb: NO_TRANS,
            m: 2,
            n: 4,
            k: 3,
            alpha: 1.0,
            lda: 2,
            ldb: 3,
            beta: 0.0,
            ldc: 2,
        };
        let r = vec![2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0];
        let l = vec![3.0f32, 3.0, 3.0, 3.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0];
        let o = vec![1.0f32; 8];
        let a_dev = gpu.gpu.htod_sync_copy(&r).unwrap();

        let b_dev = gpu.gpu.htod_sync_copy(&l).unwrap();
        let mut c_dev = gpu.gpu.htod_sync_copy(&o).unwrap();

        unsafe {
            let _ = gpu.blas.gemm(blas_cfg, &a_dev, &b_dev, &mut c_dev);
            // let _ = gpu.blas.gemm(blas_cfg, ptr, &b_dev, &mut c_dev);
        }

        let ar = gpu.gpu.sync_reclaim(c_dev);
        print!("{:?}", ar);
        // assert_eq!(ar.unwrap(), [18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0]);
    }
}