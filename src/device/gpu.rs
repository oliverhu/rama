use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::transformer::Config;
use crate::transformer::hbm::RunStateGPU;

use super::device::Device;

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
        let cu_file = std::fs::read_to_string("./src/device/math.cu").unwrap();
        let ptx = compile_ptx(cu_file).unwrap();
        dev.load_ptx(ptx, "module", &
            ["matmul",
             "copy_from_slice",
             "rmsnorm",
             "apply_position",
             "multi_head_attention",
             "array_add",
             "array_mult",
             "sinu"]).unwrap();
        Self {
            gpu: dev,
        }
    }

    pub fn array_add(&self, output: CUdeviceptr, inp: CUdeviceptr, n: usize) {
        let f = self.gpu.get_func("module", "array_add").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (output, inp, n,)) }.unwrap();
    }

    pub fn array_mult(&self, output: CUdeviceptr, inp: CUdeviceptr, n: i32) {
        let f = self.gpu.get_func("module", "array_mult").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (output, inp, n,)) }.unwrap();
    }

    pub fn sinu(&self, output: CUdeviceptr, n: i32) {
        let f = self.gpu.get_func("module", "sinu").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (output, n,)) }.unwrap();

    }

    pub fn multi_head_attention(&self, gpu_state: &RunStateGPU, cfg: &Config, layer: usize, pos: usize) {

        // extern \"C\" __global__ void multi_head_attention(float *xb, float *att, float *q, float *k_cache, float *v_cache, int layer, int dim, int pos, int head_size, int seq_len, int n_heads) {

        let head_size = cfg.dim / cfg.n_heads;
        let f = self.gpu.get_func("module", "multi_head_attention").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(cfg.n_heads as u32), (
            gpu_state.xb,
            gpu_state.att,
            gpu_state.q,
            gpu_state.key_cache,
            gpu_state.value_cache,
            layer,
            cfg.dim,
            pos,
            head_size,
            cfg.seq_len,
            cfg.n_heads,
        )) }.unwrap();

    }

    pub fn copy_from_slice(&self, src: CUdeviceptr, dest: CUdeviceptr, n: i32) {
        let f = self.gpu.get_func("module", "copy_from_slice").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (src, dest, n,)).unwrap(); };
    }

    pub fn rmsnorm(&self, o: CUdeviceptr, x: CUdeviceptr, w: CUdeviceptr, n: i32) {
        let f = self.gpu.get_func("module", "rmsnorm").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (o, x, w, n,)) }.unwrap();
    }

    pub fn apply_position(&self, q: CUdeviceptr, k: CUdeviceptr, pos_real: CUdeviceptr, pos_img: CUdeviceptr, n_heads: i32, head_size: i32) {
        let f = self.gpu.get_func("module", "apply_position").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems((head_size / 2 + 1) as u32), (q, k, pos_real, pos_img, n_heads, head_size)) }.unwrap();
    }

}

impl Device<CUdeviceptr, CUdeviceptr> for GPU {
    fn matmul_1d(&self, o: CUdeviceptr, w: CUdeviceptr, x: CUdeviceptr, n: usize) {
        self.matmul(o, w, x, n, n, 1)
    }

    fn matmul(&self, o: CUdeviceptr, a: CUdeviceptr, b: CUdeviceptr, width: usize, o_rows: usize, o_cols: usize) {
        let f = self.gpu.get_func("module", "matmul").unwrap();
        let cfg = LaunchConfig {
            block_dim: (COL_TILE_WIDTH as u32, ROW_TILE_WIDTH as u32, 1),
            grid_dim: ((o_cols/COL_TILE_WIDTH + 2) as u32, (o_rows/ROW_TILE_WIDTH + 2) as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (a, b, o, width, o_rows, o_cols)) }.unwrap();
    }
    fn rmsnorm(&self, o: CUdeviceptr, x: CUdeviceptr, w: CUdeviceptr, n: usize) {
        let f = self.gpu.get_func("module", "rmsnorm").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (o, x, w, n,)) }.unwrap();
    }

    fn softmax(&self, arr: CUdeviceptr, size: usize) {
        let f = self.gpu.get_func("module", "softmax").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(size as u32), (arr, size)) }.unwrap();
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_test_call_another() {
        let a_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let gpu = GPU::new();
        let f = gpu.gpu.get_func("module", "test_call_another").unwrap();
        let a_dev = gpu.gpu.htod_sync_copy(&a_host).unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(6), (&a_dev, 4,)) }.unwrap();
        let b_host = gpu.gpu.sync_reclaim(a_dev).unwrap();
        let b_host_eval = [100.0f32, 100.0, 100.0, 100.0, 5.0, 6.0];
        println!("{:?}", b_host);
        assert_eq!(b_host, b_host_eval)
    }
}