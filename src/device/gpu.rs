use std::sync::Arc;

use cudarc::cublas::{CudaBlas, GemmConfig, Gemm};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, CudaViewMut, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::transformer::ram::RunStateView;
use crate::transformer::{Config, MutView, View};

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

impl MutView<'_, CudaSlice<f32>> {
    fn cudaview(&self) -> CudaView<'_, f32> {
        let r = self.range.clone();
        self.as_ref().slice(r)
    }

    fn cudamutview(&mut self) -> CudaViewMut<'_, f32> {
        let r = self.range.clone();
        self.as_mut().slice_mut(r)
    }
}

impl View<'_, CudaSlice<f32>> {
    fn cudaview(&self) -> CudaView<'_, f32> {
        let r = self.range.clone();
        self.as_ref().slice(r)
    }
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

    pub fn array_add(&self, target: &mut MutView<'_, CudaSlice<f32>>, source: &View<'_, CudaSlice<f32>>,  n: usize) {
        let f = self.gpu.get_func("module", "array_add").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (&target.cudaview(), &source.cudaview(), n,)) }.unwrap();
    }

    pub fn array_mult(&self, target: &mut MutView<'_, CudaSlice<f32>>, source: &View<'_, CudaSlice<f32>>, n: usize) {
        let f = self.gpu.get_func("module", "array_mult").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (&target.cudaview(), &source.cudaview(), n,)) }.unwrap();

    }

    pub fn sinu(&self, o: &mut MutView<'_, CudaSlice<f32>>, n: usize) {
        let f = self.gpu.get_func("module", "sinu").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (&o.cudaview(), n,)) }.unwrap();
    }

    pub fn multi_head_attention(&self, rsv: &mut RunStateView<'_, CudaSlice<f32>>,
                                cfg: &Config,
                                layer: usize,
                                pos: usize,) {
        let head_size = cfg.dim / cfg.n_heads;
        let f = self.gpu.get_func("module", "calculate_attention").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(cfg.n_heads as u32), (
            &rsv.xb.cudaview(),
            &rsv.att.cudaview(),
            &rsv.q.cudaview(),
            &rsv.key_cache.cudaview(),
            &rsv.value_cache.cudaview(),
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
            &rsv.xb.cudaview(),
            &rsv.att.cudaview(),
            &rsv.q.cudaview(),
            &rsv.key_cache.cudaview(),
            &rsv.value_cache.cudaview(),
            layer,
            cfg.dim,
            pos,
            head_size,
            cfg.seq_len,
            cfg.n_heads,
        )) }.unwrap();

    }
    pub fn copy_from_slice(&self, target: &mut MutView<'_, CudaSlice<f32>>, source: &View<'_, CudaSlice<f32>>, n: usize) {
    // pub fn copy_from_slice<S: DeviceRepr, D: DeviceRepr>(&self, src: S, dest: D, n: i32) {
        let f = self.gpu.get_func("module", "copy_from_slice").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (&source.cudaview(), &target.cudaview(), n,)).unwrap(); };
    }

    pub fn rmsnorm(&self, o: &mut MutView<'_, CudaSlice<f32>>, x: &View<'_, CudaSlice<f32>>,
                        weight: &View<'_, CudaSlice<f32>>, n: usize) {
        let f = self.gpu.get_func("module", "rmsnorm").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (&o.cudaview(), &x.cudaview(), &weight.cudaview(), n,)) }.unwrap();
    }

    pub fn apply_position(&self, q: &mut MutView<'_, CudaSlice<f32>>, k: &mut MutView<'_, CudaSlice<f32>>, pos_real: &View<'_, CudaSlice<f32>>, pos_img: &View<'_, CudaSlice<f32>>, head_size: usize) {
        let f = self.gpu.get_func("module", "apply_position").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems((head_size / 2 + 1) as u32), (&q.cudaview(), &k.cudaview(), &pos_real.cudaview(), &pos_img.cudaview(), head_size)) }.unwrap();
    }

}

impl GPU {
    pub fn matmul_1d(&self, o: &mut MutView<'_, CudaSlice<f32>>, w: &View<'_, CudaSlice<f32>>, x: &View<'_, CudaSlice<f32>>, n: usize) {
    // fn matmul_1d<MT, T, T2>(&self, o: MT, w: T, x: T2, n: usize) where MT: DeviceRepr, T: DeviceRepr, T2: DeviceRepr{
        self.matmul(o, w, x, n, n, 1)
    }

    pub fn matmul(&self, o: &mut MutView<'_, CudaSlice<f32>>, a: &View<'_, CudaSlice<f32>>, b: &View<'_, CudaSlice<f32>>, width: usize, o_rows: usize, o_cols: usize) {
        let f = self.gpu.get_func("module", "matmul").unwrap();
        let cfg = LaunchConfig {
            block_dim: (COL_TILE_WIDTH as u32, ROW_TILE_WIDTH as u32, 1),
            grid_dim: ((o_cols/COL_TILE_WIDTH + 2) as u32, (o_rows/ROW_TILE_WIDTH + 2) as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (&a.cudaview(), &b.cudaview(), &o.cudaview(), width, o_rows, o_cols)) }.unwrap();
    }

    // fn softmax<MT, T, T2>(&self, arr: MT, size: usize) where MT: DeviceRepr, T: DeviceRepr, T2: DeviceRepr {
    pub fn softmax<'a>(&self, x: &mut MutView<'a, CudaSlice<f32>>, n: usize) {
        let f = self.gpu.get_func("module", "softmax").unwrap();
        unsafe { f.launch(LaunchConfig::for_num_elems(n as u32), (&x.cudaview(), n)) }.unwrap();
    }
}

impl GPU {

    pub fn matmul_cublas(&self, o: &mut MutView<'_, CudaSlice<f32>>, a: &View<'_, CudaSlice<f32>>, b: &View<'_, CudaSlice<f32>>, width: usize, o_rows: usize, o_cols: usize) {
    // pub fn matmul_cublas<A: DevicePtrMut<f32>, B: DevicePtr<f32>, C: DevicePtr<f32>>(&self, o: &mut A, a: &B, b: &C, width: usize, o_rows: usize, o_cols: usize) {

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
        unsafe { self.blas.gemm(blas_cfg, &b.cudaview(), &a.cudaview(),  &mut o.cudamutview()).unwrap(); };
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