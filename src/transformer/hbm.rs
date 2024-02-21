use cudarc::driver::{CudaSlice, DeviceSlice};
use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha20Rng;
use crate::device::cpu::CPU;
use crate::device::gpu::GPU;
use super::ram::TransformerCPU;
use super::state::{RunState, RunStateView, TransformerWeights, TransformerWeightsView};
use super::{Config, Storage, Transformer};
pub struct TransformerGPU {
    pub config: Config,
    pub weights: TransformerWeights<CudaSlice<f32>>,
    pub state: RunState<CudaSlice<f32>>,
    pub device: GPU,
    // for debugging purposes.
    pub _cpu_state: RunState<Vec<f32>>,
}

impl Storage for CudaSlice<f32> {
    fn length(&self) -> usize {
        self.len()
    }
}

impl Transformer for TransformerGPU {
    fn cpu_state(&self) -> &RunState<Vec<f32>> {
        &self._cpu_state
    }
    fn from_file(cp_path: &str) -> Self {
        let mut tcpu = TransformerCPU::from_file(cp_path);
        let gpu = GPU::new();
        Self {
            config: tcpu.get_config(),
            weights: TransformerWeights::from_weight(&mut tcpu.weights, &gpu),
            state: RunState::from_state(&mut tcpu.state, &gpu),
            device: gpu,
            _cpu_state: tcpu.state,
        }
    }

    fn get_config(&self) -> Config {
        self.config.clone()
    }

}

// Allocate data in GPU memory and return a pointer to the location. Leak this object since the
// lifecycle of the GPU object is the same as the local CudaSlice object.
fn allocate(gpu: &GPU, data: &Vec<f32>) -> CudaSlice<f32> {
    gpu.gpu.htod_sync_copy(&data).unwrap()
}

impl RunState<CudaSlice<f32>> {
    pub fn from_state(state: &mut RunState<Vec<f32>>, device: &GPU) -> Self {
        Self {
            x: allocate(device, &state.x.as_mut()),
            xb: allocate(device, &state.xb.as_mut()),
            xb2: allocate(device, &state.xb2.as_mut()),
            hb: allocate(device, &state.hb.as_mut()),
            hb2: allocate(device, &state.hb2.as_mut()),
            q: allocate(device, &state.q.as_mut()),
            k: allocate(device, &state.k.as_mut()),
            v: allocate(device, &state.v.as_mut()),
            att: allocate(device, &state.att.as_mut()),
            logits: allocate(device, &state.logits.as_mut()),
            key_cache: allocate(device, &state.key_cache.as_mut()),
            value_cache: allocate(device, &state.value_cache.as_mut()),
        }
    }

    #[allow(dead_code)]
    // GPU state debug utility
    pub fn into_state(&self, device: &GPU, state: &mut RunState<Vec<f32>>) {
        device.gpu.dtoh_sync_copy_into(&self.x, &mut state.x.as_mut()).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.xb, &mut state.xb.as_mut()).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.xb2, &mut state.xb2.as_mut()).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.hb, &mut state.hb.as_mut()).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.hb2, &mut state.hb2.as_mut()).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.q, &mut state.q.as_mut()).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.k, &mut state.k.as_mut()).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.v, &mut state.v.as_mut()).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.att, &mut state.att.as_mut()).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.logits, &mut state.logits.as_mut()).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.key_cache, &mut state.key_cache.as_mut()).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.value_cache, &mut state.value_cache.as_mut()).unwrap();
    }
}


impl TransformerWeights<CudaSlice<f32>> {
    pub fn from_weight(tw: &mut TransformerWeights<Vec<f32>>, device: &GPU) -> Self {
        let token_embedding_table = allocate(device, &tw.token_embedding_table.as_mut());
        let rms_att_weight = allocate(device, &tw.rms_att_weight.as_mut());
        let rms_ffn_weight = allocate(device, &tw.rms_ffn_weight.as_mut());
        let wq = allocate(device, &tw.wq.as_mut());
        let wk = allocate(device, &tw.wk.as_mut());
        let wv = allocate(device, &tw.wv.as_mut());
        let wo = allocate(device, &tw.wo.as_mut());
        let w1 = allocate(device, &tw.w1.as_mut());
        let w2 = allocate(device, &tw.w2.as_mut());
        let w3 = allocate(device, &tw.w3.as_mut());
        let rms_final_weight = allocate(device, &tw.rms_final_weight.as_mut());
        let freq_cis_real = allocate(device, &tw.freq_cis_real.as_mut());
        let freq_cis_imag = allocate(device, &tw.freq_cis_imag.as_mut());

        let wcls = allocate(device, &tw.wcls.as_mut());

        Self {
            token_embedding_table: token_embedding_table,
            rms_att_weight: rms_att_weight,
            rms_ffn_weight: rms_ffn_weight,
            wq: wq,
            wk: wk,
            wv: wv,
            wo: wo,
            w1: w1,
            w2: w2,
            w3: w3,
            rms_final_weight: rms_final_weight,
            freq_cis_real: freq_cis_real,
            freq_cis_imag: freq_cis_imag,
            wcls_exists: true,
            wcls,
        }

    }
}


impl<'a, 'b: 'a> TransformerWeightsView<'a, CudaSlice<f32>> {

    pub fn from_gpu_ws(ws: &'a TransformerWeights<CudaSlice<f32>>) -> TransformerWeightsView<'a, CudaSlice<f32>> {
        TransformerWeightsView {
            token_embedding_table: View::new(&ws.token_embedding_table),
            rms_att_weight: View::new(&ws.rms_att_weight),
            rms_ffn_weight: View::new(&ws.rms_ffn_weight),
            wq: View::new(&ws.wq),
            wk: View::new(&ws.wk),
            wv: View::new(&ws.wv),
            wo: View::new(&ws.wo),
            w1: View::new(&ws.w1),
            w2: View::new(&ws.w2),
            w3: View::new(&ws.w3),
            rms_final_weight: View::new(&ws.rms_final_weight),
            freq_cis_real: View::new(&ws.freq_cis_real),
            freq_cis_imag: View::new(&ws.freq_cis_imag),
            wcls: View::new(&ws.wcls),
            wcls_exists: ws.wcls_exists
        }

    }
}

