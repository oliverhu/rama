use cudarc::driver::CudaSlice;
use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha20Rng;
use crate::device::cpu::CPU;
use crate::device::device::Device;
use crate::device::gpu::GPU;
use super::{ram::{RunState, TransformerWeights, TransformerCPU}, Transformer, Config};

pub struct RunStateGPU {
    pub x: CudaSlice<f32>,
    pub xb: CudaSlice<f32>,
    pub xb2: CudaSlice<f32>,
    pub hb: CudaSlice<f32>,
    pub hb2: CudaSlice<f32>,
    pub q: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub att: CudaSlice<f32>,
    pub logits: CudaSlice<f32>,
    pub key_cache: CudaSlice<f32>,
    pub value_cache: CudaSlice<f32>,
}

impl RunStateGPU {
    pub fn from_state(state: &RunState, device: &GPU) -> Self {
        Self {
            x: allocate(device, &state.x),
            xb: allocate(device, &state.xb),
            xb2: allocate(device, &state.xb2),
            hb: allocate(device, &state.hb),
            hb2: allocate(device, &state.hb2),
            q: allocate(device, &state.q),
            k: allocate(device, &state.k),
            v: allocate(device, &state.v),
            att: allocate(device, &state.att),
            logits: allocate(device, &state.logits),
            key_cache: allocate(device, &state.key_cache),
            value_cache: allocate(device, &state.value_cache),
        }
    }

    #[allow(dead_code)]
    // GPU state debug utility
    pub fn into_state(&self, device: &GPU, state: &mut RunState) {
        device.gpu.dtoh_sync_copy_into(&self.x, &mut state.x).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.xb, &mut state.xb).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.xb2, &mut state.xb2).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.hb, &mut state.hb).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.hb2, &mut state.hb2).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.q, &mut state.q).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.k, &mut state.k).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.v, &mut state.v).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.att, &mut state.att).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.logits, &mut state.logits).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.key_cache, &mut state.key_cache).unwrap();
        device.gpu.dtoh_sync_copy_into(&self.value_cache, &mut state.value_cache).unwrap();
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct TransformerWeightsGPU {
    token_embedding_table: CudaSlice<f32>,
    rms_att_weight: CudaSlice<f32>,
    rms_ffn_weight: CudaSlice<f32>,

    wq: CudaSlice<f32>,
    wk: CudaSlice<f32>,
    wv: CudaSlice<f32>,
    wo: CudaSlice<f32>,
    w1: CudaSlice<f32>,
    w2: CudaSlice<f32>,
    w3: CudaSlice<f32>,

    rms_final_weight: CudaSlice<f32>,
    freq_cis_real: CudaSlice<f32>,
    freq_cis_imag: CudaSlice<f32>,
    wcls_exists: bool,
    wcls: CudaSlice<f32>,
}

// Allocate data in GPU memory and return a pointer to the location. Leak this object since the
// lifecycle of the GPU object is the same as the local CudaSlice object.
fn allocate(gpu: &GPU, data: &Vec<f32>) -> CudaSlice<f32> {
    gpu.gpu.htod_sync_copy(&data).unwrap()
}

impl TransformerWeightsGPU {
    pub fn from_weight(tw: &TransformerWeights, device: &GPU) -> Self {
        let token_embedding_table = allocate(device, &tw.token_embedding_table);
        let rms_att_weight = allocate(device, &tw.rms_att_weight);
        let rms_ffn_weight = allocate(device, &tw.rms_ffn_weight);
        let wq = allocate(device, &tw.wq);
        let wk = allocate(device, &tw.wk);
        let wv = allocate(device, &tw.wv);
        let wo = allocate(device, &tw.wo);
        let w1 = allocate(device, &tw.w1);
        let w2 = allocate(device, &tw.w2);
        let w3 = allocate(device, &tw.w3);
        let rms_final_weight = allocate(device, &tw.rms_final_weight);
        let freq_cis_real = allocate(device, &tw.freq_cis_real);
        let freq_cis_imag = allocate(device, &tw.freq_cis_imag);
        let wcls: CudaSlice<f32>;
        let mut wcls_exists= false;

        match &tw.wcls {
            Some(val) =>
            {
                wcls = allocate(device, &val);
                wcls_exists = true;
            }
            None => {wcls = token_embedding_table.clone();},
        }

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
            wcls_exists,
            wcls,
        }

    }
}

pub struct TransformerGPU {
    pub config: Config,
    pub weights: TransformerWeightsGPU,
    pub state: RunStateGPU,
    pub device: GPU,
    // for debugging purposes.
    pub _cpu_state: RunState,
}

impl Transformer for TransformerGPU {
    fn cpu_state(&self) -> &RunState {
        &self._cpu_state
    }

    fn forward(&mut self, token: usize, pos: usize) {
        let gpu_weights = &self.weights;
        let cfg = &self.config;
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim;
        let head_size = dim / cfg.n_heads;
        let gpu = &self.device;
        let gpu_state = &mut self.state;

        self.device.copy_from_slice(&gpu_weights.token_embedding_table.slice(token * dim..), &gpu_state.x, dim as i32);

        let pos_real = gpu_weights.freq_cis_real.slice(pos * (head_size / 2)..);
        let pos_img = gpu_weights.freq_cis_imag.slice(pos * (head_size / 2)..);


        for layer in 0..cfg.n_layers {

            gpu.rmsnorm(&gpu_state.xb, &gpu_state.x, &gpu_weights.rms_att_weight.slice(layer * dim..), dim as i32);

            gpu.matmul_cublas(&mut gpu_state.q, &gpu_weights.wq.slice(layer * dim * dim..), &gpu_state.xb, dim, dim, 1);
            // gpu.matmul_cublas(&mut gpu_state.q, &gpu_weights.wq.slice(layer * dim * dim..), &gpu_state.xb, dim, dim, 1);
            gpu.matmul_cublas(&mut gpu_state.q, &gpu_weights.wq.slice(layer * dim * dim..), &gpu_state.xb, dim, dim, 1);

            gpu.matmul_cublas(&mut gpu_state.k, &gpu_weights.wk.slice(layer * dim * dim..), &gpu_state.xb, dim, dim, 1);

            gpu.matmul_cublas(&mut gpu_state.v, &gpu_weights.wv.slice(layer * dim * dim..), &gpu_state.xb, dim, dim, 1);

            for h in 0..cfg.n_heads {

                let q = &gpu_state.q.slice(h * head_size..);
                let k = &gpu_state.k.slice(h * head_size..);
                gpu.apply_position(q, k, &pos_real, &pos_img, cfg.n_heads as i32, head_size as i32);
            }

            // -- %%
            let lo = layer * cfg.seq_len * dim;
            gpu.copy_from_slice(&gpu_state.k, &gpu_state.key_cache.slice(lo + pos * dim..), dim as i32);
            gpu.copy_from_slice(&gpu_state.v, &gpu_state.value_cache.slice(lo + pos * dim..), dim as i32);
            // self.state.into_state(&mut self._cpu_state);
            gpu.multi_head_attention(&gpu_state, &cfg, layer, pos);
            // self.state.into_state(&mut self._cpu_state);

            gpu.matmul_cublas(&mut gpu_state.xb2, &gpu_weights.wo.slice(layer * dim * dim..), &gpu_state.xb, dim, dim, 1);

            gpu.array_add(&gpu_state.x, &gpu_state.xb2, dim);

            gpu.rmsnorm(&gpu_state.xb, &gpu_state.x, &gpu_weights.rms_ffn_weight.slice(layer * dim..), dim as i32);



            gpu.matmul_cublas(&mut gpu_state.hb, &gpu_weights.w1.slice(layer * hidden_dim * dim..), &gpu_state.xb, dim, hidden_dim, 1);
            gpu.matmul_cublas(&mut gpu_state.hb2, &gpu_weights.w3.slice(layer * hidden_dim * dim..), &gpu_state.xb, dim, hidden_dim, 1);

            //---
            gpu.sinu(&gpu_state.hb, hidden_dim as i32);


            gpu.array_mult(&gpu_state.hb, &gpu_state.hb2, hidden_dim as i32);


            gpu.matmul_cublas(&mut gpu_state.xb, &gpu_weights.w2.slice(layer * dim * hidden_dim..), &gpu_state.hb, hidden_dim, dim, 1);

            gpu.array_add(&gpu_state.x, &gpu_state.xb, dim);

        }


        gpu.copy_from_slice(&gpu_state.x, &gpu_state.xb, dim as i32);

        gpu.rmsnorm(&gpu_state.x, &gpu_state.xb, &gpu_weights.rms_final_weight, dim as i32);

        gpu.matmul_cublas(&mut gpu_state.logits, &gpu_weights.wcls, &gpu_state.x, dim, cfg.vocab_size, 1);

    }

    fn from_file(cp_path: &str) -> Self {
        let tcpu = TransformerCPU::from_file(cp_path);
        let gpu = GPU::new();
        Self {
            config: tcpu.get_config(),
            weights: TransformerWeightsGPU::from_weight(&tcpu.weights, &gpu),
            state: RunStateGPU::from_state(&tcpu.state, &gpu),
            device: gpu,
            _cpu_state: tcpu.state,
        }
    }

    fn get_config(&self) -> Config {
        self.config.clone()
    }

    fn sample(&mut self, temperature: f32) -> usize {
        let next;
        let rng_seed = 10;
        let mut rng = ChaCha20Rng::seed_from_u64(rng_seed);
        // let mut logits = vec![0.0f32; self.config.vocab_size];
        let mut logits = self.device.gpu.sync_reclaim(self.state.logits.clone()).unwrap();
        // unsafe { let _ = memcpy_dtoh_sync(&mut logits, self.state.logits); };
        if temperature == 0.0 {
            // greedy decoding, choose argmax
            next = logits.iter().enumerate()
                .reduce(|(i1, v1), (i2, v2)| if v1 > v2 { (i1, v1) } else { (i2, v2) })
                .map(|(i, _)| i).unwrap();
        } else {
            // temperature scaling
            if temperature < 1.0 {
                logits.iter_mut().for_each(|z| *z /= temperature);
            }
            // compute probabilities
            let cpu = CPU {};
            cpu.softmax(&mut logits, 0);
            // next = sample(&transformer.state.logits, &mut rng);
            next = sample_top_q(&logits, self.config.vocab_size, temperature, &mut rng);

        }
        next
    }
}

fn sample_top_q(probabilities: &Vec<f32>, num: usize, topp: f32, rng: &mut ChaCha20Rng) -> usize {
    let cutoff = (1.0f32 - topp) / ((num - 1) as f32);
    let mut prob_index = probabilities.iter().enumerate().filter(
        |(_, &p)| p > cutoff
    ).collect::<Vec<(usize, &f32)>>();
    prob_index.sort_by(
        |(_, &a2), (_, &b2)|
        b2.partial_cmp(&a2).unwrap()
    );

    let mut cum_prob = 0.0f32;
    let mut last_index = prob_index.len() - 1;
    for i in 0..prob_index.len() {
        cum_prob += prob_index[i].1;
        if cum_prob > topp {
            last_index = i;
            break;
        }
    }

    let r = rng.gen::<f32>() * cum_prob;
    let mut cdf = 0.0f32;

    for i in 0..last_index {
        cdf += prob_index[i].1;
        if r < cdf {
            return prob_index[i].0;
        }
    }

    return prob_index[last_index].0;


}