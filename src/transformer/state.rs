use std::{fs::File, io::BufReader};

use cudarc::driver::CudaSlice;

use crate::device::gpu::GPU;

use super::{read_vec, Config, MutView, Storage, View};


#[derive(Default)]
pub struct RunState<T: Storage> {
    pub x: T,
    pub xb: T,
    pub xb2: T,
    pub hb: T,
    pub hb2: T,
    pub q: T,
    pub k: T,
    pub v: T,
    pub att: T,
    pub logits: T,
    pub key_cache: T,
    pub value_cache: T,
}

pub struct RunStateView<'a, T: Storage> {
    pub x: MutView<'a, T>,
    pub xb: MutView<'a, T>,
    pub xb2: MutView<'a, T>,
    pub hb: MutView<'a, T>,
    pub hb2: MutView<'a, T>,
    pub q: MutView<'a, T>,
    pub k: MutView<'a, T>,
    pub v: MutView<'a, T>,
    pub att: MutView<'a, T>,
    pub logits: MutView<'a, T>,
    pub key_cache: MutView<'a, T>,
    pub value_cache: MutView<'a, T>,
}

impl<'a, T: Storage> RunStateView<'a, T> {
    pub fn from_rs(rs: &mut RunState<T>) -> RunStateView<'_, T> {
        RunStateView {
            x: MutView::new(&mut rs.x),
            xb: MutView::new(&mut rs.xb),
            xb2: MutView::new(&mut rs.xb2),
            hb: MutView::new(&mut rs.hb),
            hb2: MutView::new(&mut rs.hb2),
            q: MutView::new(&mut rs.q),
            k: MutView::new(&mut rs.k),
            v: MutView::new(&mut rs.v),
            att: MutView::new(&mut rs.att),
            logits: MutView::new(&mut rs.logits),
            key_cache: MutView::new(&mut rs.key_cache),
            value_cache: MutView::new(&mut rs.value_cache),
        }
    }
}

impl RunState<Vec<f32>> {
    pub fn from_config(cfg: &Config) -> Self {
        let kv_dim = cfg.dim * cfg.n_kv_heads / cfg.n_heads;
        Self {
            x: vec![0.0; cfg.dim as usize],
            xb: vec![0.0; cfg.dim as usize],
            xb2: vec![0.0; cfg.dim as usize],
            hb: vec![0.0; cfg.hidden_dim as usize],
            hb2: vec![0.0; cfg.hidden_dim as usize],
            q: vec![0.0; cfg.dim as usize],
            k: vec![0.0; cfg.dim as usize],
            v: vec![0.0; cfg.dim as usize],
            att: vec![0.0; cfg.n_heads * cfg.seq_len as usize],
            logits: vec![0.0; cfg.vocab_size as usize],
            key_cache: vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim as usize],
            value_cache: vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim as usize],
        }
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


// Transformer Weights

#[repr(C)]
#[derive(Debug, Clone)]
pub struct TransformerWeights<T: Storage> {
    pub token_embedding_table: T,
    pub rms_att_weight: T,
    pub rms_ffn_weight: T,

    pub wq: T,
    pub wk: T,
    pub wv: T,
    pub wo: T,
    pub w1: T,
    pub w2: T,
    pub w3: T,

    pub rms_final_weight: T,
    pub freq_cis_real: T,
    pub freq_cis_imag: T,
    pub wcls_exists: bool,
    pub wcls: T,
}


impl TransformerWeights<CudaSlice<f32>> {
    pub fn from_weight(tw: &mut TransformerWeights, device: &GPU) -> Self {
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
        let wcls: CudaSlice<f32>;
        let mut wcls_exists= false;

        match &tw.wcls {
            Some(val) =>
            {
                wcls = allocate(device, &val.as_ref());
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

pub struct TransformerWeightsView<'a, T: Storage> {
    pub token_embedding_table: View<'a, T>,
    pub rms_att_weight:  View<'a, T>,
    pub rms_ffn_weight: View<'a, T>,
    pub wq: View<'a, T>,
    pub wk: View<'a, T>,
    pub wv: View<'a, T>,
    pub wo: View<'a, T>,
    pub w1: View<'a, T>,
    pub w2: View<'a, T>,
    pub w3: View<'a, T>,
    pub rms_final_weight:View<'a, T>,
    pub freq_cis_real: View<'a, T>,
    pub freq_cis_imag: View<'a, T>,
    pub wcls: Option<View<'a, T>>,
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
            wcls: {
                if ws.wcls_exists {
                  Some(View::new(&ws.wcls))
                } else {
                  None
                }
            },
            // wcls: Some(View::new(&ws.freq_cis_imag)),
        }

    }
}

impl<'a, 'b: 'a> TransformerWeightsView<'a, Vec<f32>> {

    pub fn from_ws(ws: &'a TransformerWeights) -> TransformerWeightsView<'a, Vec<f32>> {
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
            wcls: {
                match &ws.wcls {
                  None => None,
                  Some(wcls) => Some(View::new(wcls)),
                }
                // { None } else {
                //     Some(read_vec::<f32>(f, c.vocab_size * c.dim))
                // }
            },
            // wcls: Some(View::new(&ws.freq_cis_imag)),
        }

    }
}

impl TransformerWeights {
    fn from_file(f: &mut BufReader<File>, c: &Config) -> Self {
        let head_size = c.dim / c.n_heads;
        Self {
            token_embedding_table: read_vec(f, c.vocab_size * c.dim),
            rms_att_weight: read_vec(f, c.n_layers * c.dim),
            wq: read_vec(f, c.n_layers * c.dim * c.dim),
            wk: read_vec(f, c.n_layers * c.dim * c.dim),
            wv: read_vec(f, c.n_layers * c.dim * c.dim),
            wo: read_vec(f, c.n_layers * c.dim * c.dim),
            rms_ffn_weight: read_vec(f, c.n_layers * c.dim),
            w1: read_vec(f, c.n_layers * c.dim * c.hidden_dim),
            w2: read_vec(f, c.n_layers * c.dim * c.hidden_dim),
            w3: read_vec(f, c.n_layers * c.dim * c.hidden_dim),
            rms_final_weight: read_vec(f, c.dim),
            freq_cis_real: read_vec(f, c.seq_len * head_size / 2),
            freq_cis_imag: read_vec(f, c.seq_len * head_size / 2),
            wcls: {
                if c.shared_weight { None } else {
                    Some(read_vec::<f32>(f, c.vocab_size * c.dim))
                }
            },
        }
    }

}