use cudarc::driver::DevicePtr;
use cudarc::driver::result::memcpy_dtoh_sync;
use cudarc::driver::sys::CUdeviceptr;
use device::gpu::GPU;
pub struct RunStateGPU {
    x: CUdeviceptr,
    xb: CUdeviceptr,
    xb2: CUdeviceptr,
    hb: CUdeviceptr,
    hb2: CUdeviceptr,
    q: CUdeviceptr,
    k: CUdeviceptr,
    v: CUdeviceptr,
    att: CUdeviceptr,
    logits: CUdeviceptr,
    key_cache: CUdeviceptr,
    value_cache: CUdeviceptr,
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
}


#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TransformerWeightsGPU {
    token_embedding_table: CUdeviceptr,
    rms_att_weight: CUdeviceptr,
    rms_ffn_weight: CUdeviceptr,

    wq: CUdeviceptr,
    wk: CUdeviceptr,
    wv: CUdeviceptr,
    wo: CUdeviceptr,
    w1: CUdeviceptr,
    w2: CUdeviceptr,
    w3: CUdeviceptr,

    rms_final_weight: CUdeviceptr,
    freq_cis_real: CUdeviceptr,
    freq_cis_imag: CUdeviceptr,
    wcls_exists: bool,
    wcls: CUdeviceptr,
}

// Allocate data in GPU memory and return a pointer to the location. Leak this object since the
// lifecycle of the GPU object is the same as the local CudaSlice object.
fn allocate(gpu: &GPU, data: &Vec<f32>) -> CUdeviceptr {
    let cs = gpu.gpu.htod_sync_copy(&data).unwrap();
    let ptr = *cs.device_ptr();
    std::mem::forget(cs);
    ptr
}

impl TransformerWeightsGPU {
    fn from_hw(tw: &TransformerWeights, device: &GPU) -> Self {
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
        let mut wcls: CUdeviceptr = 0;
        let mut wcls_exists= false;

        match &tw.wcls {
            Some(val) =>
            {
                wcls = allocate(device, &val);
                wcls_exists = true;
            }
            None => {wcls = token_embedding_table;},
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

pub fn forward_gpu(transformer: &mut Transformer, gpu_weights: &mut TransformerWeightsGPU,
    gpu_state: &mut RunStateGPU, gpu: &GPU, token: usize, pos: usize) {

    let cfg = &transformer.config;
    let dim = cfg.dim;
    let hidden_dim = cfg.hidden_dim;
    let head_size = dim / cfg.n_heads;


    let mut buf1 = vec![0.0f32;9216000];
    gpu.debug(&mut buf1, gpu_weights.token_embedding_table);

    gpu.copy_from_slice(gpu_weights.token_embedding_table + (token * dim * FLOAT_SIZE) as u64, gpu_state.x, dim as i32);

    for layer in 0..cfg.n_layers {

        let mut v = vec![0.0f32;cfg.dim];
        let mut q = vec![0.0f32;cfg.dim];
        let mut k: Vec<f32> = vec![0.0f32;cfg.dim];
        let mut xb = vec![0.0f32;cfg.dim];
        let mut x = vec![0.0f32;cfg.dim];
        let mut rms_att_weight = vec![0.0f32;cfg.dim * cfg.n_layers];
        let mut k_cache = vec![0.0f32;cfg.n_layers * cfg.seq_len * cfg.dim];
        let mut v_cache = vec![0.0f32;cfg.n_layers * cfg.seq_len * cfg.dim];

        let _ = &gpu.rmsnorm(gpu_state.xb, gpu_state.x, gpu_weights.rms_att_weight + (layer * dim * FLOAT_SIZE) as u64, 0, dim as i32);

        // let _ = &gpu.rmsnorm(gpu_state.xb, gpu_state.x, gpu_weights.rms_att_weight, (layer * dim) as i32, dim as i32);

        gpu.debug(&mut x, gpu_state.x);
        gpu.debug(&mut xb, gpu_state.xb);
        gpu.debug(&mut rms_att_weight, gpu_weights.rms_att_weight  as u64);

        let _ = &gpu.matmul2(gpu_state.q, gpu_weights.wq + (layer * dim * dim * FLOAT_SIZE) as u64, gpu_state.xb, dim, dim as i32, 1);

        gpu.debug(&mut q, gpu_state.q);

        let _ = &gpu.matmul2(gpu_state.k, gpu_weights.wk + (layer * dim * dim * FLOAT_SIZE) as u64, gpu_state.xb, dim, dim as i32, 1);

        gpu.debug(&mut k, gpu_state.k);

        let _ = &gpu.matmul2(gpu_state.v, gpu_weights.wv + (layer * dim * dim * FLOAT_SIZE) as u64, gpu_state.xb, dim, dim as i32, 1);

        gpu.debug(&mut v, gpu_state.v);


        for h in 0..cfg.n_heads {

            let q = gpu_state.q + (h * head_size) as u64;
            let k = gpu_state.k + (h * head_size) as u64;
            let _ = &gpu.apply_position(q, k, gpu_weights.freq_cis_real, gpu_weights.freq_cis_imag, cfg.n_heads as i32, head_size as i32);
        }

        gpu.debug(&mut v, gpu_state.v);
        gpu.debug(&mut q, gpu_state.q as u64);
        gpu.debug(&mut k, gpu_state.k as u64);
        gpu.debug(&mut xb, gpu_state.xb);

        // -- %%
        let lo = layer * cfg.seq_len * dim;
        let _ = &gpu.copy_from_slice(gpu_state.k, gpu_state.key_cache + ((lo + pos * dim) * FLOAT_SIZE) as u64, dim as i32);
        let _ = &gpu.copy_from_slice(gpu_state.v, gpu_state.value_cache + ((lo + pos * dim) * FLOAT_SIZE) as u64, dim as i32);

        gpu.debug(&mut k_cache, gpu_state.key_cache);
        gpu.debug(&mut v_cache, gpu_state.value_cache);


        if layer == 1 {
            print!("");
        }
        let _ = &gpu.multi_head_attention(gpu_state, cfg, layer, pos);

        // ---
        let mut att = vec![0.0f32;cfg.n_heads * cfg.seq_len];
        gpu.debug(&mut att, gpu_state.att);
        let mut xb = vec![0.0f32;cfg.dim];
        gpu.debug(&mut xb, gpu_state.xb);

        //----
        // > xb broken after 96
        let _ = &gpu.matmul2(gpu_state.xb2, gpu_weights.wo + (layer * dim * dim * FLOAT_SIZE) as u64, gpu_state.xb, dim, dim as i32, 1);
        // let _ = &gpu.matmul2(gpu_state.v, gpu_weights.wv + (layer * dim * dim) as u64, gpu_state.xb, dim, dim as i32, 1);

        let mut mainxb = vec![0.0f32;cfg.dim];


        let mut xb2 = vec![0.0f32;cfg.dim];
        gpu.debug(&mut xb2, gpu_state.xb2);
        let mut wo = vec![0.0f32;cfg.dim * cfg.dim];
        gpu.debug(&mut wo, gpu_weights.wo + (layer * dim * dim) as u64);
        let mut xb: Vec<f32> = vec![0.0f32;cfg.dim];
        gpu.debug(&mut xb, gpu_state.xb);

        // matmul(&mut mainxb, &wo, &xb, dim);



        let _ = &gpu.array_add(gpu_state.x, gpu_state.xb2, dim);

        let mut buf1 = vec![0.0f32;cfg.dim];
        gpu.debug(&mut buf1, gpu_state.x);
//------
        let _ = &gpu.rmsnorm(gpu_state.xb, gpu_state.x, gpu_weights.rms_ffn_weight, (layer * dim) as i32, dim as i32);

        let mut xb = vec![0.0f32;cfg.dim];
        gpu.debug(&mut xb, gpu_state.xb);

        let _ = &gpu.matmul2(gpu_state.hb, gpu_weights.w1 + (layer * hidden_dim * dim * FLOAT_SIZE) as u64, gpu_state.xb, dim, hidden_dim as i32, 1);

        let _ = &gpu.matmul2(gpu_state.hb2, gpu_weights.w3 + (layer * hidden_dim * dim * FLOAT_SIZE) as u64, gpu_state.xb, dim, hidden_dim as i32, 1);

        let mut hb = vec![0.0f32;cfg.dim];
        gpu.debug(&mut hb, gpu_state.hb);
        let mut buf2 = vec![0.0f32;cfg.dim];
        let mut buf3 = vec![0.0f32;cfg.dim];

        gpu.debug(&mut buf2, gpu_state.hb2);
        gpu.debug(&mut buf3, gpu_state.xb);


        //---
        let _ = &gpu.sinu(gpu_state.hb, hidden_dim as i32);

        let mut hb = vec![0.0f32;cfg.dim];
        gpu.debug(&mut hb, gpu_state.hb);

        let _ = &gpu.array_mult(gpu_state.hb, gpu_state.hb2, hidden_dim as i32);

        let mut hb = vec![0.0f32;cfg.dim];
        gpu.debug(&mut hb, gpu_state.hb);

        let _ = &gpu.matmul2(gpu_state.xb, gpu_weights.w2 + (layer * dim * hidden_dim * FLOAT_SIZE) as u64, gpu_state.hb, hidden_dim, dim as i32, 1);

        let _  = &gpu.array_add(gpu_state.x, gpu_state.xb, dim);

        let mut key_cache = vec![0.0f32;cfg.dim];
        let mut x = vec![0.0f32;cfg.dim];
        let mut xb = vec![0.0f32;cfg.dim];
        gpu.debug(&mut key_cache, gpu_state.key_cache);
        gpu.debug(&mut x, gpu_state.x);
        gpu.debug(&mut xb, gpu_state.xb);
        // unsafe { let _ = memcpy_dtoh_sync(& mut buf2, gpu_state.xb); };
        print!("");
    }

    let mut buf1 = vec![0.0f32;dim];
    gpu.debug(&mut buf1, gpu_state.x);

    let _ = &gpu.copy_from_slice(gpu_state.x, gpu_state.xb, dim as i32);

    let mut buf1 = vec![0.0f32;dim];
    gpu.debug(&mut buf1, gpu_state.x);

    let _ = &gpu.rmsnorm(gpu_state.x, gpu_state.xb, gpu_weights.rms_final_weight, 0, dim as i32);

    let mut x = vec![0.0f32;dim];
    gpu.debug(&mut x, gpu_state.x);
    let mut xb = vec![0.0f32;dim];
    gpu.debug(&mut xb, gpu_state.xb);
    let mut rms_final_weight = vec![0.0f32;dim];
    gpu.debug(&mut rms_final_weight, gpu_weights.rms_final_weight);

    let mut mainbuf4 = vec![0.0f32;dim];
    rmsnorm(&mut mainbuf4, &xb, &rms_final_weight);

    let _ = &gpu.matmul2(gpu_state.logits, gpu_weights.wcls, gpu_state.x, dim, cfg.vocab_size as i32, 1);

    let mut logits = vec![0.0f32;dim];
    gpu.debug(&mut logits, gpu_state.logits);
    let mut buf3 = vec![0.0f32;dim];
    gpu.debug(&mut buf3, gpu_state.x);

    print!("");
}