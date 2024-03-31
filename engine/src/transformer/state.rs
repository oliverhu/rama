use super::{MutView, Storage, View};

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

// Transformer Weights. For RMS, we will stick with FP32 while
// the rest of the model weights can be quantized in opted in.
// In unquantized mode, T is Vec<f32> and each number represents a digit.
// With quantization, Q is QuantizedTensor of size_of_each * Vec<i8>
// and size_of_each / group_size * Vec<f32>
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TransformerWeights<T: Storage, Q: Storage> {
    pub token_embedding_table: T,
    pub q_token: Q,
    pub rms_att_weight: T,
    pub rms_ffn_weight: T,

    pub wq: Q,
    pub wk: Q,
    pub wv: Q,
    pub wo: Q,
    pub w1: Q,
    pub w2: Q,
    pub w3: Q,

    pub rms_final_weight: T,
    pub freq_cis_real: T,
    pub freq_cis_imag: T,
    pub wcls_exists: bool,
    pub wcls: Q,
}

pub struct TransformerWeightsView<'a, T: Storage, Q: Storage> {
    pub token_embedding_table: View<'a, T>,
    pub q_token: View<'a, Q>,
    pub rms_att_weight:  View<'a, T>,
    pub rms_ffn_weight: View<'a, T>,
    pub wq: View<'a, Q>,
    pub wk: View<'a, Q>,
    pub wv: View<'a, Q>,
    pub wo: View<'a, Q>,
    pub w1: View<'a, Q>,
    pub w2: View<'a, Q>,
    pub w3: View<'a, Q>,
    pub rms_final_weight:View<'a, T>,
    pub freq_cis_real: View<'a, T>,
    pub freq_cis_imag: View<'a, T>,
    pub wcls_exists: bool,
    pub wcls: View<'a, Q>,
}

impl<'a, T> TransformerWeightsView<'a, Vec<f32>, Vec<T>> where Vec<T>: Storage {
    #[allow(dead_code)]
    pub fn from_ws(ws: &'a TransformerWeights<Vec<f32>, Vec<T>>) -> TransformerWeightsView<'a, Vec<f32>, Vec<T>> {
        TransformerWeightsView {
            token_embedding_table: View::new(&ws.token_embedding_table),
            q_token: View::new(&ws.q_token),
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
                    View::new(&ws.wcls)
                } else {
                    View::new(&ws.q_token)
                }
            },
            wcls_exists: ws.wcls_exists,
        }

    }
}
