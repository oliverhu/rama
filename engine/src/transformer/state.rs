use super::{MutView, Storage, View};

/// RunState has two generic variables, T & Q. T is
/// used for storing f32 related storage types, Q is
/// used to store u8/u4 types to represent the values.
#[derive(Default)]
pub struct RunState<T: Storage, Q: Storage> {
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
    pub xq: Q, // quantized x
    pub hq: Q, // quantized h
}

pub struct RunStateView<'a, T: Storage, Q: Storage> {
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
    pub xq: MutView<'a, Q>, // quantized x
    pub hq: MutView<'a, Q>, // quantized h
}

impl<'a, T: Storage, Q: Storage> RunStateView<'a, T, Q> {
    pub fn from_rs(rs: &mut RunState<T, Q>) -> RunStateView<'_, T, Q> {
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
            xq: MutView::new(&mut rs.xq),
            hq: MutView::new(&mut rs.hq),
        }
    }
}

// Transformer Weights
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
    pub wcls_exists: bool,
    pub wcls: View<'a, T>,
}

impl<'a> TransformerWeightsView<'a, Vec<f32>> {
    #[allow(dead_code)]
    pub fn from_ws(ws: &'a TransformerWeights<Vec<f32>>) -> TransformerWeightsView<'a, Vec<f32>> {
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
                    View::new(&ws.wcls)
                } else {
                    View::new(&ws.token_embedding_table)
                }
            },
            wcls_exists: ws.wcls_exists,
        }

    }
}
