use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::transformer::{infer::sample_top_q, state::{RunState, RunStateView}, Config, MutView, Storage, View};
use crate::transformer::ram_q80::QuantizedTensor;
use wide::f32x4;

use super::device::{Device, QuantDevice};

#[derive(Debug)]
pub struct CPU {}

const GS: usize = 2;
impl QuantDevice<Vec<f32>, Vec<QuantizedTensor>> for CPU {
    fn matmul_q(&self, o: &mut MutView<'_, Vec<f32>>, a: &View<'_, Vec<QuantizedTensor>>, b: &View<'_, Vec<QuantizedTensor>>, width: usize, o_rows: usize, o_cols: usize) {
        let or = o.range.clone();
        let o = &mut o.as_mut()[or];

        // Since we only slice by layers.
        let aq = &a.as_ref()[a.range.clone()][0];
        let bq = &b.as_ref()[b.range.clone()][0];

        o.iter_mut().enumerate().for_each(
            |(idx, o)| {
                let r = idx / o_cols;
                let c = idx % o_cols;
                // Reset to the naive impl of matmul for easier debugging. Calculate the flattened position for value & scale.
                let mut v = 0f32;

                for k in 0..width {
                    let pre_scale = aq.q[r * width + k] * bq.q[k * o_cols + c];
                    v += pre_scale as f32 * aq.s[(r * width + k) / GS] * bq.s[(k * o_cols + c) / GS];
                }
                *o = v as f32;
            }

        );
    }

    // Quantize function is only used for x -> xq and h -> hq, so we kinda only cares about
    // one single layer, so we always get the first element out.
    fn quantize(&self, o: &mut MutView<'_, Vec<QuantizedTensor>>, a: &View<'_, Vec<f32>>, n: usize) {
        let num_groups = n / GS;
        let q_max = 127.0f32;
        for group in 0..num_groups {
            let mut wmax = 0.0f32;
            for i in 0..GS {
                let val = f32::abs(a.data[group * GS + i]);
                if val > wmax {
                    wmax = val;
                }
            }
            let scale = wmax / q_max;
            let qx = &mut o.data[0];
            let ax = a.data;
            for i in 0..GS {
                let quant_value = ax[group * (GS as usize) + i] / scale;
                let quantized = quant_value.round() as i8;
                qx.q[group * GS + i] = quantized;
            }
        }



    }
}

#[test]
fn test_matmul() {

    // Test quantized matmul
    let uqt = vec![QuantizedTensor { q: vec![1, 2, 3, 4], s: vec![1.0, 1.0] }];
    let uqt_view = View::new(&uqt);
    let uqt_view2 = View::new(&uqt);

    let qt = vec![QuantizedTensor { q: vec![1, 2, 3, 4], s: vec![0.5, 0.5] }];
    let qt_view = View::new(&qt);
    let qt_2 = qt.clone();
    let qt_view_2 = View::new(&qt_2);

    let mut result = vec![1.0f32, 2.0f32, 3.0, 4.0];
    let mut qt_mut = MutView::new(&mut result);

    let cpu = CPU {};
    cpu.matmul_q(&mut qt_mut, &qt_view, &qt_view_2, 2, 2, 2);

    assert_eq!(*qt_mut.data, vec![1.75f32, 2.5, 3.75, 5.5]);

    cpu.matmul_q(&mut qt_mut, &uqt_view, &uqt_view2, 2, 2, 2);
    assert_eq!(*qt_mut.data, vec![7.0f32, 10.0, 15.0, 22.0]);

    // Test normal matmul, put them together right now for the same of comparing.
    let qt = vec![1.0f32, 2.0, 3.0, 4.0];
    let qt_view = View::new(&qt);
    let qt_2 = vec![1.0f32, 2.0, 3.0, 4.0];
    let qt2_view = View::new(&qt_2);
    cpu.matmul_test(&mut qt_mut, &qt_view, &qt2_view, 2, 2, 2);
    println!("{:?}", *qt_mut.data);
}

impl Device<Vec<f32>, Vec<QuantizedTensor>> for CPU {

    fn array_add(&self, target: &mut MutView<'_, Vec<f32>>, source: &View<'_, Vec<f32>>, _n: usize) {
        let s_range = source.range.clone();
        let t_range = target.range.clone();
        target.as_mut()[t_range].iter_mut().zip(source.as_ref()[s_range].iter())
        .for_each(|(a, b)| *a += *b);
    }

    fn multi_head_attention(&self, rsv: &mut RunStateView<'_, Vec<f32>, Vec<QuantizedTensor>>,
                                cfg: &Config,
                                layer: usize,
                                pos: usize,) {
        let head_size = cfg.dim / cfg.n_heads;
        let lo = layer * cfg.seq_len * cfg.dim;
        let mut atts: Vec<&mut [f32]> = rsv.att.as_mut().chunks_mut(cfg.seq_len).collect();
        let qs: Vec<&mut [f32]> = rsv.q.as_mut().chunks_mut(head_size).collect();
        let xbs: Vec<&mut [f32]> = rsv.xb.as_mut().chunks_mut(head_size).collect();
        atts.par_iter_mut().zip(xbs).enumerate().for_each(|(h, (att,xb))| {
            let q = &qs[h];
            for t in 0..(pos + 1) {
                let ko = lo + t * cfg.dim + h * head_size;
                let bind = &rsv.key_cache.slice(ko..(ko + head_size));
                let k = &bind.as_ref()[ko..(ko + head_size)];
                att[t] = q.iter().zip(k.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>() / (head_size as f32).sqrt();
            }
            self.softmax_num(&mut att[..(pos + 1)], pos + 1);
            xb.fill(0.0);
            for t in 0..(pos + 1) {
                let ko = lo + t * cfg.dim + h * head_size;
                let v = rsv.value_cache.slice(ko..(ko + head_size));
                let a = att[t];
                xb.iter_mut().zip(&v.as_ref()[ko..(ko + head_size)]).for_each(|(xbi, &vi)| *xbi += a * vi);
            }

        });
    }

    fn sinu(&self, o: &mut MutView<'_, Vec<f32>>, _n: usize) {
        let or = o.range.clone();
        o.as_mut()[or].par_iter_mut().for_each(|a|*a = *a * (1.0 / (1.0 + (-*a).exp())));
    }

    fn array_mult(&self, target: &mut MutView<'_, Vec<f32>>, source: &View<'_, Vec<f32>>, _n: usize) {
        let s_range = source.range.clone();
        let t_range = target.range.clone();
        target.as_mut()[t_range].iter_mut().zip(source.as_ref()[s_range].iter())
        .for_each(|(a, b)| *a *= *b);
    }

    fn copy_from_slice(&self, target: &mut MutView<'_, Vec<f32>>, source: &View<'_, Vec<f32>>, _n: usize) {
        let s_range = source.range.clone();
        let t_range = target.range.clone();
        target.as_mut()[t_range].copy_from_slice(
          &source.as_ref()[s_range]
        );
    }

    fn apply_position(&self, q: &mut MutView<'_, Vec<f32>>, k: &mut MutView<'_, Vec<f32>>, pos_real: &View<'_, Vec<f32>>, pos_img: &View<'_, Vec<f32>>, head_size: usize) {
        let qrc = q.range.clone();
        let qr = &mut q.as_mut()[qrc];

        let krc = k.range.clone();
        let kr = &mut k.as_mut()[krc];

        let prrc = pos_real.range.clone();
        let prr = &pos_real.as_ref()[prrc];

        let pirc = pos_img.range.clone();
        let pir = &pos_img.as_ref()[pirc];

        for i in 0..(head_size / 2) {
            // Instead of doing matmul, this is a more simplified way (from the paper above)
            let (fcr, fci) = (prr[i], pir[i]);
            (qr[i * 2], qr[i * 2 + 1]) = (
                qr[i * 2] * fcr - qr[i * 2 + 1] * fci,
                qr[i * 2] * fci + qr[i * 2 + 1] * fcr);
            (kr[i * 2], kr[i * 2 + 1]) = (
                kr[i * 2] * fcr - kr[i * 2 + 1] * fci,
                kr[i * 2] * fci + kr[i * 2 + 1] * fcr);
        }
    }

    fn rmsnorm(&self, o: &mut MutView<'_, Vec<f32>>, x: &View<'_, Vec<f32>>,
                        weight: &View<'_, Vec<f32>>, _n: usize) {
        let xr = x.range.clone();
        let x = &x.as_ref()[xr];

        let or = o.range.clone();
        let o = &mut o.as_mut()[or];

        let wr = weight.range.clone();
        let weight = &weight.as_ref()[wr];

        let v: f32 =
        1.0f32 /
        (x.iter().map(|x| x * x ).sum::<f32>() / x.len() as f32 + 1e-5f32)
        .sqrt();
        for i in 0..o.len() {
            o[i] = weight[i] * (v * x[i]);
        }
    }

    fn softmax<'a>(&self, x: &mut MutView<'a, Vec<f32>>, _n: usize) {
        let x = x.as_mut();
        let max = x.par_iter().copied().reduce(|| x[0], |a, b| a.max(b));
        x.par_iter_mut().for_each(|a| *a=(*a-max).exp());
        let sum = x.par_iter().sum::<f32>();
        x.par_iter_mut().for_each(|a| *a /= sum);
    }

    fn matmul(&self, o: &mut MutView<'_, Vec<f32>>, a: &View<'_, Vec<f32>>, b: &View<'_, Vec<f32>>, width: usize, _o_rows: usize, o_cols: usize)
    {
        let or = o.range.clone();
        let o = &mut o.as_mut()[or];

        let ar = a.range.clone();
        let a = &a.as_ref()[ar];

        let br = b.range.clone();
        let b = &b.as_ref()[br];
        o.par_iter_mut().enumerate().for_each(
            |(idx, o)| {
                let r = idx / o_cols;
                let c = idx % o_cols;
                let mut v = f32x4::splat(0.0);
                for k in (0..width).step_by(4) {
                    let a_wide = f32x4::from(&a[r * width + k..r * width + k + 4]);
                    let b_values = [b[k * o_cols + c], b[(k + 1) * o_cols + c], b[(k + 2) * o_cols + c], b[(k + 3) * o_cols + c]];
                    let b_wide = f32x4::from(&b_values[..]);
                    v += a_wide * b_wide;
                }
                *o = v.reduce_add();
            }

        );
    }

    fn sample<'a>(&self, cfg: &Config, rsv: &mut RunStateView<'a, Vec<f32>, Vec<QuantizedTensor>>, temperature: f32, topp: f32) -> usize {
        let next;

        let lr = rsv.logits.range.clone();
        let logits = &mut rsv.logits.as_mut()[lr];
        // let logits = self.rsv.logits.as_mut();
        let rng_seed = 100;
        let mut rng = ChaCha20Rng::seed_from_u64(rng_seed);
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
            self.softmax_num(logits, 0);
            // next = sample(&transformer.rsv.logits, &mut rng);
            next = sample_top_q(logits, cfg.vocab_size, topp, &mut rng);
        }
        next
    }

    fn to_cpu(&self, _state: &RunStateView<Vec<f32>, Vec<QuantizedTensor>>, _cpu_state: &mut RunState<Vec<f32>, Vec<f32>>) {
        // no need to do anything since it is already CPU.
    }
}

impl CPU {
    pub fn softmax_num(&self, x: &mut [f32], _n: usize) {
        let max = x.par_iter().copied().reduce(|| x[0], |a, b| a.max(b));
        x.par_iter_mut().for_each(|a| *a=(*a-max).exp());
        let sum = x.par_iter().sum::<f32>();
        x.par_iter_mut().for_each(|a| *a /= sum);
    }

    #[allow(dead_code)]
    fn matmul_test(&self, o: &mut MutView<'_, Vec<f32>>, a: &View<'_, Vec<f32>>, b: &View<'_, Vec<f32>>, width: usize, _o_rows: usize, o_cols: usize)
    {
        let or = o.range.clone();
        let o = &mut o.as_mut()[or];

        let ar = a.range.clone();
        let a = &a.as_ref()[ar];

        let br = b.range.clone();
        let b = &b.as_ref()[br];
        o.par_iter_mut().enumerate().for_each(
            |(idx, o)| {
                let r = idx / o_cols;
                let c = idx % o_cols;
                let mut v = 0.0;
                for k in 0..width {
                    v += a[r * width + k] * b[k * o_cols + c];
                }
                *o = v;
            }

        );
    }
}