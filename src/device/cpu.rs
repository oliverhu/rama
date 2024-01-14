use rayon::prelude::*;

use crate::transformer::{CPUView, CPUMutView};

pub struct CPU {}

// impl Device<&CPUView, &CPUView, &CPUView> for CPU {
impl CPU {
    pub fn matmul_1d(&self, o: &mut CPUMutView, w: &CPUView, x: &CPUView, n: usize) {
        let le = o.as_mut().len();
        let _ = self.matmul(o, w, &x, n, le, 1);
    }

    pub fn rmsnorm(&self, o: &mut CPUMutView, x: &CPUView, weight: &CPUView, _n: usize) {
        let x = x.as_ref();
        let o = o.as_mut();
        let weight = weight.as_ref();

        let v: f32 =
        1.0f32 /
        (x.iter().map(|x| x * x ).sum::<f32>() / x.len() as f32 + 1e-5f32)
        .sqrt();
        for i in 0..o.len() {
            o[i] = weight[i] * (v * x[i]);
        }
    }

    pub fn softmax(&self, x: &mut CPUMutView, _n: usize) {
        let x = x.as_mut();
        let max = x.par_iter().copied().reduce(|| x[0], |a, b| a.max(b));
        x.par_iter_mut().for_each(|a| *a=(*a-max).exp());
        let sum = x.par_iter().sum::<f32>();
        x.par_iter_mut().for_each(|a| *a /= sum);
    }

    pub fn softmax_num(&self, x: &mut [f32], _n: usize) {
        let max = x.par_iter().copied().reduce(|| x[0], |a, b| a.max(b));
        x.par_iter_mut().for_each(|a| *a=(*a-max).exp());
        let sum = x.par_iter().sum::<f32>();
        x.par_iter_mut().for_each(|a| *a /= sum);
    }

    pub fn matmul(&self, o: &mut CPUMutView, a: &CPUView, b: &CPUView, width: usize, _o_rows: usize, o_cols: usize) {
        let o = o.as_mut();
        let a = a.as_ref();
        let b = b.as_ref();
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