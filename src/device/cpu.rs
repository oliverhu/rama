use rayon::prelude::*;

use super::device::*;
pub struct CPU {}

impl Device<&mut Vec<f32>, &Vec<f32>> for CPU {
    fn matmul_1d(&self, o: &mut Vec<f32>, w: &Vec<f32>, x: &Vec<f32>, n: usize) {
        let le = o.len();
        let _ = self.matmul(o, w, &x, n, le, 1);
    }

    fn rmsnorm(&self, o: &mut Vec<f32>, x: &Vec<f32>, weight: &Vec<f32>) {
        let v: f32 =
        1.0f32 /
        (x.iter().map(|x| x * x ).sum::<f32>() / x.len() as f32 + 1e-5f32)
        .sqrt();
        for i in 0..o.len() {
            o[i] = weight[i] * (v * x[i]);
        }
    }

    fn softmax(&self, x: &mut Vec<f32>) {
        let max = x.par_iter().copied().reduce(|| x[0], |a, b| a.max(b));
        x.par_iter_mut().for_each(|a| *a=(*a-max).exp());
        let sum = x.par_iter().sum::<f32>();
        x.par_iter_mut().for_each(|a| *a /= sum);
    }

    fn matmul(&self, o: &mut Vec<f32>, a: &Vec<f32>, b: &Vec<f32>, width: usize, _o_rows: usize, o_cols: usize) {

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_mul() {
        let a_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c_host = vec![0.0f32; 4];
        let cpu = CPU {};
        let _ = &cpu.matmul(&mut c_host, &a_host, &b_host, 3, 2, 2);
        assert_eq!(c_host, [22.0, 28.0, 49.0, 64.0]);

    }

}