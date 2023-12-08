use rayon::prelude::*;

use super::device::*;
pub struct CPU {}
impl Device for CPU {
    type Err = ();
    fn matmul(o: &mut [f32], a: &[f32], b: &[f32], width: usize, _o_rows: usize, o_cols: usize) -> Result<(), ()> {
        // for r in 0..o_rows {
        //     for c in 0..o_cols {
        //         let mut v = 0.0;
        //         for k in 0..width {
        //             v += a[r * width + k] * b[k * o_cols + c];
        //         }
        //         o[r * o_cols + c] = v;
        //     }
        // }

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

        Ok(())

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_mul() {
        let a_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c_host = [0.0f32; 4];
        let _ = CPU::matmul(&mut c_host, &a_host, &b_host, 3, 2, 2);
        assert_eq!(c_host, [22.0, 28.0, 49.0, 64.0]);

    }

    // fn test_softmax() {}
    // fn test_rmsnorm() {}
}