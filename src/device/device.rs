pub trait Device<MT, T, T2> {
    fn matmul_1d(&self, o: MT, w: T, x: T2, n: usize);
    fn matmul(&self, o: MT, a: T, b: T2,
        width: usize, o_rows: usize, o_cols: usize);
    fn rmsnorm(&self, o:MT, x: T, weight: T2, n: usize);
    fn softmax(&self, x: MT, n: usize);
}