pub trait Device<MT, T> {
    fn matmul_1d(&self, o: MT, w: T, x: T, n: usize);
    fn matmul(&self, o: MT, a: T, b: T,
        width: usize, o_rows: usize, o_cols: usize);
    fn rmsnorm(&self, o:MT, x: T, weight: T);
    fn softmax(&self, x: MT);
}