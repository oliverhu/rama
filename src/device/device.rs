pub trait Device {
    type Err;
    // device must implement matmul that returns a result.
    fn matmul(o: &mut [f32], a: &[f32], b: &[f32],
        width: usize, o_rows: usize, o_cols: usize)
            -> Result<(), Self::Err>;
}