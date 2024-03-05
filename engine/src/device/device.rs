use crate::transformer::{state::{RunState, RunStateView}, Config, MutView, Storage, View};

pub trait Device<T: Storage> {
    fn array_add(&self, target: &mut MutView<'_, T>, source: &View<'_, T>, n: usize);
    fn array_mult(&self, target: &mut MutView<'_, T>, source: &View<'_, T>, n: usize);
    fn sinu(&self, o: &mut MutView<'_, T>, n: usize);
    fn multi_head_attention(&self, rsv: &mut RunStateView<'_, T>,
            cfg: &Config, layer: usize, pos: usize);
    fn copy_from_slice(&self, target: &mut MutView<'_, T>, source: &View<'_, T>, n: usize);
    fn rmsnorm(&self, o: &mut MutView<'_, T>, x: &View<'_, T>,
                        weight: &View<'_, T>, n: usize);
    fn apply_position(&self, q: &mut MutView<'_, T>, k: &mut MutView<'_, T>, pos_real: &View<'_, T>, pos_img: &View<'_, T>, head_size: usize);
    fn matmul_1d(&self, o: &mut MutView<'_, T>, w: &View<'_, T>, x: &View<'_, T>, n: usize);
    fn matmul(&self, o: &mut MutView<'_, T>, a: &View<'_, T>, b: &View<'_, T>, width: usize, o_rows: usize, o_cols: usize);
    fn softmax<'a>(&self, x: &mut MutView<'a, T>, n: usize);

    fn sample<'a>(&self, cfg: &Config, rsv: &mut RunStateView<'a, T>, temperature: f32, topp: f32) -> usize;

    // debugging related.

    // copy current state to a run state view container for debugging.
    fn to_cpu(&self, state: &RunStateView<T>, cpu_state: &mut RunState<Vec<f32>>);


}