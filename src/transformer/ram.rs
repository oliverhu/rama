use std::{io::BufReader, fs::File};
use crate::device::cpu::CPU;
use super::{state::{RunState, RunStateView, TransformerWeights, TransformerWeightsView}, Config, Transformer};
use rand_chacha::ChaCha20Rng;
use rand::{Rng, SeedableRng};
pub struct TransformerCPU {
    pub config: Config,
    pub weights: TransformerWeights<Vec<f32>>,
    pub state: RunState<Vec<f32>>,
    pub device: CPU
}

pub fn sample<'a>(cfg: &Config, rsv: &mut RunStateView<'a, Vec<f32>>, device: &CPU, temperature: f32) -> usize {
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
        device.softmax_num(logits, 0);
        // next = sample(&transformer.rsv.logits, &mut rng);
        next = sample_top_q(logits, cfg.vocab_size, 0.9, &mut rng);

    }
    next
}

impl<'a> Transformer for TransformerCPU {
    fn cpu_state(&self) -> &RunState<Vec<f32>> {
        &self.state
    }
    fn get_config(&self) -> Config {
        self.config.clone()
    }


    fn from_file(cp_path: &str) -> Self {
        let rd = &mut BufReader::new(File::open(cp_path).unwrap());
        let config = Config::from_file(rd);
        let weights = TransformerWeights::from_file(rd, &config);
        let state = RunState::from_config(&config);

        let cpu = CPU {  };
        Self {
            config: config,
            weights: weights,
            state: state,
            device: cpu,
        }
    }

}

fn sample_top_q(probabilities: &[f32], num: usize, topp: f32, rng: &mut ChaCha20Rng) -> usize {
    let cutoff = (1.0f32 - topp) / ((num - 1) as f32);
    let mut prob_index = probabilities.iter().enumerate().filter(
        |(_, &p)| p > cutoff
    ).collect::<Vec<(usize, &f32)>>();
    prob_index.sort_by(
        |(_, &a2), (_, &b2)|
        b2.partial_cmp(&a2).unwrap()
    );

    let mut cum_prob = 0.0f32;
    let mut last_index = prob_index.len() - 1;
    for i in 0..prob_index.len() {
        cum_prob += prob_index[i].1;
        if cum_prob > topp {
            last_index = i;
            break;
        }
    }

    let r = rng.gen::<f32>() * cum_prob;
    let mut cdf = 0.0f32;

    for i in 0..last_index {
        cdf += prob_index[i].1;
        if r < cdf {
            return prob_index[i].0;
        }
    }

    return prob_index[last_index].0;


}