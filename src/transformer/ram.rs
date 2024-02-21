use std::{io::BufReader, fs::File};
use crate::device::cpu::CPU;
use super::{state::{RunState, TransformerWeights}, Config, Transformer};
pub struct TransformerCPU {
    pub config: Config,
    pub weights: TransformerWeights<Vec<f32>>,
    pub state: RunState<Vec<f32>>,
    pub device: CPU
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
            config,
            weights,
            state,
            device: cpu,
        }
    }

}
