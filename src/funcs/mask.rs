use super::Function;
use crate::tensor::*;

#[derive(Debug, Clone)]
pub struct Mask {
    mask: Tensor<f32>,
    mask_rev: Tensor<f32>,
    value: f32,
}
impl Mask {
    pub fn new(mask: Tensor<bool>, value: f32) -> Box<dyn Function> {
        Box::new(Self {
            mask: (&mask).into(),
            mask_rev: (&!&mask).into(),
            value,
        })
    }
}

impl Function for Mask {
    fn run(&mut self, inps: &[&Tensor<f32>], _training: bool) -> Result<Tensor<f32>, TensorError> {
        inps[0].map(self.mask.dim(), |t| {
            let dat = t
                .blob()
                .iter()
                .zip(self.mask.blob().iter())
                .map(|(v, m)| if *m == 1. { self.value } else { *v })
                .collect::<Vec<_>>();
            Ok(Tensor::raw(t.shape(), dat)?)
        })
    }
    fn grad(
        &self,
        _inps: &[&Tensor<f32>],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        Ok(vec![(out_grad * &self.mask_rev)?])
    }
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}
