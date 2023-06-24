use cudarc::cudnn::{self, Conv1dBackwardData, Conv1dBackwardFilter, Conv1dForward, CudnnDataType};
use cudarc::driver::DeviceSlice;

use crate::{
    shapes::*,
    tensor::{Cuda, Tensor, Tensorlike},
};

use std::sync::Arc;

trait HasCudnnKernel<E> {}
#[cfg(feature = "f16")]
impl HasCudnnKernel<half::f16> for Cuda {}
impl HasCudnnKernel<f32> for Cuda {}
impl HasCudnnKernel<f64> for Cuda {}

fn make_3d<S: Shape>(strides: S::Concrete, pad: usize) -> [usize; 3] {
    match S::NUM_DIMS {
        2 => [pad, strides[0], strides[1]],
        3 => [strides[0], strides[1], strides[2]],
        _ => unreachable!("Only implemented for 2d & 3d arrays"),
    }
}

impl<E: Dtype + CudnnDataType> super::Conv1DKernel<E> for Cuda
where
    Self: HasCudnnKernel<E>,
{
    fn alloc<S: Shape>(&self, shape: S) -> Result<Tensor<S, E, Self>, Self::Err> {
        let data = unsafe { self.alloc_empty::<E>(shape.num_elements()) }?;
        Ok(self.build_tensor(shape, shape.strides(), data))
    }
    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: super::Conv1DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let mut conv = self.cudnn.create_conv1d::<E>(
            [op.padding as i32],
            [op.stride as i32],
            [op.dilation as i32],
            cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;
        conv.set_group_count(op.groups as i32)?;
        let img = self.cudnn.create_3d_tensor_ex::<E>(
            make_3d::<L>(lhs.shape.concrete(), 1).map(|x| x as i32),
            make_3d::<L>(lhs.strides, 0).map(|x| x as i32),
        )?;
        let filter = self.cudnn.create_3d_filter::<E>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            make_3d::<R>(rhs.shape.concrete(), 1).map(|x| x as i32),
        )?;
        let y = self.cudnn.create_3d_tensor_ex::<E>(
            make_3d::<O>(out.shape.concrete(), 1).map(|x| x as i32),
            make_3d::<O>(out.strides, 0).map(|x| x as i32),
        )?;
        let op = Conv1dForward {
            conv: &conv,
            x: &img,
            w: &filter,
            y: &y,
        };

        let algo = op.pick_algorithm()?;
        let workspace_size_in_bytes = op.get_workspace_size(algo)?;

        unsafe {
            let mut workspace = self.get_workspace::<u8>(workspace_size_in_bytes)?;
            let mut workspace = workspace
                .transmute_mut::<u8>(workspace_size_in_bytes)
                .unwrap();
            assert_eq!(workspace.len(), workspace_size_in_bytes);
            op.launch(
                algo,
                Some(&mut workspace),
                (E::ONE, Default::default()),
                lhs.data.as_ref(),
                rhs.data.as_ref(),
                Arc::get_mut(&mut out.data).unwrap(),
            )?;
        }

        Ok(())
    }

    fn backward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: super::Conv1DOp,
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec,
        out: &impl Tensorlike<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let mut conv = self.cudnn.create_conv1d::<E>(
            [op.padding as i32],
            [op.stride as i32],
            [op.dilation as i32],
            cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;
        conv.set_group_count(op.groups as i32)?;
        let img = self.cudnn.create_3d_tensor_ex::<E>(
            make_3d::<L>(lhs.shape.concrete(), 1).map(|x| x as i32),
            make_3d::<L>(lhs.strides, 0).map(|x| x as i32),
        )?;
        let filter = self.cudnn.create_3d_filter::<E>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            make_3d::<R>(rhs.shape.concrete(), 1).map(|x| x as i32),
        )?;
        let out = self.cudnn.create_3d_tensor_ex::<E>(
            make_3d::<O>(out.shape().concrete(), 1).map(|x| x as i32),
            make_3d::<O>(out.strides(), 0).map(|x| x as i32),
        )?;

        {
            let op = Conv1dBackwardData {
                conv: &conv,
                dx: &img,
                w: &filter,
                dy: &out,
            };
            let algo = op.pick_algorithm()?;
            let workspace_size_in_bytes = op.get_workspace_size(algo)?;

            unsafe {
                let mut workspace = self.get_workspace::<u8>(workspace_size_in_bytes)?;
                let mut workspace = workspace
                    .transmute_mut::<u8>(workspace_size_in_bytes)
                    .unwrap();
                assert_eq!(workspace.len(), workspace_size_in_bytes);
                op.launch(
                    algo,
                    Some(&mut workspace),
                    (E::ONE, Default::default()),
                    grad_lhs,
                    rhs.data.as_ref(),
                    grad_out,
                )
            }?;
        }

        {
            let op = Conv1dBackwardFilter {
                conv: &conv,
                x: &img,
                dw: &filter,
                dy: &out,
            };

            let algo = op.pick_algorithm()?;
            let workspace_size_in_bytes = op.get_workspace_size(algo)?;

            unsafe {
                let mut workspace = self.get_workspace::<u8>(workspace_size_in_bytes)?;
                let mut workspace = workspace
                    .transmute_mut::<u8>(workspace_size_in_bytes)
                    .unwrap();
                assert_eq!(workspace.len(), workspace_size_in_bytes);
                op.launch(
                    algo,
                    Some(&mut workspace),
                    (E::ONE, Default::default()),
                    lhs.data.as_ref(),
                    grad_rhs,
                    grad_out,
                )
            }?;
        }
        Ok(())
    }
}
