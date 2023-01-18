use crate::{
    shapes::{Axes, BroadcastStridesTo, ReduceShapeTo, Shape},
    tensor::cuda::{Cuda, CudaArray},
};

use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};

use std::sync::Arc;

const MODULE_NAME: &str = "min_to";
const FWD_FN_NAME: &str = "min_to_forward";
const BWD_FN_NAME: &str = "min_to_backward";
const ALL_FN_NAMES: [&str; 3] = [FWD_FN_NAME, BWD_FN_NAME, "fill_with"];
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/min_to.ptx"));

impl super::MinReduceKernel<f32> for Cuda {
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Self::Storage<Src, f32>,
    ) -> Result<Self::Storage<Dst, f32>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        if !self.dev.has_func(MODULE_NAME, FWD_FN_NAME) {
            self.dev
                .load_ptx(PTX_SRC.into(), MODULE_NAME, &ALL_FN_NAMES)?;
        }

        let mut storage = self.dev.alloc_zeros_async::<f32>(dst.num_elements())?;
        let fill_fn = self.dev.get_func(MODULE_NAME, "fill_with").unwrap();
        unsafe {
            fill_fn.launch_async(
                LaunchConfig::for_num_elems(dst.num_elements() as u32),
                (&mut storage, f32::INFINITY, dst.num_elements()),
            )
        }?;

        let fwd_fn = self.dev.get_func(MODULE_NAME, FWD_FN_NAME).unwrap();

        let dims: CudaSlice<usize> = self.dev.take_async(inp.shape.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(inp.strides.into())?;
        let out_strides = BroadcastStridesTo::<Src, Ax>::broadcast_strides(&dst, dst.strides());
        let out_strides: CudaSlice<usize> = self.dev.take_async(out_strides.into())?;

        let inp_numel = inp.shape.num_elements();
        let cfg = LaunchConfig::for_num_elems(inp_numel as u32);
        let params = (
            inp_numel,         // size_t numel,
            Src::NUM_DIMS,     // size_t num_dims,
            &dims,             // const size_t *dims,
            inp.data.as_ref(), // const float *inp,
            &inp_strides,      // const size_t *inp_strides,
            &mut storage,      // float *out,
            &out_strides,      // const size_t *out_strides
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;
        Ok(CudaArray {
            data: Arc::new(storage),
            shape: dst,
            strides: dst.strides(),
        })
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        inp: &Self::Storage<Src, f32>,
        grad_inp: &mut Self::Storage<Src, f32>,
        out: &Self::Storage<Dst, f32>,
        grad_out: &Self::Storage<Dst, f32>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let bwd_fn = self.dev.get_func(MODULE_NAME, BWD_FN_NAME).unwrap();

        let dims: CudaSlice<usize> = self.dev.take_async(grad_inp.shape.concrete().into())?;
        let inp_strides: CudaSlice<usize> = self.dev.take_async(grad_inp.strides.into())?;
        let out_strides: Src::Concrete =
            BroadcastStridesTo::<Src, Ax>::broadcast_strides(&grad_out.shape, grad_out.strides);
        let out_strides: CudaSlice<usize> = self.dev.take_async(out_strides.into())?;

        let inp_numel = grad_inp.shape.num_elements();
        let cfg = LaunchConfig::for_num_elems(inp_numel as u32);
        let params = (
            inp_numel,                         // size_t numel,
            Src::NUM_DIMS,                     // size_t num_dims,
            &dims,                             // const size_t *dims,
            inp.data.as_ref(),                 // const float *inp,
            Arc::make_mut(&mut grad_inp.data), // float *grad_inp,
            &inp_strides,                      // const size_t *inp_strides,
            out.data.as_ref(),                 // const float *out,
            grad_out.data.as_ref(),            // const float *grad_out,
            &out_strides,                      // const size_t *out_strides
        );
        unsafe { bwd_fn.launch_async(cfg, params) }?;
        Ok(())
    }
}