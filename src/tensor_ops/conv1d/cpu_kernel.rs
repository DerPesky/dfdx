use crate::shapes::{Dtype, Shape};
use crate::tensor::{cpu::*, *};
use crate::tensor_ops::matmul::cpu_kernel::MatMulImpl;

use super::{Conv1DKernel, Conv1DOp};

use std::sync::Arc;

impl Conv1DOp {
    #[inline(always)]
    fn unfold_idx(&self, [k, x]: [usize; 2]) -> Option<[usize; 1]> {
        let mut ox = x + self.padding;
        if ox < self.dilation * k {
            return None;
        }

        ox -= self.dilation * k;
        if ox % self.stride != 0 {
            return None;
        }

        ox /= self.stride;
        if ox >= self.len_out {
            return None;
        }

        Some(ox)
    }
}

impl Cpu {
    #[inline]
    fn fwd<E: Dtype>(
        &self,
        op: &Conv1DOp,
        img: &[E],
        filters: &[E],
        out: &mut [E],
        buf: &mut [E],
    ) -> Result<(), CpuError>
    where
        Self: MatMulImpl<E>,
    {
        {
            let mut i = 0;
            for c in 0..(op.groups * op.chan_in) {
                for k in 0..op.kernel {
                    for ox in 0..op.len_out {
                        let x = (ox * op.stride + op.dilation * k).wrapping_sub(op.padding);
                        if x < op.len_in {
                            buf[i] = img[c * op.len_in + x];
                        }

                        i += 1;
                    }
                }
            }
        }

        // (G, O / G, C * K) * (G, C * K, OH) = (G, O / G, OH)
        let m = op.chan_out / op.groups;
        let k = op.chan_in * op.kernel;
        let n = op.len_out;
        for g in 0..op.groups {
            Self::matmul(
                (m, k, n),
                false,
                filters[g * m * k..].as_ptr(),
                [k, 1],
                buf[g * k * n..].as_ptr(),
                [n, 1],
                out[g * m * n..].as_mut_ptr(),
                [n, 1],
            );
        }

        Ok(())
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn bwd<E: Dtype>(
        &self,
        op: &Conv1DOp,
        img: &[E],
        grad_img: &mut [E],
        filters_tr: &[E],
        grad_filters_tr: &mut [E],
        grad_out: &[E],
        buf: &mut [E],
    ) -> Result<(), CpuError>
    where
        Self: MatMulImpl<E>,
    {
        {
            let mut i = 0;
            for o in 0..op.chan_out {
                for k in 0..op.kernel {
                    for x in 0..op.len_in {
                        if let Some(ox) = op.unfold_idx([k, x]) {
                            buf[i] = grad_out[o * op.len_out + ox];
                        }

                        i += 1;
                    }
                }
            }
        }

        {
            // img_g += filters^T * unfold(grad_out)
            // (G, C, H) += (G, C, O/G * K) * (G, O/G * K, H)
            let m = op.chan_in;
            let k = (op.chan_out / op.groups) * op.kernel;
            let n = op.len_in;
            for g in 0..op.groups {
                Self::matmul(
                    (m, k, n),
                    true,
                    filters_tr[g * m * k..].as_ptr(),
                    [k, 1],
                    buf[g * k * n..].as_ptr(),
                    [n, 1],
                    grad_img[g * m * n..].as_mut_ptr(),
                    [n, 1],
                );
            }
        }

        {
            // weight_g^T += img * unfold(patches)^T
            // (G, C, O/G * K) += (G, C, H) * (G, H, O/G * K)
            let m = op.chan_in;
            let k = op.len_in;
            let n = (op.chan_out / op.groups) * op.kernel;
            for g in 0..op.groups {
                Self::matmul(
                    (m, k, n),
                    true,
                    img[g * m * k..].as_ptr(),
                    [k, 1],
                    buf[g * k * n..].as_ptr(),
                    [1, k],
                    grad_filters_tr[g * m * n..].as_mut_ptr(),
                    [n, 1],
                );
            }
        }

        Ok(())
    }
}

impl<E: Dtype> Conv1DKernel<E> for Cpu
where
    Self: MatMulImpl<E>,
{
    fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self>, Self::Err> {
        self.try_zeros_like(&s)
    }

    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: Conv1DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Self::Err> {
        let patches = (op.groups * op.chan_in, op.kernel, op.len_out);
        let mut patches = self.try_alloc_zeros::<E>(patches.num_elements())?;
        let [lstride, ostride] = match L::NUM_DIMS {
            2 => [0; 2],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();
        let rhs = rhs.data.as_ref();
        let out = Arc::make_mut(&mut out.data);
        for i_batch in 0..op.batch {
            self.fwd(
                &op,
                &lhs[i_batch * lstride..],
                rhs,
                &mut out[i_batch * ostride..],
                &mut patches,
            )?;
        }

        Ok(())
    }

    fn backward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: Conv1DOp,
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec,
        out: &impl Tensorlike<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let f_tr_shape = [op.groups, op.chan_in, op.chan_out / op.groups, op.kernel];
        let patches_shape = [op.chan_out, op.kernel, op.len_in];
        let mut patches = self.try_alloc_zeros::<E>(patches_shape.num_elements())?;
        let mut f1023 = self.try_alloc_zeros::<E>(f_tr_shape.num_elements())?;
        let mut grad_f1023 = self.try_alloc_zeros::<E>(f_tr_shape.num_elements())?;

        {
            // transpose filters in f1023
            let buf = rhs.data.as_ref();
            let mut f_idx = NdIndex::new(f_tr_shape, f_tr_shape.strides());
            while let Some((i, [g, c, o, k])) = f_idx.next_with_idx() {
                let idx = (g * (op.chan_out / op.groups) + o) * rhs.strides[0]
                    + c * rhs.strides[1]
                    + k * rhs.strides[2];
                f1023[i] = buf[idx];
            }
        }

        let [lstride, ostride] = match L::NUM_DIMS {
            2 => [0; 2],
            _ => unreachable!(),
        };
        let lhs = lhs.data.as_ref();

        for i_batch in 0..op.batch {
            self.bwd(
                &op,
                &lhs[i_batch * lstride..],
                &mut grad_lhs[i_batch * lstride..],
                &f1023,
                &mut grad_f1023,
                &grad_out[i_batch * ostride..],
                &mut patches,
            )?;
        }

        {
            // untranspose filters
            let mut f_idx = NdIndex::new(f_tr_shape, f_tr_shape.strides());
            while let Some((i, [g, c, o, k, k2])) = f_idx.next_with_idx() {
                let idx = (g * (op.chan_out / op.groups) + o) * rhs.strides[0]
                    + c * rhs.strides[1]
                    + k * rhs.strides[2];
                grad_rhs[idx] += grad_f1023[i];
            }
        }

        Ok(())
    }
}
