use num_traits::Float;
use rand_distr::uniform::SampleUniform;

use crate::{prelude::conv1d::TryConv1D, shapes::*, tensor::*, tensor_ops::*};

use super::*;

pub mod builder {
    #[derive(Debug)]
    pub struct Conv2D<
        const IN_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL_SIZE: usize,
        const STRIDE: usize = 1,
        const PADDING: usize = 0,
        const DILATION: usize = 1,
        const GROUPS: usize = 1,
    >;

    #[derive(Debug)]
    pub struct Conv1D<
        const IN_CHAN: usize,
        const OUT_CHAN: usize,
        const KERNEL_SIZE: usize,
        const STRIDE: usize = 1,
        const PADDING: usize = 0,
        const DILATION: usize = 1,
        const GROUPS: usize = 1,
    >;
}

impl<
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const L: usize,
        const G: usize,
        E,
        D,
    > BuildOnDevice<D, E> for builder::Conv2D<I, O, K, S, P, L, G>
where
    E: Dtype,
    D: Device<E>,
    Const<{ I / G }>: Sized,
    Conv2D<I, O, K, S, P, L, G, E, D>: BuildModule<D, E>,
{
    type Built = Conv2D<I, O, K, S, P, L, G, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

impl<
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const L: usize,
        const G: usize,
        E,
        D,
    > BuildOnDevice<D, E> for builder::Conv1D<I, O, K, S, P, L, G>
where
    E: Dtype,
    D: Device<E>,
    Const<{ I / G }>: Sized,
    Conv1D<I, O, K, S, P, L, G, E, D>: BuildModule<D, E>,
{
    type Built = Conv1D<I, O, K, S, P, L, G, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

/// **Requires Nightly** Performs *unbiased* 2d convolutions on 3d and 4d images.
///
/// **Pytorch Equivalent**: `torch.nn.Conv2d(..., bias=False)`
///
/// To create a biased conv, combine with [crate::nn::modules::Bias2D]:
/// ```ignore
/// # use dfdx::prelude::*;
/// type BiasedConv = (Conv2D<3, 5, 4>, Bias2D<5>);
/// ```
///
/// Generics:
/// - `IN_CHAN`: The number of input channels in an image.
/// - `OUT_CHAN`: The number of channels in the output of the layer.
/// - `KERNEL_SIZE`: The size of the kernel applied to both width and height of the images.
/// - `STRIDE`: How far to move the kernel each step. Defaults to `1`
/// - `PADDING`: How much zero padding to add around the images. Defaults to `0`.
/// - `DILATION`: Controls the spacing between kernel points. Defaults to `1`.
/// - `GROUPS`: Controls the connections between inputs and outputs.
///     `IN_CHAN` and `OUT_CHAN` must both be divisible by `GROUPS`. For example,
///
/// See [conv animations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) for helpful
/// visualization of all of these parameters.

#[derive(Debug, Clone)]
pub struct Conv2D<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize,
    const PADDING: usize,
    const DILATION: usize,
    const GROUPS: usize,
    E: Dtype,
    D: Storage<E>,
> where
    Const<{ IN_CHAN / GROUPS }>: Sized,
{
    pub weight: Tensor<Rank4<OUT_CHAN, { IN_CHAN / GROUPS }, KERNEL_SIZE, KERNEL_SIZE>, E, D>,
}

impl<
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const L: usize,
        const G: usize,
        E,
        D,
    > TensorCollection<E, D> for Conv2D<I, O, K, S, P, L, G, E, D>
where
    Const<{ I / G }>: Sized,
    E: Dtype + Float + SampleUniform,
    D: Device<E>,
{
    type To<E2: Dtype, D2: Device<E2>> = Conv2D<I, O, K, S, P, L, G, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            Self::tensor(
                "weight",
                |s| &s.weight,
                |s| &mut s.weight,
                TensorOptions::reset_with(|t| {
                    let b = E::ONE / E::from_usize(I * K * K).unwrap().sqrt();
                    t.try_fill_with_distr(rand_distr::Uniform::new(-b, b))
                }),
            ),
            |weight| Conv2D { weight },
        )
    }
}

impl<
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const L: usize,
        const G: usize,
        E,
        D,
        Img,
    > Module<Img> for Conv2D<I, O, K, S, P, L, G, E, D>
where
    Const<{ I / G }>: Sized,
    E: Dtype,
    D: Device<E>,
    (Img, Tensor<Rank4<O, { I / G }, K, K>, E, D>):
        TryConv2D<Const<S>, Const<P>, Const<L>, Const<G>>,
{
    type Output = <(Img, Tensor<Rank4<O, { I / G }, K, K>, E, D>) as TryConv2D<
        Const<S>,
        Const<P>,
        Const<L>,
        Const<G>,
    >>::Convolved;
    type Error = <(Img, Tensor<Rank4<O, { I / G }, K, K>, E, D>) as TryConv2D<
        Const<S>,
        Const<P>,
        Const<L>,
        Const<G>,
    >>::Error;

    fn try_forward(&self, x: Img) -> Result<Self::Output, Self::Error> {
        (x, self.weight.clone()).try_conv2d(Const, Const, Const, Const)
    }
}

impl<
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const L: usize,
        const G: usize,
        E: Dtype,
        D: Storage<E>,
    > NonMutableModule for Conv2D<I, O, K, S, P, L, G, E, D>
where
    Const<{ I / G }>: Sized,
{
}

/// **Requires Nightly** Performs *unbiased* 1d convolutions on 2d and 3d images.
///
/// **Pytorch Equivalent**: `torch.nn.Conv1d(..., bias=False)`
///
/// To create a biased conv, combine with [crate::nn::modules::Bias1D]:
/// ```ignore
/// # use dfdx::prelude::*;
/// type BiasedConv = (Conv1D<3, 5, 4>, Bias1D<5>);
/// ```
///
/// Generics:
/// - `IN_CHAN`: The number of input channels in an image.
/// - `OUT_CHAN`: The number of channels in the output of the layer.
/// - `KERNEL_SIZE`: The size of the kernel applied to the width of the images.
/// - `STRIDE`: How far to move the kernel each step. Defaults to `1`.
/// - `PADDING`: How much zero padding to add around the images. Defaults to `0`.
/// - `DILATION`: Controls the spacing between kernel points. Defaults to `1`.
/// - `GROUPS`: Controls the connections between inputs and outputs.
///     `IN_CHAN` and `OUT_CHAN` must both be divisible by `GROUPS`. For example,
///
/// See [conv animations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) for helpful
/// visualization of all of these parameters.

#[derive(Debug, Clone)]
pub struct Conv1D<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize,
    const PADDING: usize,
    const DILATION: usize,
    const GROUPS: usize,
    E: Dtype,
    D: Storage<E>,
> where
    Const<{ IN_CHAN / GROUPS }>: Sized,
{
    pub weight: Tensor<Rank3<OUT_CHAN, { IN_CHAN / GROUPS }, KERNEL_SIZE>, E, D>,
}

impl<
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const L: usize,
        const G: usize,
        E,
        D,
    > TensorCollection<E, D> for Conv1D<I, O, K, S, P, L, G, E, D>
where
    Const<{ I / G }>: Sized,
    E: Dtype + Float + SampleUniform,
    D: Device<E>,
{
    type To<E2: Dtype, D2: Device<E2>> = Conv1D<I, O, K, S, P, L, G, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            Self::tensor(
                "weight",
                |s| &s.weight,
                |s| &mut s.weight,
                TensorOptions::reset_with(|t| {
                    let b = E::ONE / E::from_usize(I * K).unwrap().sqrt();
                    t.try_fill_with_distr(rand_distr::Uniform::new(-b, b))
                }),
            ),
            |weight| Conv1D { weight },
        )
    }
}

impl<
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const L: usize,
        const G: usize,
        E,
        D,
        Img,
    > Module<Img> for Conv1D<I, O, K, S, P, L, G, E, D>
where
    Const<{ I / G }>: Sized,
    E: Dtype,
    D: Device<E>,
    (Img, Tensor<Rank3<O, { I / G }, K>, E, D>): TryConv1D<Const<S>, Const<P>, Const<L>, Const<G>>,
{
    type Output = <(Img, Tensor<Rank3<O, { I / G }, K>, E, D>) as TryConv1D<
        Const<S>,
        Const<P>,
        Const<L>,
        Const<G>,
    >>::Convolved;
    type Error = <(Img, Tensor<Rank3<O, { I / G }, K>, E, D>) as TryConv1D<
        Const<S>,
        Const<P>,
        Const<L>,
        Const<G>,
    >>::Error;

    fn try_forward(&self, x: Img) -> Result<Self::Output, Self::Error> {
        (x, self.weight.clone()).try_conv1d(Const, Const, Const, Const)
    }
}

impl<
        const I: usize,
        const O: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        const L: usize,
        const G: usize,
        E: Dtype,
        D: Storage<E>,
    > NonMutableModule for Conv1D<I, O, K, S, P, L, G, E, D>
where
    Const<{ I / G }>: Sized,
{
}
#[cfg(test)]
mod tests {
    use crate::{
        optim::*,
        tensor::{AsArray, SampleTensor, ZerosTensor},
        tests::*,
    };

    use super::{builder::Conv2D, *};

    #[rustfmt::skip]
    #[test]
    fn test_forward_3d_sizes() {
        let dev: TestDevice = Default::default();
        let x = dev.zeros::<Rank3<3, 10, 10>>();
        let _: Tensor<Rank3<2, 8, 8>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<4, 8, 8>, _, _, _> = dev.build_module::<Conv2D<3, 4, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<4, 9, 9>, _, _, _> = dev.build_module::<Conv2D<3, 4, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<4, 7, 7>, _, _, _> = dev.build_module::<Conv2D<3, 4, 4>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<2, 4, 4>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<2, 3, 3>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<2, 10, 10>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 1, 1>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<2, 12, 12>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 1, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank3<2, 6, 6>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 2, 2>, TestDtype>().forward(x.clone());
    }

    #[test]
    fn test_grouped_forward_sizes() {
        let dev: TestDevice = Default::default();

        let x = dev.zeros::<Rank3<16, 10, 10>>();

        let m = dev.build_module::<Conv2D<16, 32, 3, 1, 0, 1, 1>, TestDtype>();
        let _: Tensor<Rank4<32, 16, 3, 3>, _, _> = m.weight;
        let _: Tensor<Rank3<32, 8, 8>, _, _> = m.forward(x.clone());

        let m = dev.build_module::<Conv2D<16, 32, 3, 1, 0, 1, 2>, TestDtype>();
        let _: Tensor<Rank4<32, 8, 3, 3>, _, _> = m.weight;
        let _: Tensor<Rank3<32, 8, 8>, _, _> = m.forward(x.clone());

        let m = dev.build_module::<Conv2D<16, 32, 3, 1, 0, 1, 4>, TestDtype>();
        let _: Tensor<Rank4<32, 4, 3, 3>, _, _> = m.weight;
        let _: Tensor<Rank3<32, 8, 8>, _, _> = m.forward(x.clone());

        let m = dev.build_module::<Conv2D<16, 32, 3, 1, 0, 1, 8>, TestDtype>();
        let _: Tensor<Rank4<32, 2, 3, 3>, _, _> = m.weight;
        let _: Tensor<Rank3<32, 8, 8>, _, _> = m.forward(x.clone());

        let m = dev.build_module::<Conv2D<16, 32, 3, 1, 0, 1, 16>, TestDtype>();
        let _: Tensor<Rank4<32, 1, 3, 3>, _, _> = m.weight;
        let _: Tensor<Rank3<32, 8, 8>, _, _> = m.forward(x);
    }

    #[rustfmt::skip]
    #[test]
    fn test_forward_4d_sizes() {
        let dev: TestDevice = Default::default();
        let x = dev.zeros::<Rank4<5, 3, 10, 10>>();
        let _: Tensor<Rank4<5, 2, 8, 8>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 4, 8, 8>, _, _, _> = dev.build_module::<Conv2D<3, 4, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 4, 9, 9>, _, _, _> = dev.build_module::<Conv2D<3, 4, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 4, 7, 7>, _, _, _> = dev.build_module::<Conv2D<3, 4, 4>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 2, 4, 4>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 2, 3, 3>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 3>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 2, 10, 10>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 1, 1>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 2, 12, 12>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 1, 2>, TestDtype>().forward(x.clone());
        let _: Tensor<Rank4<5, 2, 6, 6>, _, _, _> = dev.build_module::<Conv2D<3, 2, 3, 2, 2>, TestDtype>().forward(x.clone());
    }

    #[test]
    fn test_2_conv_sizes() {
        let dev = Cpu::default();
        type A = Conv2D<1, 2, 3>;
        type B = Conv2D<2, 4, 3>;
        let _: Tensor<Rank3<4, 6, 6>, _, _> = dev
            .build_module::<(A, B), TestDtype>()
            .forward(dev.zeros::<Rank3<1, 10, 10>>());
    }

    #[test]
    fn test_3_conv_sizes() {
        type A = Conv2D<1, 2, 3>;
        type B = Conv2D<2, 4, 3>;
        type C = Conv2D<4, 1, 1, 1, 1>;

        let dev = Cpu::default();
        let _: Tensor<Rank3<1, 8, 8>, _, _> = dev
            .build_module::<(A, B, C), TestDtype>()
            .forward_mut(dev.zeros::<Rank3<1, 10, 10>>());
    }

    #[test]
    fn test_conv_with_optimizer() {
        let dev: TestDevice = Default::default();

        let mut m = dev.build_module::<Conv2D<2, 4, 3>, TestDtype>();

        let weight_init = m.weight.clone();

        let mut opt = Sgd::new(&m, Default::default());
        let out = m.forward(dev.sample_normal::<Rank4<8, 2, 28, 28>>().leaky_trace());
        let g = out.square().mean().backward();

        assert_ne!(
            g.get(&m.weight).array(),
            [[[[TestDtype::zero(); 3]; 3]; 2]; 4]
        );

        opt.update(&mut m, &g).expect("unused params");

        assert_ne!(weight_init.array(), m.weight.array());
    }
}
