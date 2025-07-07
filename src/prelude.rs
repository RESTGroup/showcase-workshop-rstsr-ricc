#![allow(unused)]

/* #region for API callers */

pub use crate::rhf::minimal_rhf;

/* #endregion */

/* #region for developers */

// RSTSR backend specification
pub(crate) type DeviceTsr = DeviceFaer;

pub(crate) use libcint::prelude::*;
pub(crate) use rstsr::prelude::*;

pub(crate) use crate::*;

pub(crate) type Tsr = Tensor<f64, DeviceTsr>;
pub(crate) type TsrView<'a> = TensorView<'a, f64, DeviceTsr>;

/* #endregion */
