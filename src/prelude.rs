#![allow(unused)]

/* #region for API callers */

pub use crate::rhf::{RHFOutput, get_energy_nuc, minimal_rhf, minimal_ri_rhf};

/* #endregion */

/* #region for developers */

// RSTSR backend specification
#[cfg(not(feature = "use_openblas"))]
pub(crate) type DeviceTsr = DeviceFaer;
#[cfg(feature = "use_openblas")]
pub(crate) type DeviceTsr = DeviceOpenBLAS;

pub(crate) use libcint::prelude::*;
pub(crate) use rstsr::prelude::*;

pub(crate) use crate::*;

pub(crate) type Tsr = Tensor<f64, DeviceTsr>;
pub(crate) type TsrView<'a> = TensorView<'a, f64, DeviceTsr>;

/* #endregion */
