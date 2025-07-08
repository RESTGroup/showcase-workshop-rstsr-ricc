#![allow(unused)]

/* #region for API callers */

pub use crate::ccsd::{CCSDConfig, RCCSDInfo, RCCSDIntermediates, RCCSDResults};
pub use crate::ccsdt::{RCCSDTIntermediates, RCCSDTResults};
pub use crate::rhf::RHFResults;

// RSTSR backend specification
#[cfg(not(feature = "use_openblas"))]
pub type DeviceTsr = DeviceFaer;
#[cfg(feature = "use_openblas")]
pub type DeviceTsr = DeviceOpenBLAS;
pub type Tsr<D = IxD> = Tensor<f64, DeviceTsr, D>;
pub type TsrView<'a, D = IxD> = TensorView<'a, f64, DeviceTsr, D>;
pub type TsrMut<'a, D = IxD> = TensorMut<'a, f64, DeviceTsr, D>;

/* #endregion */

/* #region for developers */

pub(crate) use itertools::*;
pub(crate) use libcint::prelude::*;
pub(crate) use rayon::prelude::*;
pub(crate) use rstsr::prelude::*;

pub(crate) use crate::*;

/* #endregion */
