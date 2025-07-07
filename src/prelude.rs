#![allow(unused)]

/* #region for API callers */

pub use crate::ccsd::{CCSDConfig, RCCSDInfo, RCCSDIntermediates, RCCSDResults};
pub use crate::ccsdt::{RCCSDTIntermediates, RCCSDTResults};
pub use crate::rhf::RHFResults;

/* #endregion */

/* #region for developers */

// RSTSR backend specification
#[cfg(not(feature = "use_openblas"))]
pub(crate) type DeviceTsr = DeviceFaer;
#[cfg(feature = "use_openblas")]
pub(crate) type DeviceTsr = DeviceOpenBLAS;

pub(crate) use libcint::prelude::*;
pub(crate) use rayon::prelude::*;
pub(crate) use rstsr::prelude::*;

pub(crate) use crate::*;

pub(crate) type Tsr<D = IxD> = Tensor<f64, DeviceTsr, D>;
pub(crate) type TsrView<'a, D = IxD> = TensorView<'a, f64, DeviceTsr, D>;
pub(crate) type TsrMut<'a, D = IxD> = TensorMut<'a, f64, DeviceTsr, D>;

/* #endregion */
