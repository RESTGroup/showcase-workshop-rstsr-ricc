use crate::prelude::*;

/// Obtain integrals (in row-major, same to PySCF but reverse of libcint).
///
/// # Usage
///
/// ```norun
/// let tsr = intor_row_major(&cint_data, "int1e_kin");
/// ```
pub fn intor_row_major(cint_data: &CInt, intor: &str) -> Tsr {
    // use up all rayon available threads for tensor operations
    let device = DeviceTsr::default();

    // intor, "s1", full_shls_slice
    let (out, shape) = cint_data.integrate_row_major(intor, None, None).into();

    // output row-major tensor
    rt::asarray((out, shape.c(), &device))
}

pub fn intor_3c2e_row_major(cint_data: &CInt, aux_cint_data: &CInt, intor: &str) -> Tsr {
    // use up all rayon available threads for tensor operations
    let device = DeviceTsr::default();

    // intor, "s1", full_shls_slice
    let (out, shape) = CInt::integrate_cross_row_major(intor, [cint_data, cint_data, aux_cint_data], None, None).into();

    // row-major by transposition of col-major shape
    rt::asarray((out, shape.c(), &device))
}

pub fn tensor_from_file(fname: &str) -> Tsr {
    // c-contiguous numpy array to f-contiguous rstsr
    let device = DeviceTsr::default();
    let bytes = std::fs::read(fname).unwrap();
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape = npy.shape().iter().map(|x| *x as usize).rev().collect::<Vec<usize>>();
    let data = npy.into_vec().unwrap();
    rt::asarray((data, shape, &device))
}
