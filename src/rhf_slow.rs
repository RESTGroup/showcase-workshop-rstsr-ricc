use crate::prelude::*;

pub const NITER: usize = 40; // hardcoded SCF iterations

pub fn get_energy_nuc(cint_data: &CInt) -> f64 {
    let device = DeviceTsr::default();

    let atom_coords = {
        let coords = cint_data.atom_coords();
        let coords = coords.into_iter().flatten().collect::<Vec<f64>>();
        rt::asarray((coords, &device)).into_shape((-1, 3))
    };
    let atom_charges = rt::asarray((cint_data.atom_charges(), &device));
    let mut dist = rt::sci::cdist((atom_coords.view(), atom_coords.view()));
    dist.diagonal_mut(None).fill(f64::INFINITY);
    0.5 * (&atom_charges * atom_charges.i((.., None)) / dist).sum()
}

pub fn minimal_rhf(cint_data: &CInt) -> RHFResults {
    let time = std::time::Instant::now();
    let device = DeviceTsr::default();

    let nocc = (cint_data.atom_charges().into_iter().sum::<f64>() / 2.0) as usize;
    let nao = cint_data.nao();

    let e_nuc = get_energy_nuc(cint_data);
    println!("Nuclear repulsion energy: {e_nuc}");

    let hcore = util::intor_row_major(cint_data, "int1e_kin") + util::intor_row_major(cint_data, "int1e_nuc");
    let ovlp = util::intor_row_major(cint_data, "int1e_ovlp");
    let int2e = util::intor_row_major(cint_data, "int2e");

    let mut dm = ovlp.zeros_like();
    let mut mo_coeff = rt::zeros(([nao, nao], &device));
    let mut mo_energy = rt::zeros(([nao], &device));
    for _ in 0..NITER {
        let fock = &hcore + ((1.0_f64 * &int2e - 0.5_f64 * int2e.swapaxes(1, 2)) * &dm).sum_axes([-1, -2]);
        (mo_energy, mo_coeff) = rt::linalg::eigh((fock.view(), ovlp.view())).into();
        dm = 2.0_f64 * mo_coeff.i((.., ..nocc)) % mo_coeff.i((.., ..nocc)).t();
    }
    let eng_scratch = &hcore + ((0.5_f64 * &int2e - 0.25_f64 * int2e.swapaxes(1, 2)) * &dm).sum_axes([-1, -2]);
    let e_elec = (&dm * &eng_scratch).sum();
    let e_tot = e_nuc + e_elec;
    println!("Total elec energy: {e_elec}");
    println!("Total RHF energy: {e_tot}");
    println!("Elapsed time for RHF: {:.2?} in {NITER} iterations", time.elapsed());

    RHFResults { mo_energy, mo_coeff, dm, e_nuc, e_elec, e_tot }
}
