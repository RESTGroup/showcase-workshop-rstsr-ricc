use crate::prelude::*;

pub const NITER: usize = 40; // hardcoded SCF iterations

pub struct RHFResults {
    pub mo_coeff: Tsr,
    pub mo_energy: Tsr,
    pub dm: Tsr,
    pub e_nuc: f64,
    pub e_elec: f64,
    pub e_tot: f64,
}

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

    RHFResults { mo_energy, mo_coeff, dm, e_nuc, e_elec, e_tot }
}

pub fn minimal_ri_rhf(cint_data: &CInt, aux_cint_data: &CInt) -> RHFResults {
    let device = DeviceTsr::default();

    let nocc = (cint_data.atom_charges().into_iter().sum::<f64>() / 2.0) as usize;
    let nao = cint_data.nao();
    let naux = aux_cint_data.nao();

    let e_nuc = get_energy_nuc(cint_data);
    println!("Nuclear repulsion energy: {e_nuc}");

    let hcore = util::intor_row_major(cint_data, "int1e_kin") + util::intor_row_major(cint_data, "int1e_nuc");
    let ovlp = util::intor_row_major(cint_data, "int1e_ovlp");
    let int3c2e = util::intor_3c2e_row_major(cint_data, aux_cint_data, "int3c2e");
    let int2c2e = util::intor_row_major(aux_cint_data, "int2c2e");

    // prepare cholesky-decomposed int2c2e
    let int3c2e_trans = int3c2e.into_shape([nao * nao, naux]).into_reverse_axes();
    let int2c2e_l = rt::linalg::cholesky((int2c2e.view(), Lower));
    let cderi = rt::linalg::solve_triangular((int2c2e_l.view(), int3c2e_trans, Lower));
    let cderi = cderi.into_reverse_axes().into_shape([nao, nao, naux]);

    let get_j = |dm: TsrView| -> Tsr {
        let cderi_flat = cderi.reshape([nao * nao, naux]);
        let scr = dm.reshape(nao * nao) % &cderi_flat;
        (cderi_flat % scr).into_shape([nao, nao])
    };

    let get_k = |mo_coeff: TsrView| -> Tsr {
        let occ_coeff = mo_coeff.i((.., ..nocc));
        let scr = (cderi.reshape([nao, nao * naux]).t() % occ_coeff).into_shape([nao, naux, nocc]);
        let scr_flat = scr.reshape([nao, naux * nocc]);
        2.0_f64 * (&scr_flat % scr_flat.t())
    };

    let mut dm = ovlp.zeros_like();
    let mut mo_coeff = rt::zeros(([nao, nao], &device));
    let mut mo_energy = rt::zeros(([nao], &device));
    for _ in 0..NITER {
        let fock = &hcore + get_j(dm.view()) - 0.5_f64 * get_k(mo_coeff.view());
        (mo_energy, mo_coeff) = rt::linalg::eigh((fock.view(), ovlp.view())).into();
        dm = 2.0_f64 * mo_coeff.i((.., ..nocc)) % mo_coeff.i((.., ..nocc)).t();
    }
    let eng_scratch = &hcore + 0.5_f64 * get_j(dm.view()) - 0.25_f64 * get_k(mo_coeff.view());
    let e_elec = (&dm * &eng_scratch).sum();
    let e_tot = e_nuc + e_elec;
    println!("Total elec energy: {e_elec}");
    println!("Total RHF energy: {e_tot}");

    RHFResults { mo_energy, mo_coeff, dm, e_nuc, e_elec, e_tot }
}
