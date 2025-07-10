use crate::prelude::*;

pub const NITER: usize = 40; // hardcoded SCF iterations

pub fn minimal_ri_rhf(cint_data: &CInt, aux_cint_data: &CInt) -> RHFResults {
    let time = std::time::Instant::now();
    let device = DeviceTsr::default();

    let nocc = (cint_data.atom_charges().into_iter().sum::<f64>() / 2.0) as usize;
    let nao = cint_data.nao();
    let naux = aux_cint_data.nao();

    let e_nuc = rhf_slow::get_energy_nuc(cint_data);
    println!("Nuclear repulsion energy: {e_nuc}");

    let hcore = util::intor_row_major(cint_data, "int1e_kin") + util::intor_row_major(cint_data, "int1e_nuc");
    let ovlp = util::intor_row_major(cint_data, "int1e_ovlp");
    let int3c2e = util::intor_3c2e_row_major(cint_data, aux_cint_data, "int3c2e");
    let int2c2e = util::intor_row_major(aux_cint_data, "int2c2e");

    // prepare cholesky-decomposed int2c2e
    let int2c2e_l = rt::linalg::cholesky((int2c2e.view(), Lower));
    let cderi = rt::linalg::solve_triangular((int2c2e_l.view(), int3c2e.reshape([nao * nao, naux]).t(), Lower));
    let cderi = cderi.into_shape([naux, nao, nao]).into_contig(RowMajor);

    let get_j = |dm: TsrView| -> Tsr {
        let cderi_flat = cderi.reshape([naux, nao * nao]);
        let dm_flat = dm.reshape([nao * nao]);
        (cderi_flat.t() % (&cderi_flat % dm_flat)).into_shape([nao, nao])
    };

    let get_k = |mo_coeff: TsrView| -> Tsr {
        let occ_coeff = mo_coeff.i((.., ..nocc));
        let scr = (occ_coeff.t() % cderi.reshape([naux * nao, nao]).t()).into_shape([nocc, naux, nao]);
        let scr_flat = scr.reshape([naux * nocc, nao]);
        2.0_f64 * (scr_flat.t() % &scr_flat)
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
    println!("Elapsed time for RI-RHF: {:.2?} in {NITER} iterations", time.elapsed());

    RHFResults { mo_energy, mo_coeff, dm, e_nuc, e_elec, e_tot }
}
