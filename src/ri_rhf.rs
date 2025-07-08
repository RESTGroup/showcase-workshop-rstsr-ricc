use crate::prelude::*;

const NITER: usize = 40;

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
    let int2c2e = util::intor_row_major(aux_cint_data, "int2c2e");

    // int3c2e should be triangular-packed (lower for row-major), and auxiliary basis at first
    // this requires col-major API in libcint-rs, or manually transposing
    // the following code uses the former, i.e. col-major API -> transpose to row-major
    let int3c2e = {
        let (out, shape) = CInt::integrate_cross("int3c2e", [cint_data, cint_data, aux_cint_data], "s2ij", None).into();
        rt::asarray((out, shape.f(), &device)).into_reverse_axes()
    };

    // prepare cholesky-decomposed int2c2e
    let int2c2e_l = rt::linalg::cholesky((int2c2e.view(), Lower));
    let cderi = rt::linalg::solve_triangular((int2c2e_l.view(), int3c2e, Lower));

    let get_j = |dm: TsrView| -> Tsr {
        let mut dm_scaled = 2.0_f64 * dm.to_owned();
        let dm_diag = dm.diagonal(None);
        dm_scaled.diagonal_mut(None).assign(dm_diag);
        let dm_scaled_tp = dm_scaled.pack_tril();
        let j_tp = (&cderi % dm_scaled_tp) % &cderi;
        j_tp.unpack_tril(FlagSymm::Sy)
    };

    let get_k = |mo_coeff: TsrView| -> Tsr {
        let occ_coeff = mo_coeff.i((.., ..nocc));
        let scr_xob = unsafe { rt::empty(([naux, nocc, nao], &device)) };
        (0..naux).into_par_iter().for_each(|p| {
            let mut scr_xob = unsafe { scr_xob.force_mut() };
            let cderi_ob = cderi.i(p).unpack_tril(FlagSymm::Sy);
            scr_xob.i_mut(p).matmul_from(&occ_coeff.t(), &cderi_ob, 1.0, 0.0);
        });
        let scr_xob_flat = scr_xob.reshape([naux * nocc, nao]);
        2.0_f64 * (scr_xob_flat.t() % &scr_xob_flat)
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
