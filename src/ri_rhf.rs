use crate::prelude::*;

const MAX_ITER: usize = 64;
const TOL_E: f64 = 1e-9;
const TOL_D: f64 = 1e-6;

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

    println!("Prepare integrals: {:.3?}", time.elapsed());

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

    // this is safe version of get_k
    // the efficiency should be the same as the unsafe version
    // (but maybe not everybody is happy to use zipped iterators)
    #[allow(unused)]
    let get_k_safe = |mo_coeff: TsrView| -> Tsr {
        let occ_coeff = mo_coeff.i((.., ..nocc));
        let mut scr_xob = unsafe { rt::empty(([naux, nocc, nao], &device)) };
        let mut scr_xob_iter = scr_xob.axes_iter_mut(0);
        (0..naux).into_par_iter().zip(scr_xob_iter).for_each(|(p, mut scr_ob)| {
            let cderi_ob = cderi.i(p).unpack_tril(FlagSymm::Sy);
            scr_ob.matmul_from(&occ_coeff.t(), &cderi_ob, 1.0, 0.0);
        });
        let scr_xob_flat = scr_xob.reshape([naux * nocc, nao]);
        2.0_f64 * (scr_xob_flat.t() % &scr_xob_flat)
    };

    let mut dm = rt::zeros(([nao, nao], &device));
    let mut mo_coeff = rt::zeros(([nao, nao], &device));
    let mut mo_energy = rt::zeros(([nao], &device));
    let mut diis_obj = DIISIncore::new(DIISIncoreFlags::default(), &device);
    let mut old_dm = rt::zeros(([nao, nao], &device));
    let mut e_tot_old = 0.0;
    let mut converged = false;
    let mut niter = 0;

    for _ in 0..MAX_ITER {
        niter += 1;

        let time = std::time::Instant::now();

        // j/k evaluation
        let j = get_j(dm.view());
        let k = get_k(mo_coeff.view());

        // fock matrix and energy evaluation
        let e_elec = ((&hcore + 0.5_f64 * &j - 0.25_f64 * &k) * &dm).sum();
        let e_tot = e_nuc + e_elec;
        let mut fock = &hcore + &j - 0.5_f64 * &k;

        // diis update
        if niter > 2 {
            fock = diis_obj.update(fock, None, None).into_shape([nao, nao]);
        }

        // new density and coefficients
        (mo_energy, mo_coeff) = rt::linalg::eigh((fock.view(), ovlp.view())).into();
        dm = 2.0_f64 * mo_coeff.i((.., ..nocc)) % mo_coeff.i((.., ..nocc)).t();

        // convergence check
        let e_tot_diff = (e_tot - e_tot_old).abs();
        let dm_diff = (&dm - &old_dm).l2_norm() / (dm.size() as f64).sqrt();
        let elapsed = time.elapsed();
        println!(
            "Iteration {niter:>2}: E_tot = {e_tot:20.12}, E_diff = {e_tot_diff:8.2e}, DM_diff = {dm_diff:8.2e}, Elapsed = {elapsed:.3?}"
        );
        if e_tot_diff < TOL_E && dm_diff < TOL_D {
            converged = true;
            break;
        }
        e_tot_old = e_tot;
        old_dm = dm.clone();
    }

    if !converged {
        panic!("RI-RHF did not converge after {MAX_ITER} iterations");
    }

    // fock has been updated by diis, so we need to recompute j/k
    let j = get_j(dm.view());
    let k = get_k(mo_coeff.view());
    let e_elec = ((&hcore + 0.5_f64 * &j - 0.25_f64 * &k) * &dm).sum();
    let e_tot = e_nuc + e_elec;
    println!("Total elec energy: {e_elec}");
    println!("Total RHF energy: {e_tot}");
    println!("Elapsed time for RI-RHF: {:.3?} in {niter} iterations", time.elapsed());

    RHFResults { mo_energy, mo_coeff, dm, e_nuc, e_elec, e_tot }
}
