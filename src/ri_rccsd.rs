#![allow(non_snake_case)]

use crate::prelude::*;

/* #region algorithms implementations for RI-RCCSD */

pub fn get_riccsd_intermediates_cderi(mol_info: &RCCSDInfo, intermediates: &mut RCCSDIntermediates) {
    let naux = mol_info.naux();
    let nao = mol_info.nao();
    let nmo = mol_info.nmo();
    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();
    let mo_coeff = &mol_info.mo_coeff;
    let cint_data = &mol_info.cint_data;
    let aux_cint_data = &mol_info.aux_cint_data;

    // prepare cholesky-decomposed int2c2e in atomic-orbital
    let time = std::time::Instant::now();
    let int3c2e = util::intor_3c2e_row_major(cint_data, aux_cint_data, "int3c2e");
    let int2c2e = util::intor_row_major(aux_cint_data, "int2c2e");
    println!("Time elapsed (integrals): {:.3?}", time.elapsed());

    let time = std::time::Instant::now();
    let int3c2e_trans = int3c2e.into_shape([nao * nao, naux]).into_reverse_axes();
    let int2c2e_l = rt::linalg::cholesky((int2c2e.view(), Lower));
    let cderi = rt::linalg::solve_triangular((int2c2e_l.view(), int3c2e_trans, Lower));
    let cderi_uvp = cderi.into_reverse_axes().into_shape([nao, nao, naux]);
    println!("Time elapsed (cderi cholesky and solve_triangular): {:.3?}", time.elapsed());

    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let time = std::time::Instant::now();
    let cderi_svp = (mo_coeff.t() % cderi_uvp.reshape((nao, nao * naux))).into_shape((nmo, nao, naux));

    let b_oo = mo_coeff.i((.., so)).t() % cderi_svp.i(so);
    let b_ov = mo_coeff.i((.., sv)).t() % cderi_svp.i(so);
    let b_vv = mo_coeff.i((.., sv)).t() % cderi_svp.i(sv);
    println!("Time elapsed (ao2mo): {:.3?}", time.elapsed());

    intermediates.b_oo = Some(b_oo);
    intermediates.b_ov = Some(b_ov);
    intermediates.b_vv = Some(b_vv);
}

pub fn get_riccsd_intermediates_1(
    mol_info: &RCCSDInfo,
    intermediates: &mut RCCSDIntermediates,
    t1: TsrView,
    t2: TsrView,
) {
    let naux = mol_info.naux();
    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();

    let b_ov = intermediates.b_ov.as_ref().unwrap();
    let device = b_ov.device().clone();

    // M1j = np.einsum("jbP, jb -> P", B[so, sv], t1)
    let m1_j = t1.reshape(-1) % b_ov.reshape((-1, naux));

    // M1oo = np.einsum("jaP, ia -> ijP", B[so, sv], t1)
    let m1_oo: Tsr = rt::zeros(([nocc, nocc, naux], &device));
    (0..nocc).into_par_iter().for_each(|j| {
        let mut m1_oo = unsafe { m1_oo.force_mut() };
        *&mut m1_oo.i_mut((.., j)) += &t1 % &b_ov.i(j);
    });

    // M2a = np.einsum("jbP, ijab -> iaP", B[so, sv], (2 * t2 - t2.swapaxes(-1, -2)))
    let m2a_ov: Tsr = rt::zeros(([nocc, nvir, naux], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        let mut m2a_ov = unsafe { m2a_ov.force_mut() };
        let scr_jba: Tsr = 2 * t2.i((.., i)) - t2.i(i);
        *&mut m2a_ov.i_mut(i) += scr_jba.reshape((-1, nvir)).t() % b_ov.reshape((-1, naux));
    });

    intermediates.m1_j = Some(m1_j);
    intermediates.m1_oo = Some(m1_oo);
    intermediates.m2a_ov = Some(m2a_ov);
}

pub fn get_riccsd_intermediates_2(mol_info: &RCCSDInfo, intermediates: &mut RCCSDIntermediates, t1: TsrView) {
    let naux = mol_info.naux();
    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();

    let b_oo = intermediates.b_oo.as_ref().unwrap();
    let b_ov = intermediates.b_ov.as_ref().unwrap();
    let b_vv = intermediates.b_vv.as_ref().unwrap();
    let device = b_oo.device().clone();
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();

    // M1aov = np.einsum("ijP, ja -> iaP", B[so, so], t1)
    let m1a_ov = rt::zeros(([nocc, nvir, naux], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        let mut m1a_ov = unsafe { m1a_ov.force_mut() };
        *&mut m1a_ov.i_mut(i) += t1.t() % &b_oo.i(i);
    });

    // M1bov  = np.einsum("abP, ib -> iaP", B[sv, sv], t1)
    let m1b_ov = &t1 % b_vv.reshape((nvir, -1));
    let m1b_ov = m1b_ov.into_shape((nocc, nvir, naux));

    // M1vv = np.einsum("ibP, ia -> abP", B[so, sv], t1)
    let m1_vv = t1.t() % b_ov.reshape((nocc, -1));
    let m1_vv = m1_vv.into_shape((nvir, nvir, naux));

    // M2b = np.einsum("ikP, ka -> iaP", M1oo, t1)
    let m2b_ov = rt::zeros(([nocc, nvir, naux], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        let mut m2b_ov = unsafe { m2b_ov.force_mut() };
        *&mut m2b_ov.i_mut(i) += t1.t() % m1_oo.i(i);
    });

    intermediates.m1a_ov = Some(m1a_ov);
    intermediates.m1b_ov = Some(m1b_ov);
    intermediates.m1_vv = Some(m1_vv);
    intermediates.m2b_ov = Some(m2b_ov);
}

pub fn get_riccsd_energy(intermediates: &RCCSDIntermediates) -> f64 {
    let b_ov = intermediates.b_ov.as_ref().unwrap();
    let m1_j = intermediates.m1_j.as_ref().unwrap();
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();
    let m2a_ov = intermediates.m2a_ov.as_ref().unwrap();

    let e_t1_j = 2.0_f64 * (m1_j % m1_j);
    let e_t1_k = -(m1_oo.reshape(-1) % m1_oo.swapaxes(0, 1).reshape(-1));
    let e_t2 = b_ov.reshape(-1) % m2a_ov.reshape(-1);
    let e_corr = e_t1_j + e_t1_k + e_t2;
    e_corr.to_scalar()
}

pub fn get_riccsd_rhs1(
    mol_info: &RCCSDInfo,
    mut rhs1: TsrMut,
    intermediates: &RCCSDIntermediates,
    t1: TsrView,
    t2: TsrView,
) {
    let naux = mol_info.naux();
    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();

    let b_oo = intermediates.b_oo.as_ref().unwrap();
    let b_ov = intermediates.b_ov.as_ref().unwrap();
    let b_vv = intermediates.b_vv.as_ref().unwrap();
    let m1_j = intermediates.m1_j.as_ref().unwrap();
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();
    let m1a_ov = intermediates.m1a_ov.as_ref().unwrap();
    let m1b_ov = intermediates.m1b_ov.as_ref().unwrap();
    let m2a_ov = intermediates.m2a_ov.as_ref().unwrap();
    let m2b_ov = intermediates.m2b_ov.as_ref().unwrap();

    let device = b_oo.device().clone();

    // === TERM 1 === //
    // RHS1 += - 1 * np.einsum("lcP, lkP, ikac -> ia", B[so, sv], M1oo, (2 * t2 - t2.swapaxes(-1, -2)))

    // "lcP, lkP -> kc", B[so, sv], M1oo
    let mut scr_kc: Tsr = rt::zeros(([nocc, nvir], &device));
    for l in 0..nocc {
        scr_kc += m1_oo.i(l) % b_ov.i(l).t();
    }

    // "kc, ikac -> ia", scr_kc, (2 * t2 - t2.swapaxes(-1, -2)))
    (0..nocc).into_par_iter().for_each(|i| {
        let mut rhs1 = unsafe { rhs1.force_mut() };
        let mut t2_i: Tsr = 2.0 * t2.i(i) - t2.i((.., i));
        t2_i *= scr_kc.i((.., None, ..));
        *&mut rhs1.i_mut(i) -= t2_i.sum_axes([0, 2]);
    });

    // === TERM 2 === //
    // RHS1 += - 1 * np.einsum("kcP, icP, ka -> ia", B[so, sv], (M2a - M1aov), t1)

    rhs1 -= (m2a_ov - m1a_ov).reshape((nocc, -1)) % b_ov.reshape((nocc, -1)).t() % t1;

    // === TERM 3 === //
    // RHS1 +=   1 * np.einsum("icP, acP -> ia", M2a, B[sv, sv])

    rhs1 += m2a_ov.reshape((nocc, -1)) % b_vv.reshape((nvir, -1)).t();

    // === TERM 4 === //
    // RHS1 += - 1 * np.einsum("ikP, kaP -> ia", (B[so, so] + M1oo), (M2a + M1bov))

    for k in 0..nocc {
        rhs1 -= (b_oo.i(k) + m1_oo.i((.., k))) % (m2a_ov.i(k) + m1b_ov.i(k)).t();
    }

    // === TERM 5 === //
    // RHS1 +=   2 * np.einsum("iaP, P -> ia", (B[so, sv] + M1bov - M1aov + M2a - 0.5 * M2b), M1j)

    let scr_iaP: Tsr = b_ov + m1b_ov - m1a_ov + m2a_ov - 0.5 * m2b_ov;
    rhs1 += 2.0 * (scr_iaP.reshape((-1, naux)) % m1_j).reshape((nocc, nvir));
}

pub fn get_riccsd_rhs2_lt2_contract(
    mol_info: &RCCSDInfo,
    mut rhs2: TsrMut,
    intermediates: &RCCSDIntermediates,
    t2: TsrView,
) {
    let naux = mol_info.naux();
    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();

    let b_oo = intermediates.b_oo.as_ref().unwrap();
    let b_ov = intermediates.b_ov.as_ref().unwrap();
    let b_vv = intermediates.b_vv.as_ref().unwrap();
    let m1_j = intermediates.m1_j.as_ref().unwrap();
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();
    let m1_vv = intermediates.m1_vv.as_ref().unwrap();
    let m1b_ov = intermediates.m1b_ov.as_ref().unwrap();
    let m2a_ov = intermediates.m2a_ov.as_ref().unwrap();

    let device = rhs2.device().clone();

    // === TERM 1 === //
    // Loo = (
    //     + np.einsum("kcP, icP -> ik", B[so, sv], M2a)
    //     - np.einsum("liP, lkP -> ik", B[so, so], M1oo)
    //     + np.einsum("ikP, P -> ik", 2 * B[so, so] + M1oo, M1j))
    // RHS2 -= np.einsum("ik, kjab -> ijab", Loo, t2)

    let mut l_oo = m2a_ov.reshape((nocc, -1)) % b_ov.reshape((nocc, -1)).t();
    let scr: Tsr = 2 * b_oo + m1_oo;
    l_oo += (scr.reshape((-1, naux)) % m1_j).into_shape((nocc, nocc));
    for l in 0..nocc {
        l_oo -= b_oo.i(l) % m1_oo.i(l).t();
    }

    // the following line is intutive, but memory footprint is too large
    // rhs2 -= (l_oo % t2.reshape((nocc, -1))).into_shape((nocc, nocc, nvir, nvir));

    // change mutable tensor shape, then use memory-friendly `matmul_from`
    let mut rhs2 = rt::asarray((rhs2.raw_mut(), [nocc, nocc * nvir * nvir].c(), &device));
    rhs2.matmul_from(&l_oo, &t2.reshape((nocc, -1)), -1.0, 1.0);

    // Lvv = (
    //     - 1 * np.einsum("kcP, kaP -> ac", B[so, sv], M2a + M1bov)
    //     + 1 * np.einsum("acP, P -> ac", 2 * B[sv, sv] - M1vv, M1j))
    // RHS2 += np.einsum("ac, ijcb -> ijab", Lvv, t2)

    let scr: Tsr = 2 * b_vv - m1_vv;
    let mut l_vv = (scr.reshape((-1, naux)) % m1_j).into_shape((nvir, nvir));
    for k in 0..nocc {
        l_vv -= (m2a_ov.i(k) + m1b_ov.i(k)) % b_ov.i(k).t();
    }

    // the following line is intutive, but memory footprint is too large
    // rhs2 += (t2.reshape((-1, nvir)) % l_vv.t()).into_shape((nocc, nocc, nvir, nvir));

    // change mutable tensor shape, then use memory-friendly `matmul_from`
    let mut rhs2 = rt::asarray((rhs2.raw_mut(), [nocc * nocc * nvir, nvir].c(), &device));
    rhs2.matmul_from(&t2.reshape((-1, nvir)), &l_vv.t(), 1.0, 1.0);
}

pub fn get_riccsd_rhs2_direct_dot(mol_info: &RCCSDInfo, rhs2: TsrMut, intermediates: &RCCSDIntermediates) {
    let nocc = mol_info.nocc();

    let b_ov = intermediates.b_ov.as_ref().unwrap();
    let m1a_ov = intermediates.m1a_ov.as_ref().unwrap();
    let m1b_ov = intermediates.m1b_ov.as_ref().unwrap();
    let m2a_ov = intermediates.m2a_ov.as_ref().unwrap();
    let m2b_ov = intermediates.m2b_ov.as_ref().unwrap();

    // RHS2 += np.einsum("iaP, jbP -> ijab", B[so, sv] + M2a, 0.5 * B[so, sv] + 0.5 * M2a - M1aov + M1bov - M2b)
    // RHS2 -= np.einsum("iaP, jbP -> ijab", M1aov, M1bov)

    let scr_iaP: Tsr = b_ov + m2a_ov;
    let scr_jbP: Tsr = 0.5 * b_ov + 0.5 * m2a_ov - m1a_ov + m1b_ov - m2b_ov;

    (0..nocc).into_par_iter().for_each(|i| {
        (0..nocc).into_par_iter().for_each(|j| {
            let mut rhs2 = unsafe { rhs2.force_mut() };
            *&mut rhs2.i_mut((i, j)) += scr_iaP.i(i) % scr_jbP.i(j).t();
            *&mut rhs2.i_mut((i, j)) -= m1a_ov.i(i) % m1b_ov.i(j).t();
        });
    });
}

pub fn get_riccsd_rhs2_o3v3(mol_info: &RCCSDInfo, rhs2: TsrMut, intermediates: &RCCSDIntermediates, t2: TsrView) {
    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();

    let b_ov = intermediates.b_ov.as_ref().unwrap();
    let b_oo = intermediates.b_oo.as_ref().unwrap();
    let b_vv = intermediates.b_vv.as_ref().unwrap();
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();
    let m1_vv = intermediates.m1_vv.as_ref().unwrap();

    let device = b_ov.device().clone();

    // t2t = np.ascontiguousarray(t2.swapaxes(-1, -2))
    // scr1 = np.einsum("kdP, lcP -> kldc", B[so, sv], B[so, sv])
    // scr2 = np.einsum("kldc, ilda -> ikca", scr1, t2)
    // scr3 = np.einsum("kldc, ilda -> ikca", scr1, t2t)
    // scr4 = np.einsum("ikP, acP -> ikca", (B[so, so] + M1oo), (B[sv, sv] - M1vv))
    // RHS2 += np.einsum("ikca, jkcb -> ijab", - scr4 + scr2 - scr3, t2t)
    // RHS2 += np.einsum("ikcb, jkca -> ijab", - scr4 + 0.5 * scr2, t2)

    let scr_ikP = b_oo + m1_oo;
    let scr_acP = b_vv - m1_vv;

    let scr1 = unsafe { rt::empty(([nocc, nocc, nvir, nvir], &device)) };
    (0..nocc).into_par_iter().for_each(|k| {
        (0..k + 1).into_par_iter().for_each(|l| {
            let mut scr1 = unsafe { scr1.force_mut() };
            let scr = b_ov.i(k) % b_ov.i(l).t();
            scr1.i_mut((k, l)).assign(&scr);
            if k != l {
                scr1.i_mut((l, k)).assign(scr.t());
            }
        });
    });

    (0..nocc).into_par_iter().for_each(|i| {
        let t2t_i = t2.swapaxes(0, 1).i(i).into_contig(RowMajor);
        let scr2 = unsafe { rt::empty(([nocc, nvir, nvir], &device)) };
        let scr3 = unsafe { rt::empty(([nocc, nvir, nvir], &device)) };
        let scr4 = unsafe { rt::empty(([nocc, nvir, nvir], &device)) };
        (0..nocc).into_par_iter().for_each(|k| {
            let mut scr2 = unsafe { scr2.force_mut() };
            let mut scr3 = unsafe { scr3.force_mut() };
            scr2.i_mut(k).matmul_from(&scr1.i(k).reshape((-1, nvir)).t(), &t2.i(i).reshape((-1, nvir)), 1.0, 0.0);
            scr3.i_mut(k).matmul_from(&scr1.i(k).reshape((-1, nvir)).t(), &t2t_i.reshape((-1, nvir)), 1.0, 0.0);
        });
        (0..nvir).into_par_iter().for_each(|c| {
            let mut scr4 = unsafe { scr4.force_mut() };
            scr4.i_mut((.., c)).matmul_from(&scr_ikP.i(i), &scr_acP.i((.., c)).t(), 1.0, 0.0);
        });
        let scr5 = -scr3 - &scr4 + &scr2;
        let scr6 = -scr4 + 0.5_f64 * &scr2;
        (0..nocc).into_par_iter().for_each(|j| {
            let mut rhs2 = unsafe { rhs2.force_mut() };
            for k in 0..nocc {
                rhs2.i_mut((i, j)).matmul_from(&scr5.i(k).t(), &t2.i((k, j)), 1.0, 1.0);
                rhs2.i_mut((i, j)).matmul_from(&t2.i((j, k)).t(), &scr6.i(k), 1.0, 1.0);
            }
        });
    });
}

pub fn get_riccsd_rhs2_o4v2(
    mol_info: &RCCSDInfo,
    mut rhs2: TsrMut,
    intermediates: &RCCSDIntermediates,
    t1: TsrView,
    t2: TsrView,
) {
    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();

    let b_ov = intermediates.b_ov.as_ref().unwrap();
    let b_oo = intermediates.b_oo.as_ref().unwrap();
    let m1_oo = intermediates.m1_oo.as_ref().unwrap();

    let device = b_ov.device().clone();

    // === O4V2 (HHL) === //
    // Woooo = 0
    // Woooo += np.einsum("ikP, jlP         -> ijkl", B[so, so] + M1oo, B[so, so] + M1oo)
    // Woooo += np.einsum("kcP, ldP, ijcd   -> ijkl", B[so, sv], B[so, sv], t2)
    // RHS2 += 0.5 * np.einsum("ijkl, klab   -> ijab", Woooo, t2,   )
    // RHS2 += 0.5 * np.einsum("ijkl, ka, lb -> ijab", Woooo, t1, t1)

    let scr_bm1_oo = b_oo + m1_oo;

    let mut scr_ijkl: Tsr = rt::zeros(([nocc, nocc, nocc, nocc], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        (0..i + 1).into_par_iter().for_each(|j| {
            let mut scr_ijkl = unsafe { scr_ijkl.force_mut() };
            let scr = scr_bm1_oo.i(i) % scr_bm1_oo.i(j).t();
            *&mut scr_ijkl.i_mut((i, j)) += &scr;
            if i != j {
                *&mut scr_ijkl.i_mut((j, i)) += scr.t();
            }
        });
    });

    let scr_klcd: Tsr = rt::zeros(([nocc, nocc, nvir, nvir], &device));
    (0..nocc).into_par_iter().for_each(|k| {
        (0..k + 1).into_par_iter().for_each(|l| {
            let mut scr_klcd = unsafe { scr_klcd.force_mut() };
            let scr = b_ov.i(k) % b_ov.i(l).t();
            *&mut scr_klcd.i_mut((k, l)) += &scr;
            if k != l {
                *&mut scr_klcd.i_mut((l, k)) += scr.t();
            }
        });
    });

    // scr_ijkl += (t2.reshape((nocc * nocc, -1)) % scr_klcd.reshape((nocc * nocc, -1)).t()).reshape(scr_ijkl.shape());
    let mut scr_ijkl_mut = rt::asarray((scr_ijkl.raw_mut(), [nocc * nocc, nocc * nocc].c(), &device));
    scr_ijkl_mut.matmul_from(&t2.reshape((nocc * nocc, -1)), &scr_klcd.reshape((nocc * nocc, -1)).t(), 1.0, 1.0);

    // the following line is intutive, but memory footprint is too large
    // let tau2 = &t2; // + t1.i((.., None, .., None)) * t1.i((None, .., None, ..));
    // rhs2 += 0.5 * (scr_ijkl.reshape((nocc * nocc, -1)) % tau2.reshape((-1, nvir * nvir))).into_shape(t2.shape());

    // t1 contraction to rhs2
    (0..nocc).into_par_iter().for_each(|i| {
        (0..nocc).into_par_iter().for_each(|j| {
            let mut rhs2 = unsafe { rhs2.force_mut() };
            *&mut rhs2.i_mut((i, j)) += 0.5 * t1.t() % scr_ijkl.i((i, j)) % &t1;
        });
    });

    // t2 contraction to rhs2
    let mut rhs2 = rt::asarray((rhs2.raw_mut(), [nocc * nocc, nvir * nvir].c(), &device));
    rhs2.matmul_from(&scr_ijkl.reshape((nocc * nocc, -1)), &t2.reshape((-1, nvir * nvir)), 0.5, 1.0);
}

pub fn get_riccsd_rhs2_o2v4(
    mol_info: &RCCSDInfo,
    rhs2: TsrMut,
    intermediates: &RCCSDIntermediates,
    t1: TsrView,
    t2: TsrView,
) {
    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();

    let b_vv = intermediates.b_vv.as_ref().unwrap();
    let m1_vv = intermediates.m1_vv.as_ref().unwrap();

    let device = b_vv.device().clone();

    // ====== O2V4 (PPL) ====== //
    // Wvvvv = 0
    // Wvvvv += np.einsum("acP, bdP -> abcd", B[sv, sv] - M1vv, B[sv, sv] - M1vv)
    // Wvvvv -= np.einsum("acP, bdP -> abcd", M1vv, M1vv)
    // RHS2 += 0.5 * np.einsum("abcd, ijcd   -> ijab", Wvvvv, t2,   )
    // RHS2 += 0.5 * np.einsum("abcd, ic, jd -> ijab", Wvvvv, t1, t1)

    // batch initialization
    let nbatch_a = 8;
    let nbatch_b = 32;
    let mut batched_slices = vec![];
    for a_start in (0..nvir).step_by(nbatch_a) {
        let a_end = (a_start + nbatch_a).min(nvir);
        for b_start in (0..a_end).step_by(nbatch_b) {
            let b_end = (b_start + nbatch_b).min(a_end);
            batched_slices.push(([a_start, a_end], [b_start, b_end]));
        }
    }

    batched_slices.into_par_iter().for_each(|([a_start, a_end], [b_start, b_end])| {
        let nbatch_a = a_end - a_start;
        let nbatch_b = b_end - b_start;

        // generate Wvvvv in batch (a, b)
        let mut scr_abcd: Tsr = rt::zeros(([nbatch_a, nbatch_b, nvir, nvir], &device));
        for a in 0..nbatch_a {
            let a_shift = a + a_start;
            let scr_a_cP = b_vv.i(a_shift) - m1_vv.i(a_shift);
            for b in 0..nbatch_b {
                let b_shift = b + b_start;
                let scr_b_dP = b_vv.i(b_shift) - m1_vv.i(b_shift);
                scr_abcd.i_mut((a, b)).matmul_from(&scr_a_cP, &scr_b_dP.t(), 1.0, 1.0);
                scr_abcd.i_mut((a, b)).matmul_from(&m1_vv.i(a_shift), &m1_vv.i(b_shift).t(), -1.0, 1.0);
            }
        }

        // t2 contribution from tau2
        let mut scr_ijab: Tsr = unsafe { rt::empty(([nocc * nocc, nbatch_a * nbatch_b], &device)) };
        scr_ijab.matmul_from(
            &t2.reshape((nocc * nocc, -1)),
            &scr_abcd.reshape((nbatch_a * nbatch_b, -1)).t(),
            0.5,
            0.0,
        );

        // t1 contribution from tau2
        let mut scr_ijab = scr_ijab.into_shape((nocc, nocc, nbatch_a, nbatch_b));
        for a in 0..nbatch_a {
            for b in 0..nbatch_b {
                *&mut scr_ijab.i_mut((.., .., a, b)) += 0.5 * &t1 % scr_abcd.i((a, b)) % t1.t();
            }
        }

        // increment rhs2 from scr_ijab with symmetry
        for i in 0..nocc {
            for j in 0..nocc {
                let rhs2_ij = rhs2.i((i, j)).into_dim::<Ix2>();
                let rhs2_ji = rhs2.i((j, i)).into_dim::<Ix2>();
                let mut rhs2_ij = unsafe { rhs2_ij.force_mut() };
                let mut rhs2_ji = unsafe { rhs2_ji.force_mut() };
                let scr_ij = scr_ijab.i((i, j)).into_dim::<Ix2>();

                for a in 0..nbatch_a {
                    let a_shift = a + a_start;
                    for b in 0..nbatch_b {
                        let b_shift = b + b_start;
                        unsafe {
                            if a_shift > b_shift {
                                let scr_val = scr_ij.index_uncheck([a, b]);
                                *rhs2_ij.index_mut_uncheck([a_shift, b_shift]) += scr_val;
                                *rhs2_ji.index_mut_uncheck([b_shift, a_shift]) += scr_val;
                            } else if a_shift == b_shift {
                                *rhs2_ij.index_mut_uncheck([a_shift, b_shift]) += scr_ij.index_uncheck([a, b]);
                            }
                        }
                    }
                }
            }
        }
    });
}

pub fn get_riccsd_initial_guess(mol_info: &RCCSDInfo, intermediates: &RCCSDIntermediates) -> RCCSDResults {
    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();
    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let device = mol_info.mo_coeff.device().clone();

    let b_ov = intermediates.b_ov.as_ref().unwrap();
    let mo_energy = &mol_info.mo_energy;
    let d_ov = mo_energy.i((so, None)) - mo_energy.i((None, sv));

    let t1 = rt::zeros(([nocc, nvir], &device));
    let t2 = rt::zeros(([nocc, nocc, nvir, nvir], &device));

    let e_corr_oo = rt::zeros(([nocc, nocc], &device));
    (0..nocc).into_par_iter().for_each(|i| {
        (0..i + 1).into_par_iter().for_each(|j| {
            let g2_ab = b_ov.i(i) % b_ov.i(j).t();
            let d2_ab = d_ov.i((i, .., None)) + d_ov.i((j, None, ..));
            let t2_ab = &g2_ab / &d2_ab;
            let mut t2 = unsafe { t2.force_mut() };
            t2.i_mut((i, j)).assign(&t2_ab);
            if i != j {
                t2.i_mut((j, i)).assign(&t2_ab.t());
            }
            let e_bi1 = (&t2_ab * &g2_ab).sum_all();
            let e_bi2 = (&t2_ab * &g2_ab.swapaxes(-1, -2)).sum_all();
            let e_corr_ij = 2.0 * e_bi1 - e_bi2;
            let mut e_corr_oo = unsafe { e_corr_oo.force_mut() };
            *e_corr_oo.index_mut([i, j]) = e_corr_ij;
            *e_corr_oo.index_mut([j, i]) = e_corr_ij;
        });
    });

    let e_corr = e_corr_oo.sum_all();

    RCCSDResults { t1, t2, e_corr }
}

pub fn get_amplitude_from_rhs(mol_info: &RCCSDInfo, rhs1: Tsr, rhs2: Tsr) -> (Tsr, Tsr) {
    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();
    let so = slice!(0, nocc);
    let sv = slice!(nocc, nocc + nvir);

    let mo_energy = &mol_info.mo_energy;

    let d_ov = mo_energy.i((so, None)) - mo_energy.i((None, sv));
    let t1_new = rhs1 / &d_ov;
    // let t2_new = (&rhs2 + rhs2.transpose((1, 0, 3, 2))) / (d_ov.i((.., None, .., None)) + d_ov.i((None, .., None, ..)));
    let t2_new = rhs2;
    (0..nocc).into_par_iter().for_each(|i| {
        (0..i + 1).into_par_iter().for_each(|j| {
            let mut t2_new = unsafe { t2_new.force_mut() };
            let t2_ab = t2_new.i((i, j)) + t2_new.i((j, i)).t();
            let d2_ab = d_ov.i((i, .., None)) + d_ov.i((j, None, ..));
            let t2_ab = t2_ab / d2_ab;
            t2_new.i_mut((i, j)).assign(&t2_ab);
            if i != j {
                t2_new.i_mut((j, i)).assign(&t2_ab.t());
            }
        });
    });

    (t1_new, t2_new)
}

pub fn diis_concate_amplitude(t1: TsrView, t2: TsrView) -> Tsr {
    let nocc = t1.shape()[0];
    let nvir = t1.shape()[1];
    let mut cat = rt::zeros(([nocc * nvir + nocc * nvir * nocc * nvir], t1.device()));
    cat.i_mut(..nocc * nvir).assign(t1.reshape(-1));
    cat.i_mut(nocc * nvir..).assign(t2.reshape(-1));
    cat
}

pub fn diis_split_amplitude(cat: TsrView, nocc: usize, nvir: usize) -> (Tsr, Tsr) {
    let t1 = cat.i(..nocc * nvir).into_shape((nocc, nvir));
    let t2 = cat.i(nocc * nvir..).into_shape((nocc, nocc, nvir, nvir));
    (t1, t2)
}

pub fn update_riccsd_amplitude(
    mol_info: &RCCSDInfo,
    intermediates: &mut RCCSDIntermediates,
    cc_info: &RCCSDResults,
    diis_obj: &mut DIISIncore,
) -> RCCSDResults {
    let timer_outer = std::time::Instant::now();

    let t1 = &cc_info.t1;
    let t2 = &cc_info.t2;
    let mut rhs1 = t1.zeros_like();
    let mut rhs2 = t2.zeros_like();

    let timer = std::time::Instant::now();
    get_riccsd_intermediates_2(mol_info, intermediates, t1.view());
    println!("Time elapsed (intermediates_2): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs1(mol_info, rhs1.view_mut(), intermediates, t1.view(), t2.view());
    println!("Time elapsed (rhs1): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs2_lt2_contract(mol_info, rhs2.view_mut(), intermediates, t2.view());
    println!("Time elapsed (rhs2 lt2_contract): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs2_direct_dot(mol_info, rhs2.view_mut(), intermediates);
    println!("Time elapsed (rhs2 direct_dot): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs2_o3v3(mol_info, rhs2.view_mut(), intermediates, t2.view());
    println!("Time elapsed (rhs2 o3v3): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs2_o4v2(mol_info, rhs2.view_mut(), intermediates, t1.view(), t2.view());
    println!("Time elapsed (rhs2 o4v2): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_rhs2_o2v4(mol_info, rhs2.view_mut(), intermediates, t1.view(), t2.view());
    println!("Time elapsed (rhs2 o2v4): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    let (t1_new, t2_new) = get_amplitude_from_rhs(mol_info, rhs1, rhs2);
    println!("Time elapsed (update_t): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    let t_vec = diis_concate_amplitude(t1_new.view(), t2_new.view());
    let t_vec = diis_obj.update(t_vec, None, None);
    let (t1_new, t2_new) = diis_split_amplitude(t_vec.view(), mol_info.nocc(), mol_info.nvir());
    println!("Time elapsed (diis update): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    get_riccsd_intermediates_1(mol_info, intermediates, t1_new.view(), t2_new.view());
    println!("Time elapsed (intermediates_1): {:?}", timer.elapsed());

    let timer = std::time::Instant::now();
    let e_corr = get_riccsd_energy(intermediates);
    println!("Time elapsed (energy): {:?}", timer.elapsed());

    let result = RCCSDResults { t1: t1_new, t2: t2_new, e_corr };

    println!("Time elapsed (ccsd amplitude): {:?}", timer_outer.elapsed());

    result
}

pub fn riccsd_iteration(mol_info: &RCCSDInfo, cc_config: &CCSDConfig) -> (RCCSDResults, RCCSDIntermediates) {
    let time_outer = std::time::Instant::now();

    // cderi ao2mo
    let timer = std::time::Instant::now();
    let mut intermediates = RCCSDIntermediates::default();
    get_riccsd_intermediates_cderi(mol_info, &mut intermediates);
    println!("Time elapsed (cderi in mo): {:?}", timer.elapsed());

    // initial guess
    let timer = std::time::Instant::now();
    let mut ccsd_results = get_riccsd_initial_guess(mol_info, &intermediates);
    println!("Initial energy (MP2): {:?}", ccsd_results.e_corr);
    println!("Time elapsed (initial guess): {:?}", timer.elapsed());

    // diis preparation
    let device = mol_info.mo_coeff.device().clone();
    let mut diis_obj = DIISIncore::new(DIISIncoreFlags::default(), &device);

    // intermediates_1 should be initialized first before iteration
    let timer = std::time::Instant::now();
    get_riccsd_intermediates_1(mol_info, &mut intermediates, ccsd_results.t1.view(), ccsd_results.t2.view());
    println!("Time elapsed (initialization of intermediates_1): {:?}", timer.elapsed());

    for niter in 0..cc_config.max_cycle {
        let timer = std::time::Instant::now();
        println!("Iteration: {:?}", niter);
        let ccsd_results_new = update_riccsd_amplitude(mol_info, &mut intermediates, &ccsd_results, &mut diis_obj);

        println!("    Energy: {:?}", ccsd_results_new.e_corr);
        let diff_eng = ccsd_results_new.e_corr - ccsd_results.e_corr;
        let norm_t1 = (&ccsd_results_new.t1 - &ccsd_results.t1).l2_norm();
        let norm_t2 = (&ccsd_results_new.t2 - &ccsd_results.t2).l2_norm();
        println!("    Energy diff: {:?}", diff_eng);
        println!("    T1 norm: {:?}", norm_t1);
        println!("    T2 norm: {:?}", norm_t2);

        if diff_eng.abs() < cc_config.conv_tol_e && norm_t1 < cc_config.conv_tol_t1 && norm_t2 < cc_config.conv_tol_t2 {
            println!("CCSD converged in {niter} iterations.");
            println!("Time elapsed (CCSD total time): {:?}", time_outer.elapsed());
            return (ccsd_results_new, intermediates);
        }
        ccsd_results = ccsd_results_new;

        println!("Time elapsed (ccsd iteration): {:?}", timer.elapsed());
    }

    panic!("[fatal] CCSD did not converge.");
}

/* #endregion */
