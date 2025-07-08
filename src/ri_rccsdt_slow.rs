use crate::prelude::*;
use std::sync::{Arc, Mutex};

pub(crate) fn prepare_intermediates(
    mol_info: &RCCSDInfo,
    ccsd_intermediates: &RCCSDIntermediates,
    ccsd_results: &RCCSDResults,
) -> RCCSDTIntermediates {
    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();
    let mo_energy = mol_info.mo_energy.view();

    let t1 = ccsd_results.t1.view();
    let t2 = ccsd_results.t2.view();
    let b_oo = ccsd_intermediates.b_oo.as_ref().unwrap();
    let b_ov = ccsd_intermediates.b_ov.as_ref().unwrap();
    let b_vv = ccsd_intermediates.b_vv.as_ref().unwrap();
    let eo = mo_energy.i(..nocc);
    let device = t2.device().clone();

    // t1_t = t1.transpose(1, 0)
    let t1_t = t1.t().into_contig(RowMajor);

    // t2_t = t2.transpose(2, 3, 1, 0)
    let t2_t = t2.transpose((2, 3, 1, 0)).into_contig(RowMajor);

    // eri_vvov_t = np.einsum("iaP, bcP -> acib", cderi_mo[so, sv], cderi_mo[sv, sv])
    let eri_vvov_t = unsafe { rt::empty(([nvir, nvir, nocc, nvir], &device)) };
    (0..nvir).into_par_iter().for_each(|a| {
        (0..nvir).into_par_iter().for_each(|c| {
            let mut eri_vvov_t = unsafe { eri_vvov_t.force_mut() };
            eri_vvov_t.i_mut([a, c]).matmul_from(&b_ov.i((.., a)), &b_vv.i((.., c)).t(), 1.0, 0.0);
        });
    });

    // eri_vooo_t = np.einsum("ljP, kaP -> aljk", cderi_mo[so, so], cderi_mo[so, sv])
    let eri_vooo_t = unsafe { rt::empty(([nvir, nocc, nocc, nocc], &device)) };
    (0..nvir).into_par_iter().for_each(|a| {
        (0..nocc).into_par_iter().for_each(|l| {
            let mut eri_vooo_t = unsafe { eri_vooo_t.force_mut() };
            eri_vooo_t.i_mut([a, l]).matmul_from(&b_oo.i(l), &b_ov.i((.., a)).t(), 1.0, 0.0);
        });
    });

    // eri_vvoo_t = np.einsum("iaP, jbP -> abij", cderi_mo[so, sv], cderi_mo[so, sv])
    let eri_vvoo_t = unsafe { rt::empty(([nvir, nvir, nocc, nocc], &device)) };
    (0..nvir).into_par_iter().for_each(|a| {
        (0..nvir).into_par_iter().for_each(|b| {
            let mut eri_vvoo_t = unsafe { eri_vvoo_t.force_mut() };
            eri_vvoo_t.i_mut([a, b]).matmul_from(&b_ov.i((.., a)), &b_ov.i((.., b)).t(), 1.0, 0.0);
        });
    });

    // d_ooo = eo[None, None, :] + eo[None, :, None] + eo[:, None, None]
    let d_ooo = eo.i((.., None, None)) + eo.i((None, .., None)) + eo.i((None, None, ..));

    RCCSDTIntermediates {
        t1_t: Some(t1_t),
        t2_t: Some(t2_t),
        eri_vvov_t: Some(eri_vvov_t),
        eri_vooo_t: Some(eri_vooo_t),
        eri_vvoo_t: Some(eri_vvoo_t),
        d_ooo: Some(d_ooo),
    }
}

fn get_w(abc: [usize; 3], intermediates: &RCCSDTIntermediates) -> Tsr {
    // + np.einsum("id, djk -> ijk", eri_vvov_t[a, b], t2_t[c])
    // - np.einsum("ljk, li -> ijk", eri_vooo_t[c], t2_t[a, b])

    let t2_t = intermediates.t2_t.as_ref().unwrap();
    let eri_vvov_t = intermediates.eri_vvov_t.as_ref().unwrap();
    let eri_vooo_t = intermediates.eri_vooo_t.as_ref().unwrap();
    let nvir = t2_t.shape()[0];
    let nocc = t2_t.shape()[2];

    let [a, b, c] = abc;

    let mut w = eri_vvov_t.i([a, b]) % t2_t.i(c).reshape([nvir, nocc * nocc]);
    w.matmul_from(&t2_t.i([a, b]).t(), &eri_vooo_t.i(c).reshape([nocc, nocc * nocc]), -1.0, 1.0);
    w.into_shape([nocc, nocc, nocc])
}

fn ccsd_t_energy_contribution(abc: [usize; 3], mol_info: &RCCSDInfo, intermediates: &RCCSDTIntermediates) -> f64 {
    let nocc = mol_info.nocc();
    let t1_t = intermediates.t1_t.as_ref().unwrap();
    let eri_vvoo_t = intermediates.eri_vvoo_t.as_ref().unwrap();
    let d_ooo = intermediates.d_ooo.as_ref().unwrap();
    let ev = mol_info.mo_energy.i(nocc..);

    let [a, b, c] = abc;

    let w = get_w([a, b, c], intermediates)
        + get_w([a, c, b], intermediates).transpose([0, 2, 1])
        + get_w([b, c, a], intermediates).transpose([2, 0, 1])
        + get_w([b, a, c], intermediates).transpose([1, 0, 2])
        + get_w([c, a, b], intermediates).transpose([1, 2, 0])
        + get_w([c, b, a], intermediates).transpose([2, 1, 0]);
    let v = &w
        + t1_t.i((a, .., None, None)) * eri_vvoo_t.i([b, c]).i((None, .., ..))
        + t1_t.i((b, None, .., None)) * eri_vvoo_t.i([c, a]).t().i((.., None, ..))
        + t1_t.i((c, None, None, ..)) * eri_vvoo_t.i([a, b]).i((.., .., None));
    let d = -(ev[[a]] + ev[[b]] + ev[[c]]) + d_ooo;
    let z = 4.0 * &w + w.transpose([1, 2, 0]) + w.transpose([2, 0, 1])
        - 2.0 * w.transpose([2, 1, 0])
        - 2.0 * w.transpose([0, 2, 1])
        - 2.0 * w.transpose([1, 0, 2]);
    let e_tsr: Tsr = (z * v) / d;

    let fac = if a == c {
        1.0 / 3.0
    } else if a == b || b == c {
        1.0
    } else {
        2.0
    };

    fac * e_tsr.sum()
}

pub fn get_riccsd_pt_energy(
    mol_info: &RCCSDInfo,
    ccsd_intermediates: &RCCSDIntermediates,
    ccsd_results: &RCCSDResults,
) -> RCCSDTResults {
    let nvir = mol_info.nvir();

    let timer = std::time::Instant::now();
    let intermediates = prepare_intermediates(mol_info, ccsd_intermediates, ccsd_results);
    let abc_list =
        (0..nvir).flat_map(|a| (0..a + 1).flat_map(move |b| (0..b + 1).map(move |c| [a, b, c]))).collect::<Vec<_>>();
    println!("Time elapsed (CCSD(T) preparation): {:?}", timer.elapsed());

    let e_corr_pt = Arc::new(Mutex::new(0.0));

    let timer = std::time::Instant::now();
    abc_list.into_par_iter().for_each(|abc| {
        let e_pt_inc = ccsd_t_energy_contribution(abc, mol_info, &intermediates);
        *e_corr_pt.lock().unwrap() += e_pt_inc;
    });
    println!("Time elapsed (CCSD(T) computation): {:?}", timer.elapsed());

    let e_corr_pt = *e_corr_pt.lock().unwrap();
    RCCSDTResults { e_corr_pt }
}
