use crate::prelude::*;

pub fn get_riccsd_pt_energy(
    mol_info: &RCCSDInfo,
    ccsd_intermediates: &RCCSDIntermediates,
    ccsd_results: &RCCSDResults,
) -> RCCSDTResults {
    let time_outer = std::time::Instant::now();

    let nocc = mol_info.nocc();
    let nvir = mol_info.nvir();
    let nmo = nocc + nvir;
    let mo_energy = &mol_info.mo_energy;

    let time = std::time::Instant::now();
    let intermediates = ri_rccsdt_slow::prepare_intermediates(mol_info, ccsd_intermediates, ccsd_results);
    println!("Time elapsed (CCSD(T) preparation): {:?}", time.elapsed());

    let t1_t = intermediates.t1_t.as_ref().unwrap();
    let t2_t = intermediates.t2_t.as_ref().unwrap();
    let eri_vvoo_t = intermediates.eri_vvoo_t.as_ref().unwrap();
    let eri_vooo_t = intermediates.eri_vooo_t.as_ref().unwrap();
    let eri_vvov_t = intermediates.eri_vvov_t.as_ref().unwrap();

    let device = t1_t.device().clone();

    let time = std::time::Instant::now();

    let wp: Tsr = unsafe { rt::empty(([nvir, nvir, nvir, nocc, nocc, nocc], &device)) };
    (0..nvir).into_par_iter().for_each(|a| {
        (0..nvir).into_par_iter().for_each(|b| {
            (0..nvir).into_par_iter().for_each(|c| unsafe {
                let mut wp = wp.force_mut();
                let wp_1 = eri_vvov_t.i((a, b)) % t2_t.i(c).reshape((nvir, -1));
                let wp_2 = t2_t.i((a, b)).t() % eri_vooo_t.i(c).reshape((nocc, -1));
                wp.i_mut([a, b, c]).assign(wp_1.reshape((nocc, nocc, nocc)) - wp_2.reshape((nocc, nocc, nocc)));
            });
        });
    });
    let w = wp.transpose((0, 1, 2, 3, 4, 5))
        + wp.transpose((0, 2, 1, 3, 5, 4))
        + wp.transpose((1, 0, 2, 4, 3, 5))
        + wp.transpose((1, 2, 0, 4, 5, 3))
        + wp.transpose((2, 0, 1, 5, 3, 4))
        + wp.transpose((2, 1, 0, 5, 4, 3));
    let v = &w
        + t1_t.i((.., None, None, .., None, None)) * eri_vvoo_t.i((None, .., .., None, .., ..))
        + t1_t.i((None, .., None, None, .., None)) * eri_vvoo_t.i((.., None, .., .., None, ..))
        + t1_t.i((None, None, .., None, None, ..)) * eri_vvoo_t.i((.., .., None, .., .., None));
    let (so, sv) = (slice!(0, nocc), slice!(nocc, nmo));
    let d = -mo_energy.i((sv, None, None, None, None, None))
        - mo_energy.i((None, sv, None, None, None, None))
        - mo_energy.i((None, None, sv, None, None, None))
        + mo_energy.i((None, None, None, so, None, None))
        + mo_energy.i((None, None, None, None, so, None))
        + mo_energy.i((None, None, None, None, None, so));
    let wt = 4.0_f64 * &w + w.transpose((1, 2, 0, 3, 4, 5)) + w.transpose((2, 0, 1, 3, 4, 5));
    let vt = &v - v.transpose((2, 1, 0, 3, 4, 5));
    let e_corr_pt = (wt * vt / &d).sum() / 3.0;

    println!("Time elapsed (CCSD(T) computation): {:?}", time.elapsed());
    println!("Total time elapsed (CCSD(T) energy): {:?}", time_outer.elapsed());
    println!("Time elapsed (CCSD(T) energy): {:?}", time.elapsed());

    RCCSDTResults { e_corr_pt }
}
