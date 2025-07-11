use crate::prelude::*;

pub struct TransposedIndices {
    pub tr_012: Vec<usize>,
    pub tr_021: Vec<usize>,
    pub tr_102: Vec<usize>,
    pub tr_120: Vec<usize>,
    pub tr_201: Vec<usize>,
    pub tr_210: Vec<usize>,
}

fn prepare_transposed_indices(nocc: usize) -> TransposedIndices {
    let device = DeviceTsr::default();
    let base = rt::arange((nocc * nocc * nocc, &device)).into_shape([nocc, nocc, nocc]);
    let tr_012 = base.transpose([0, 1, 2]).reshape(-1).to_vec();
    let tr_021 = base.transpose([0, 2, 1]).reshape(-1).to_vec();
    let tr_102 = base.transpose([1, 0, 2]).reshape(-1).to_vec();
    let tr_120 = base.transpose([1, 2, 0]).reshape(-1).to_vec();
    let tr_201 = base.transpose([2, 0, 1]).reshape(-1).to_vec();
    let tr_210 = base.transpose([2, 1, 0]).reshape(-1).to_vec();

    TransposedIndices { tr_012, tr_021, tr_102, tr_120, tr_201, tr_210 }
}

fn get_w(abc: [usize; 3], mut w: TsrMut, wbuf: TsrMut, intermediates: &RCCSDTIntermediates, tr_indices: &[usize]) {
    // + np.einsum("id, djk -> ijk", eri_vvov_t[a, b], t2_t[c])
    // - np.einsum("ljk, li -> ijk", eri_vooo_t[c], t2_t[a, b])

    let t2_t = intermediates.t2_t.as_ref().unwrap();
    let eri_vvov_t = intermediates.eri_vvov_t.as_ref().unwrap();
    let eri_vooo_t = intermediates.eri_vooo_t.as_ref().unwrap();
    let nvir = t2_t.shape()[0];
    let nocc = t2_t.shape()[2];

    let [a, b, c] = abc;

    // hack for TsrMut to reshape to nocc, nocc * nocc
    let wbuf_layout = [nocc, nocc * nocc].c();
    let (wbuf_storage, _) = wbuf.into_raw_parts();
    let mut wbuf = TsrMut::new(wbuf_storage, wbuf_layout);
    wbuf.matmul_from(&eri_vvov_t.i([a, b]), &t2_t.i(c).reshape([nvir, nocc * nocc]), 1.0, 0.0);
    wbuf.matmul_from(&t2_t.i([a, b]).t(), &eri_vooo_t.i(c).reshape([nocc, nocc * nocc]), -1.0, 1.0);

    // add to w with transposed indices
    let w_raw = w.raw_mut();
    let wbuf_raw = wbuf.raw();
    w_raw.iter_mut().zip(tr_indices).for_each(|(w_elem, &tr_idx)| {
        *w_elem += unsafe { wbuf_raw.get_unchecked(tr_idx) };
    });
}

fn ccsd_t_energy_contribution(
    abc: [usize; 3],
    mol_info: &RCCSDInfo,
    intermediates: &RCCSDTIntermediates,
    tr_indices: &TransposedIndices,
) -> f64 {
    let nocc = mol_info.nocc();
    let t1_t = intermediates.t1_t.as_ref().unwrap();
    let eri_vvoo_t = intermediates.eri_vvoo_t.as_ref().unwrap();
    let d_ooo = intermediates.d_ooo.as_ref().unwrap();
    let ev = mol_info.mo_energy.i(nocc..);
    let device = t1_t.device().clone();

    let [a, b, c] = abc;

    let mut w = rt::zeros(([nocc, nocc, nocc], &device));
    let mut wbuf = unsafe { rt::empty(([nocc, nocc, nocc], &device)) };

    get_w([a, b, c], w.view_mut(), wbuf.view_mut(), intermediates, &tr_indices.tr_012);
    get_w([a, c, b], w.view_mut(), wbuf.view_mut(), intermediates, &tr_indices.tr_021);
    get_w([b, c, a], w.view_mut(), wbuf.view_mut(), intermediates, &tr_indices.tr_201);
    get_w([b, a, c], w.view_mut(), wbuf.view_mut(), intermediates, &tr_indices.tr_102);
    get_w([c, a, b], w.view_mut(), wbuf.view_mut(), intermediates, &tr_indices.tr_120);
    get_w([c, b, a], w.view_mut(), wbuf.view_mut(), intermediates, &tr_indices.tr_210);

    let eri_vvoo_t_ab = eri_vvoo_t.i([a, b]).into_dim::<Ix2>();
    let eri_vvoo_t_bc = eri_vvoo_t.i([b, c]).into_dim::<Ix2>();
    let eri_vvoo_t_ca = eri_vvoo_t.i([c, a]).into_dim::<Ix2>();
    let t1_t_a = t1_t.i(a).into_dim::<Ix1>().to_owned();
    let t1_t_b = t1_t.i(b).into_dim::<Ix1>().to_owned();
    let t1_t_c = t1_t.i(c).into_dim::<Ix1>().to_owned();
    let iter_ijk = (0..nocc).cartesian_product(0..nocc).cartesian_product(0..nocc);
    let d_abc = -(ev[[a]] + ev[[b]] + ev[[c]]);
    let w_raw = w.raw();

    let e_sum = izip!(
        iter_ijk,
        w.raw().iter(),
        d_ooo.raw().iter(),
        tr_indices.tr_012.iter(),
        tr_indices.tr_120.iter(),
        tr_indices.tr_201.iter(),
        tr_indices.tr_210.iter(),
        tr_indices.tr_021.iter(),
        tr_indices.tr_102.iter()
    )
    .fold(0.0, |acc, (((i, j), k), &w_val, &d_ijk, &tr_012, &tr_120, &tr_201, &tr_210, &tr_021, &tr_102)| unsafe {
        let v_val = w_val
            + t1_t_a.raw().get_unchecked(i) * eri_vvoo_t_bc.index_uncheck([j, k])
            + t1_t_b.raw().get_unchecked(j) * eri_vvoo_t_ca.index_uncheck([k, i])
            + t1_t_c.raw().get_unchecked(k) * eri_vvoo_t_ab.index_uncheck([i, j]);
        let z_val = 4.0 * w_raw.get_unchecked(tr_012) + w_raw.get_unchecked(tr_120) + w_raw.get_unchecked(tr_201)
            - 2.0 * (w_raw.get_unchecked(tr_210) + w_raw.get_unchecked(tr_021) + w_raw.get_unchecked(tr_102));
        let d_val = d_abc + d_ijk;
        acc + z_val * v_val / d_val
    });

    let fac = if a == c {
        1.0 / 3.0
    } else if a == b || b == c {
        1.0
    } else {
        2.0
    };

    fac * e_sum
}

pub fn get_riccsd_pt_energy(
    mol_info: &RCCSDInfo,
    ccsd_intermediates: &RCCSDIntermediates,
    ccsd_results: &RCCSDResults,
) -> RCCSDTResults {
    let time_outer = std::time::Instant::now();

    let nvir = mol_info.nvir();
    let nocc = mol_info.nocc();

    // prepare intermediates
    let timer = std::time::Instant::now();
    let intermediates = ri_rccsdt_slow::prepare_intermediates(mol_info, ccsd_intermediates, ccsd_results);
    let abc_list =
        (0..nvir).flat_map(|a| (0..a + 1).flat_map(move |b| (0..b + 1).map(move |c| [a, b, c]))).collect::<Vec<_>>();
    println!("Time elapsed (CCSD(T) preparation): {:?}", timer.elapsed());

    // prepare transposed indices
    let timer = std::time::Instant::now();
    let tr_indices = prepare_transposed_indices(nocc);
    println!("Time elapsed (transposed indices preparation): {:?}", timer.elapsed());

    let e_corr_pt = Arc::new(Mutex::new(0.0));

    let timer = std::time::Instant::now();
    // this code can also turned into `fold`
    abc_list.into_par_iter().for_each(|abc| {
        let e_pt_inc = ccsd_t_energy_contribution(abc, mol_info, &intermediates, &tr_indices);
        *e_corr_pt.lock().unwrap() += e_pt_inc;
    });
    println!("Time elapsed (CCSD(T) contraction): {:?}", timer.elapsed());
    println!("Time elapsed (CCSD(T) total time): {:?}", time_outer.elapsed());

    let e_corr_pt = *e_corr_pt.lock().unwrap();
    RCCSDTResults { e_corr_pt }
}
