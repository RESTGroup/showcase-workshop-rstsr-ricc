#![allow(clippy::deref_addrof)]

pub mod prelude;

pub mod diis;
pub mod rhf_slow;
pub mod ri_rccsd;
pub mod ri_rccsdt;
pub mod ri_rccsdt_slow;
pub mod ri_rhf;
pub mod ri_rhf_slow;
pub mod structs;
pub mod util;

#[test]
fn playground_rhf() {
    use crate::prelude::*;

    let cint_data = init_h2o_def2_tzvp();
    rhf_slow::minimal_rhf(&cint_data);
}

#[test]
fn playground_ri_rhf_slow() {
    use crate::prelude::*;

    let cint_data = init_h2o_def2_tzvp();
    let aux_cint_data = init_h2o_def2_jk();
    ri_rhf_slow::minimal_ri_rhf(&cint_data, &aux_cint_data);
}

#[test]
fn playground_ri_rhf() {
    use crate::prelude::*;

    let cint_data = init_h2o_def2_tzvp();
    let aux_cint_data = init_h2o_def2_jk();
    ri_rhf::minimal_ri_rhf(&cint_data, &aux_cint_data);
}

#[test]
fn playground_ri_ccsd() {
    use crate::prelude::*;

    let cint_data = init_h2o_def2_tzvp();
    let aux_cint_data = init_h2o_def2_jk();
    let rhf_results = ri_rhf::minimal_ri_rhf(&cint_data, &aux_cint_data);
    let ccsd_info = RCCSDInfo {
        cint_data,
        aux_cint_data,
        mo_coeff: rhf_results.mo_coeff.clone(),
        mo_energy: rhf_results.mo_energy.clone(),
    };
    let cc_config = CCSDConfig::default();

    let (ccsd_results, _) = ri_rccsd::riccsd_iteration(&ccsd_info, &cc_config);
    println!("CCSD Corr Energy: {}", ccsd_results.e_corr);
}

#[test]
fn playground_ri_ccsdt() {
    use crate::prelude::*;

    let cint_data = CInt::from_json("assets/h2o_5-pvdz.json");
    let aux_cint_data = CInt::from_json("assets/h2o_5-pvdz_ri.json");
    let rhf_results = ri_rhf::minimal_ri_rhf(&cint_data, &aux_cint_data);

    let ccsd_info = RCCSDInfo {
        cint_data,
        aux_cint_data,
        mo_coeff: rhf_results.mo_coeff.clone(),
        mo_energy: rhf_results.mo_energy.clone(),
    };

    // do not run actual CCSD, but use faked T1 amplitudes
    let mut ccsd_intrm = RCCSDIntermediates::default();
    ri_rccsd::get_riccsd_intermediates_cderi(&ccsd_info, &mut ccsd_intrm);
    let mut ccsd_results = ri_rccsd::get_riccsd_initial_guess(&ccsd_info, &ccsd_intrm);
    ccsd_results.t1 =
        rt::arange((ccsd_results.t1.size() as f64, ccsd_results.t1.device())).into_shape(ccsd_results.t1.shape());

    println!("======");

    let ccsdt_results = ri_rccsdt_slow::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
    println!("CCSD(T) Perturb Energy: {}", ccsdt_results.e_corr_pt);

    println!("======");

    let ccsdt_results = ri_rccsdt::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
    println!("CCSD(T) Perturb Energy: {}", ccsdt_results.e_corr_pt);
}
