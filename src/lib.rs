#![allow(clippy::deref_addrof)]

pub mod prelude;

pub mod ccsd;
pub mod ccsdt;
pub mod ccsdt_faster;
pub mod rhf;
pub mod util;

#[test]
fn playground_rhf() {
    use crate::prelude::*;

    let cint_data = init_h2o_def2_tzvp();
    rhf::minimal_rhf(&cint_data);
}

#[test]
fn playground_ri_rhf() {
    use crate::prelude::*;

    let cint_data = init_h2o_def2_tzvp();
    let aux_cint_data = init_h2o_def2_jk();
    rhf::minimal_ri_rhf(&cint_data, &aux_cint_data);
}

#[test]
fn playground_ri_rhf_faster() {
    use crate::prelude::*;

    let cint_data = init_h2o_def2_tzvp();
    let aux_cint_data = init_h2o_def2_jk();
    rhf::minimal_ri_rhf_faster(&cint_data, &aux_cint_data);
}

#[test]
fn playground_ri_ccsd() {
    use crate::prelude::*;

    let cint_data = init_h2o_def2_tzvp();
    let aux_cint_data = init_h2o_def2_jk();
    let rhf_results = rhf::minimal_ri_rhf(&cint_data, &aux_cint_data);
    let ccsd_info = RCCSDInfo {
        cint_data,
        aux_cint_data,
        mo_coeff: rhf_results.mo_coeff.clone(),
        mo_energy: rhf_results.mo_energy.clone(),
    };
    let cc_config = CCSDConfig::default();

    let (ccsd_results, ccsd_intrm) = ccsd::naive_riccsd_iteration(&ccsd_info, &cc_config);
    println!("CCSD Corr Energy: {}", ccsd_results.e_corr);

    let ccsdt_results = ccsdt::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
    println!("CCSD(T) Perturb Energy: {}", ccsdt_results.e_corr_pt);
}

#[test]
fn playground_ri_ccsd_faster() {
    use crate::prelude::*;

    let cint_data = CInt::from_json("assets/h2o_5-pvdz.json");
    let aux_cint_data = CInt::from_json("assets/h2o_5-pvdz_ri.json");
    let rhf_results = rhf::minimal_ri_rhf_faster(&cint_data, &aux_cint_data);

    let ccsd_info = RCCSDInfo {
        cint_data,
        aux_cint_data,
        mo_coeff: rhf_results.mo_coeff.clone(),
        mo_energy: rhf_results.mo_energy.clone(),
    };

    // do not run actual CCSD, but use faked T1 amplitudes
    let mut ccsd_intrm = ccsd::RCCSDIntermediates::default();
    ccsd::get_riccsd_intermediates_cderi(&ccsd_info, &mut ccsd_intrm);
    let mut ccsd_results = ccsd::get_riccsd_initial_guess(&ccsd_info, &ccsd_intrm);
    ccsd_results.t1 =
        rt::arange((ccsd_results.t1.size() as f64, ccsd_results.t1.device())).into_shape(ccsd_results.t1.shape());

    println!("======");

    let ccsdt_results = ccsdt::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
    println!("CCSD(T) Perturb Energy: {}", ccsdt_results.e_corr_pt);

    println!("======");

    let ccsdt_results = ccsdt_faster::get_riccsd_pt_energy_faster(&ccsd_info, &ccsd_intrm, &ccsd_results);
    println!("CCSD(T) Perturb Energy: {}", ccsdt_results.e_corr_pt);
}
