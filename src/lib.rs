#![allow(clippy::deref_addrof)]

pub mod prelude;

pub mod diis;
pub mod rhf_slow;
pub mod ri_rccsd;
pub mod ri_rccsdt;
pub mod ri_rccsdt_naive;
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

    let h2o_tzvp_initializer = r#"
        atom = "O; H 1 0.94; H 1 0.94 2 104.5"
        basis = "def2-tzvp"
    "#;
    let h2o_def2_jk_initializer = r#"
        atom = "O; H 1 0.94; H 1 0.94 2 104.5"
        basis = "def2-universal-jkfit"
    "#;
    let cint_data = CIntMol::from_toml(h2o_tzvp_initializer).cint;
    let aux_cint_data = CIntMol::from_toml(h2o_def2_jk_initializer).cint;
    let rhf_results = ri_rhf::minimal_ri_rhf(&cint_data, &aux_cint_data);

    // pyscf reference:
    // mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="def2-TZVP").build()
    // mf = scf.RHF(mol).density_fit().run()
    assert!((rhf_results.e_tot - (-76.0594504483582)).abs() < 1e-6);

    let ccsd_info = RCCSDInfo {
        cint_data,
        aux_cint_data,
        mo_coeff: rhf_results.mo_coeff.clone(),
        mo_energy: rhf_results.mo_energy.clone(),
    };
    let ccsd_config = CCSDConfig::default();

    let (ccsd_results, ccsd_intrm) = ri_rccsd::riccsd_iteration(&ccsd_info, &ccsd_config);
    println!("CCSD Corr Energy: {}", ccsd_results.e_corr);

    // pyscf reference:
    // mf_cc = cc.CCSD(mf).run()
    assert!((ccsd_results.e_corr - (-0.2782191383258219)).abs() < 1e-6);

    println!("======");

    let ccsdt_results = ri_rccsdt_naive::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
    println!("CCSD(T) Perturb Energy (naive algorithm): {}", ccsdt_results.e_corr_pt);

    // pyscf reference:
    // mf_cc.ccsd_t()
    assert!((ccsdt_results.e_corr_pt - (-0.0069126908356167)).abs() < 1e-6);

    println!("======");

    let ccsdt_results = ri_rccsdt_slow::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
    println!("CCSD(T) Perturb Energy (slow  algorithm): {}", ccsdt_results.e_corr_pt);
    assert!((ccsdt_results.e_corr_pt - (-0.0069126908356167)).abs() < 1e-6);

    println!("======");

    let ccsdt_results = ri_rccsdt::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
    println!("CCSD(T) Perturb Energy (fast  algorithm): {}", ccsdt_results.e_corr_pt);
    assert!((ccsdt_results.e_corr_pt - (-0.0069126908356167)).abs() < 1e-6);
}

#[test]
fn playground_ri_ccsdt_efficiency() {
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
    let ccsd_config = CCSDConfig::default();

    let (ccsd_results, ccsd_intrm) = ri_rccsd::riccsd_iteration(&ccsd_info, &ccsd_config);
    println!("CCSD Corr Energy: {}", ccsd_results.e_corr);

    println!("======");

    let ccsdt_results = ri_rccsdt_slow::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
    println!("CCSD(T) Perturb Energy (slow  algorithm): {}", ccsdt_results.e_corr_pt);

    println!("======");

    let ccsdt_results = ri_rccsdt::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
    println!("CCSD(T) Perturb Energy (fast  algorithm): {}", ccsdt_results.e_corr_pt);
}
