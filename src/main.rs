use clap::{Parser, Subcommand};
use libcint::prelude::*;
use showcase_workshop_rstsr_ricc::prelude::*;
use showcase_workshop_rstsr_ricc::*;

#[derive(Parser, Debug)]
#[clap(version, about = "Minimal showcase of Rust quantum chemistry programming: RHF, RI-RHF, RI-CCSD, RI-CCSD(T)", long_about = None)]
struct CliParser {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    #[clap(about = "Run a minimal RHF calculation (slow, convention 4c-2e ERI, no advanced convergence techniques)")]
    RhfSlow {
        #[clap(short, long = "mol", help = "Molecular data file with basis set (json)")]
        mol_file: String,
    },
    #[clap(about = "Run a minimal RI-RHF calculation")]
    RiRhf {
        #[clap(short, long = "mol", help = "Molecular data file with basis set (json)")]
        mol_file: String,
        #[clap(short, long = "aux", help = "Molecular data file with auxiliary basis set (json)")]
        aux_file: String,
        #[clap(short, long, help = "Run the slower version of RI-RHF")]
        slow: bool,
    },
    #[clap(name = "ri-rccsd", about = "Run a minimal RI-CCSD calculation")]
    RiRccsd {
        #[clap(short, long = "mol", help = "Molecular data file with basis set (json)")]
        mol_file: String,
        #[clap(short, long = "aux", help = "Molecular data file with auxiliary basis set (json)")]
        aux_file: String,
        #[clap(long = "aux-ri", help = "Molecular data file with auxiliary basis set for post-HF (json)")]
        aux_ri_file: Option<String>,
        #[clap(long, help = "Run standalone CCSD by `mo_coeff` and `mo_energy` provided by user")]
        standalone: bool,
        #[clap(long = "mo_coeff", help = "Molecular orbital coefficients file (npy)")]
        mo_coeff_file: Option<String>,
        #[clap(long = "mo_energy", help = "Molecular orbital energy file (npy)")]
        mo_energy_file: Option<String>,
    },
    #[clap(name = "ri-rccsdt", about = "Run a minimal RI-CCSD(T) calculation")]
    RiRccsdt {
        #[clap(short, long = "mol", help = "Molecular data file with basis set (json)")]
        mol_file: String,
        #[clap(short, long = "aux", help = "Molecular data file with auxiliary basis set (json)")]
        aux_file: String,
        #[clap(long = "aux-ri", help = "Molecular data file with auxiliary basis set for post-HF (json)")]
        aux_ri_file: Option<String>,
        #[clap(short, long, help = "Run the slower (T) perturbation of RI-RCCSD(T)")]
        slow: bool,
        #[clap(long, help = "Run standalone CCSD and CCSD(T) by `mo_coeff` and `mo_energy` provided by user")]
        standalone: bool,
        #[clap(long = "mo_coeff", help = "Molecular orbital coefficients file (npy)")]
        mo_coeff_file: Option<String>,
        #[clap(long = "mo_energy", help = "Molecular orbital energy file (npy)")]
        mo_energy_file: Option<String>,
    },
}

fn main() {
    let args = CliParser::parse();

    match args.command {
        Command::RhfSlow { mol_file } => {
            let cint_data = CInt::from_json(&mol_file);
            rhf_slow::minimal_rhf(&cint_data);
        },
        Command::RiRhf { mol_file, aux_file, slow } => {
            let cint_data = CInt::from_json(&mol_file);
            let aux_cint_data = CInt::from_json(&aux_file);
            match slow {
                true => ri_rhf_slow::minimal_ri_rhf(&cint_data, &aux_cint_data),
                false => ri_rhf::minimal_ri_rhf(&cint_data, &aux_cint_data),
            };
        },
        Command::RiRccsd { mol_file, aux_file, aux_ri_file, standalone, mo_coeff_file, mo_energy_file } => {
            let cint_data = CInt::from_json(&mol_file);
            let aux_cint_data = CInt::from_json(&aux_file);
            let ri_cint_data = aux_ri_file.map_or(aux_cint_data.clone(), |f| CInt::from_json(&f));
            let (mo_coeff, mo_energy) = match standalone {
                true => {
                    if mo_coeff_file.is_none() || mo_energy_file.is_none() {
                        panic!("Standalone mode requires both `mo_coeff` and `mo_energy` files.");
                    }
                    let mo_coeff = util::tensor_from_file(&mo_coeff_file.unwrap());
                    let mo_energy = util::tensor_from_file(&mo_energy_file.unwrap());
                    (mo_coeff, mo_energy)
                },
                false => {
                    let rhf_results = ri_rhf::minimal_ri_rhf(&cint_data, &ri_cint_data);
                    (rhf_results.mo_coeff, rhf_results.mo_energy)
                },
            };
            let ccsd_info = RCCSDInfo { cint_data, aux_cint_data: ri_cint_data, mo_coeff, mo_energy };
            let cc_config = CCSDConfig::default();
            let (ccsd_results, _) = ri_rccsd::riccsd_iteration(&ccsd_info, &cc_config);
            println!("CCSD Corr Energy: {}", ccsd_results.e_corr);
        },
        Command::RiRccsdt { mol_file, aux_file, aux_ri_file, slow, standalone, mo_coeff_file, mo_energy_file } => {
            let cint_data = CInt::from_json(&mol_file);
            let aux_cint_data = CInt::from_json(&aux_file);
            let ri_cint_data = aux_ri_file.map_or(aux_cint_data.clone(), |f| CInt::from_json(&f));
            let (mo_coeff, mo_energy) = match standalone {
                true => {
                    if mo_coeff_file.is_none() || mo_energy_file.is_none() {
                        panic!("Standalone mode requires both `mo_coeff` and `mo_energy` files.");
                    }
                    let mo_coeff = util::tensor_from_file(&mo_coeff_file.unwrap());
                    let mo_energy = util::tensor_from_file(&mo_energy_file.unwrap());
                    (mo_coeff, mo_energy)
                },
                false => {
                    let rhf_results = ri_rhf::minimal_ri_rhf(&cint_data, &ri_cint_data);
                    (rhf_results.mo_coeff, rhf_results.mo_energy)
                },
            };
            let ccsd_info = RCCSDInfo { cint_data, aux_cint_data: ri_cint_data, mo_coeff, mo_energy };
            let cc_config = CCSDConfig::default();
            let (ccsd_results, ccsd_intrm) = ri_rccsd::riccsd_iteration(&ccsd_info, &cc_config);
            println!("CCSD Corr Energy: {}", ccsd_results.e_corr);
            match slow {
                true => {
                    let ccsdt_results = ri_rccsdt_slow::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
                    println!("CCSD(T) Corr Energy: {}", ccsdt_results.e_corr_pt);
                },
                false => {
                    let ccsdt_results = ri_rccsdt::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
                    println!("CCSD(T) Corr Energy: {}", ccsdt_results.e_corr_pt);
                },
            }
        },
    }
}
