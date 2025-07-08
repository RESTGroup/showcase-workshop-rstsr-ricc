use clap::{Args, Parser, Subcommand};
use libcint::prelude::*;
use rstsr::prelude::*;
use showcase_workshop_rstsr_ricc::prelude::*;
use showcase_workshop_rstsr_ricc::*;

#[derive(Parser, Debug)]
#[clap(version, about, long_about = None)]
struct CliParser {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Args, Debug)]
struct CliArgs {
    #[clap(short, long = "mol", help = "Path to the json file containing molecular data")]
    mol_file: String,

    #[clap(short, long = "aux", help = "Path to the json file containing auxiliary basis data")]
    aux_file: String,
}

#[derive(Args, Debug)]
struct CliArgsStandalone {
    #[clap(short, long = "mol", help = "Path to the json file containing molecular data")]
    mol_file: String,

    #[clap(short, long = "aux", help = "Path to the json file containing auxiliary basis data")]
    aux_file: String,

    #[clap(long = "mo_coeff", help = "Number of SCF iterations")]
    mo_coeff_file: String,

    #[clap(long = "mo_energy", help = "Number of SCF iterations")]
    mo_energy_file: String,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[clap(name = "rhf", about = "Run a minimal RHF calculation")]
    Rhf {
        #[clap(short, long = "mol", help = "Path to the json file containing molecular data")]
        mol_file: String,
    },

    #[clap(name = "ri-rhf", about = "Run a minimal RI-RHF calculation")]
    RiRhf(CliArgs),

    #[clap(name = "ri-rhf-faster", about = "Run a minimal RI-RHF calculation (faster than ri-rhf)")]
    RiRhfFaster(CliArgs),

    #[clap(name = "ri-rccsd", about = "Run a minimal RI-CCSD calculation")]
    RiRccsd(CliArgs),

    #[clap(name = "ri-rccsdt", about = "Run a minimal RI-CCSD(T) calculation")]
    RiRccsdt(CliArgs),

    #[clap(name = "ri-rccsd-standalone", about = "Run a minimal RI-CCSD calculation with SCF data provided by user")]
    RiRccsdStandalone(CliArgsStandalone),

    #[clap(
        name = "ri-rccsdt-standalone",
        about = "Run a minimal RI-CCSD(T) calculation with SCF data provided by user"
    )]
    RiRccsdtStandalone(CliArgsStandalone),
}

fn tensor_from_file(fname: &str) -> Tsr {
    // c-contiguous numpy array to f-contiguous rstsr
    let device = DeviceTsr::default();
    let bytes = std::fs::read(fname).unwrap();
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape = npy.shape().iter().map(|x| *x as usize).rev().collect::<Vec<usize>>();
    let data = npy.into_vec().unwrap();
    rt::asarray((data, shape, &device))
}

fn main() {
    let args = CliParser::parse();

    match args.command {
        Command::Rhf { mol_file } => {
            let cint_data = CInt::from_json(&mol_file);
            rhf::minimal_rhf(&cint_data);
        },
        Command::RiRhf(cli_args) => {
            let time = std::time::Instant::now();
            let cint_data = CInt::from_json(&cli_args.mol_file);
            let aux_cint_data = CInt::from_json(&cli_args.aux_file);
            rhf::minimal_ri_rhf(&cint_data, &aux_cint_data);
            println!("Elapsed time for RI-RHF: {:.2?}", time.elapsed());
        },
        Command::RiRhfFaster(cli_args) => {
            let time = std::time::Instant::now();
            let cint_data = CInt::from_json(&cli_args.mol_file);
            let aux_cint_data = CInt::from_json(&cli_args.aux_file);
            rhf::minimal_ri_rhf_faster(&cint_data, &aux_cint_data);
            println!("Elapsed time for RI-RHF (faster): {:.2?}", time.elapsed());
        },
        Command::RiRccsd(cli_args) => {
            let cint_data = CInt::from_json(&cli_args.mol_file);
            let aux_cint_data = CInt::from_json(&cli_args.aux_file);
            let rhf_results = rhf::minimal_ri_rhf(&cint_data, &aux_cint_data);
            let ccsd_info = RCCSDInfo {
                cint_data,
                aux_cint_data,
                mo_coeff: rhf_results.mo_coeff,
                mo_energy: rhf_results.mo_energy,
            };
            let cc_config = CCSDConfig::default();
            let (ccsd_results, _) = ccsd::naive_riccsd_iteration(&ccsd_info, &cc_config);
            println!("CCSD Corr Energy: {}", ccsd_results.e_corr);
        },
        Command::RiRccsdt(cli_args) => {
            let cint_data = CInt::from_json(&cli_args.mol_file);
            let aux_cint_data = CInt::from_json(&cli_args.aux_file);
            let rhf_results = rhf::minimal_ri_rhf(&cint_data, &aux_cint_data);
            let ccsd_info = RCCSDInfo {
                cint_data,
                aux_cint_data,
                mo_coeff: rhf_results.mo_coeff,
                mo_energy: rhf_results.mo_energy,
            };
            let cc_config = CCSDConfig::default();
            let (ccsd_results, ccsd_intrm) = ccsd::naive_riccsd_iteration(&ccsd_info, &cc_config);
            println!("CCSD Corr Energy: {}", ccsd_results.e_corr);
            let ccsdt_results = ccsdt::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
            println!("CCSD(T) Perturb Energy: {}", ccsdt_results.e_corr_pt);
        },
        Command::RiRccsdStandalone(cli_args) => {
            let cint_data = CInt::from_json(&cli_args.mol_file);
            let aux_cint_data = CInt::from_json(&cli_args.aux_file);
            let mo_coeff = tensor_from_file(&cli_args.mo_coeff_file);
            let mo_energy = tensor_from_file(&cli_args.mo_energy_file);
            let ccsd_info = RCCSDInfo { cint_data, aux_cint_data, mo_coeff, mo_energy };
            let cc_config = CCSDConfig::default();
            let (ccsd_results, _) = ccsd::naive_riccsd_iteration(&ccsd_info, &cc_config);
            println!("CCSD Corr Energy: {}", ccsd_results.e_corr);
        },
        Command::RiRccsdtStandalone(cli_args) => {
            let cint_data = CInt::from_json(&cli_args.mol_file);
            let aux_cint_data = CInt::from_json(&cli_args.aux_file);
            let mo_coeff = tensor_from_file(&cli_args.mo_coeff_file);
            let mo_energy = tensor_from_file(&cli_args.mo_energy_file);
            let ccsd_info = RCCSDInfo { cint_data, aux_cint_data, mo_coeff, mo_energy };
            let cc_config = CCSDConfig::default();
            let (ccsd_results, ccsd_intrm) = ccsd::naive_riccsd_iteration(&ccsd_info, &cc_config);
            println!("CCSD Corr Energy: {}", ccsd_results.e_corr);
            let ccsdt_results = ccsdt::get_riccsd_pt_energy(&ccsd_info, &ccsd_intrm, &ccsd_results);
            println!("CCSD(T) Perturb Energy: {}", ccsdt_results.e_corr_pt);
        },
    }
}
