use crate::prelude::*;

/* #region RHF */

pub struct RHFResults {
    pub mo_coeff: Tsr,
    pub mo_energy: Tsr,
    pub dm: Tsr,
    pub e_nuc: f64,
    pub e_elec: f64,
    pub e_tot: f64,
}

/* #endregion */

/* #region RI-RCCSD */

#[derive(Debug)]
pub struct RCCSDInfo {
    pub cint_data: CInt,
    pub aux_cint_data: CInt,
    pub mo_coeff: Tsr,
    pub mo_energy: Tsr,
}

#[derive(Debug, Default)]
pub struct RCCSDIntermediates {
    pub b_oo: Option<Tsr>,
    pub b_ov: Option<Tsr>,
    pub b_vv: Option<Tsr>,
    pub m1_j: Option<Tsr>,
    pub m1_oo: Option<Tsr>,
    pub m1_vv: Option<Tsr>,
    pub m1a_ov: Option<Tsr>,
    pub m1b_ov: Option<Tsr>,
    pub m2a_ov: Option<Tsr>,
    pub m2b_ov: Option<Tsr>,
}

#[derive(Debug)]
pub struct RCCSDResults {
    pub e_corr: f64,
    pub t1: Tsr,
    pub t2: Tsr,
}

pub struct CCSDConfig {
    pub max_cycle: usize,
    pub conv_tol_e: f64,
    pub conv_tol_t1: f64,
    pub conv_tol_t2: f64,
}

impl Default for CCSDConfig {
    fn default() -> Self {
        Self { max_cycle: 64, conv_tol_e: 1.0e-7, conv_tol_t1: 1.0e-5, conv_tol_t2: 1.0e-5 }
    }
}

impl RCCSDInfo {
    pub fn nmo(&self) -> usize {
        self.mo_coeff.shape()[1]
    }

    pub fn nao(&self) -> usize {
        self.mo_coeff.shape()[0]
    }

    pub fn nocc(&self) -> usize {
        (self.cint_data.atom_charges().into_iter().sum::<f64>() / 2.0) as usize
    }

    pub fn nvir(&self) -> usize {
        self.nmo() - self.nocc()
    }

    pub fn naux(&self) -> usize {
        self.aux_cint_data.nao()
    }
}

/* #endregion */

/* #region RCCSD(T) */

#[derive(Default)]
pub struct RCCSDTIntermediates {
    pub t1_t: Option<Tsr>,
    pub t2_t: Option<Tsr>,
    pub eri_vvov_t: Option<Tsr>,
    pub eri_vooo_t: Option<Tsr>,
    pub eri_vvoo_t: Option<Tsr>,
    pub d_ooo: Option<Tsr>,
}

#[derive(Debug)]
pub struct RCCSDTResults {
    pub e_corr_pt: f64,
}

/* #endregion */
