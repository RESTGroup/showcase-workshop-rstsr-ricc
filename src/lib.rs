pub mod prelude;

pub mod rhf;
pub mod util;

#[test]
fn playground_1() {
    use crate::prelude::*;

    let cint_data = init_h2o_def2_tzvp();
    rhf::minimal_rhf(&cint_data);
}

#[test]
fn playground_2() {
    use crate::prelude::*;

    let cint_data = init_h2o_def2_tzvp();
    let aux_cint_data = init_h2o_def2_jk();
    rhf::minimal_ri_rhf(&cint_data, &aux_cint_data);
}
