pub mod prelude;

pub mod rhf;
pub mod util;

#[test]
fn playground() {
    use crate::prelude::*;

    let cint_data = init_h2o_def2_tzvp();
    rhf::minimal_rhf(&cint_data);
}
