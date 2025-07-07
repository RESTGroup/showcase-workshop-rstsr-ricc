from pyscf import gto, scf
import numpy as np

# # Single H2O/def2-TZVP

mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="def2-TZVP").build()
with open("h2o-tzvp.json", "w") as f:
    f.write(mol.dumps())

aux = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="def2-universal-jkfit").build()
with open("h2o-def2_jk.json", "w") as f:
    f.write(aux.dumps())

mf = scf.RHF(mol).density_fit(aux.basis).run()

np.save("h2o-mo_coeff.npy", np.asarray(mf.mo_coeff, order="C"))
np.save("h2o-mo_energy.npy", mf.mo_energy)

# # Single H2O-PP5/cc-pVDZ

atom_token = """
    O -2.21165 0.99428 -1.34761
    H -1.39146 1.51606 -1.47747
    H -1.97320 0.08049 -1.61809
    O 0.09403 2.29278 1.59474
    H 0.12603 2.53877 0.64902
    H -0.74393 1.78978 1.67135
    O -1.36387 -1.68942 -1.58413
    H -1.87986 -2.36904 -2.04608
    H -1.51808 -1.85775 -0.60321
    O 1.15753 -1.98493 1.42883
    H 1.51336 -1.05256 1.56992
    H 1.63126 -2.54067 2.06706
    O 2.16234 0.46384 1.59959
    H 1.45220 1.14162 1.73767
    H 2.44819 0.61600 0.67631
    O 0.26320 2.39844 -1.29615
    H 1.04651 1.79827 -1.38236
    H 0.46651 3.18119 -1.83082
    O 1.44377 -1.86519 -1.36370
    H 0.48945 -1.86011 -1.60072
    H 1.44320 -2.10978 -0.41122
    O -1.62831 -1.98091 1.04938
    H -1.92768 -1.08892 1.33229
    H -0.69028 -2.03600 1.33896
    O 2.35473 0.62384 -1.26848
    H 3.15897 0.65726 -1.80967
    H 2.00663 -0.31760 -1.36507
    O -2.29362 0.74293 1.32406
    H -2.34790 0.87628 0.33220
    H -3.13510 1.07144 1.67759
"""

mol = gto.Mole(atom=atom_token, basis="cc-pVDZ", max_memory=32000).build()
with open("h2o_pp5-pvdz.json", "w") as f:
    f.write(mol.dumps())

aux_jk = gto.Mole(atom=atom_token, basis="cc-pVDZ-jkfit").build()
with open("h2o_pp5-pvdz_jk.json", "w") as f:
    f.write(aux_jk.dumps())

aux = gto.Mole(atom=atom_token, basis="cc-pVDZ-ri").build()
with open("h2o_pp5-pvdz_ri.json", "w") as f:
    f.write(aux.dumps())

mf = scf.RHF(mol).density_fit(aux_jk.basis).run()

np.save("h2o_pp5-mo_coeff.npy", np.asarray(mf.mo_coeff, order="C"))
np.save("h2o_pp5-mo_energy.npy", mf.mo_energy)
