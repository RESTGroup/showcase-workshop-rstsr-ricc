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

# # Single H2O-10/cc-pVDZ

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
with open("h2o_10-pvdz.json", "w") as f:
    f.write(mol.dumps())

aux_jk = gto.Mole(atom=atom_token, basis="cc-pVDZ-jkfit").build()
with open("h2o_10-pvdz_jk.json", "w") as f:
    f.write(aux_jk.dumps())

aux = gto.Mole(atom=atom_token, basis="cc-pVDZ-ri").build()
with open("h2o_10-pvdz_ri.json", "w") as f:
    f.write(aux.dumps())

mf = scf.RHF(mol).density_fit(aux_jk.basis).run()

np.save("h2o_10-mo_coeff.npy", np.asarray(mf.mo_coeff, order="C"))
np.save("h2o_10-mo_energy.npy", mf.mo_energy)

# # Single H2O-7/cc-pVDZ

atom_token = """
    O -2.77257 -0.57180 0.70154
    H -3.23538 -0.03573 1.36151
    H -1.94145 -0.87427 1.14505
    O 1.37997 1.06749 1.27100
    H 1.95244 1.50168 1.92278
    H 1.99684 0.58547 0.65389
    O -0.25860 2.56362 -0.48322
    H -0.90075 3.04130 0.06298
    H 0.34514 2.13013 0.15958
    O -1.42709 0.25065 -1.59401
    H -2.06468 0.03791 -0.87376
    H -1.05375 1.12456 -1.33845
    O -0.25971 -1.30874 1.57997
    H 0.00453 -1.64438 0.69789
    H 0.25243 -0.47948 1.66479
    O 2.81930 -0.44526 -0.45874
    H 2.04387 -0.87234 -0.90037
    H 3.27241 -1.17835 -0.01350
    O 0.43333 -1.57718 -1.18253
    H 0.26658 -2.29765 -1.81026
    H -0.25527 -0.86910 -1.40431
"""

mol = gto.Mole(atom=atom_token, basis="cc-pVDZ", max_memory=32000).build()
with open("h2o_7-pvdz.json", "w") as f:
    f.write(mol.dumps())

aux_jk = gto.Mole(atom=atom_token, basis="cc-pVDZ-jkfit").build()
with open("h2o_7-pvdz_jk.json", "w") as f:
    f.write(aux_jk.dumps())

aux = gto.Mole(atom=atom_token, basis="cc-pVDZ-ri").build()
with open("h2o_7-pvdz_ri.json", "w") as f:
    f.write(aux.dumps())

mf = scf.RHF(mol).density_fit(aux_jk.basis).run()

np.save("h2o_7-mo_coeff.npy", np.asarray(mf.mo_coeff, order="C"))
np.save("h2o_7-mo_energy.npy", mf.mo_energy)

# # Single H2O-6/cc-pVDZ

atom_token="""
    H -0.834068 2.203626 0.038578
    O 0.000000 2.738219 0.012580
    H -0.096011 3.286402 -0.780636
    H 1.491362 1.824138 -0.038578
    O -2.371367 1.369109 -0.012580
    H -2.894113 1.560054 0.780636
    H -2.325431 0.379489 -0.038578
    O -2.371367 -1.369109 0.012580
    H -1.491362 -1.824138 0.038578
    H -2.798103 -1.726349 -0.780636
    O 0.000000 -2.738219 -0.012580
    H 0.096011 -3.286402 0.780636
    H 0.834068 -2.203626 -0.038578
    O 2.371367 -1.369109 0.012580
    H 2.325431 -0.379489 0.038578
    H 2.894113 -1.560054 -0.780636
    O 2.371367 1.369109 -0.012580
    H 2.798103 1.726349 0.78063
"""

mol = gto.Mole(atom=atom_token, basis="cc-pVDZ", max_memory=32000).build()
with open("h2o_6-pvdz.json", "w") as f:
    f.write(mol.dumps())

aux_jk = gto.Mole(atom=atom_token, basis="cc-pVDZ-jkfit").build()
with open("h2o_6-pvdz_jk.json", "w") as f:
    f.write(aux_jk.dumps())

aux = gto.Mole(atom=atom_token, basis="cc-pVDZ-ri").build()
with open("h2o_6-pvdz_ri.json", "w") as f:
    f.write(aux.dumps())

mf = scf.RHF(mol).density_fit(aux_jk.basis).run()

np.save("h2o_6-mo_coeff.npy", np.asarray(mf.mo_coeff, order="C"))
np.save("h2o_6-mo_energy.npy", mf.mo_energy)

# # Single H2O-5/cc-pVDZ

atom_token="""
    H -0.976098 -1.670678 0.027772
    H 2.586991 -1.244628 0.797861
    O -0.221668 -2.313473 0.011992
    O -2.287920 -0.513039 -0.033895
    H -1.935308 0.411273 -0.085374
    H -2.825368 -0.529412 0.771884
    O -1.178836 2.005715 -0.091977
    H -1.377787 2.606032 0.641794
    H -0.191861 1.916016 -0.073380
    O 1.547425 1.742939 0.048267
    H 1.781909 0.779735 0.069448
    H 1.978262 2.074232 -0.754271
    O 2.143835 -0.930610 -0.004691
    H 1.297223 -1.445873 -0.034637
    H -0.361330 -2.824381 -0.799654
"""

mol = gto.Mole(atom=atom_token, basis="cc-pVDZ", max_memory=32000).build()
with open("h2o_5-pvdz.json", "w") as f:
    f.write(mol.dumps())

aux_jk = gto.Mole(atom=atom_token, basis="cc-pVDZ-jkfit").build()
with open("h2o_5-pvdz_jk.json", "w") as f:
    f.write(aux_jk.dumps())

aux = gto.Mole(atom=atom_token, basis="cc-pVDZ-ri").build()
with open("h2o_5-pvdz_ri.json", "w") as f:
    f.write(aux.dumps())

mf = scf.RHF(mol).density_fit(aux_jk.basis).run()

np.save("h2o_5-mo_coeff.npy", np.asarray(mf.mo_coeff, order="C"))
np.save("h2o_5-mo_energy.npy", mf.mo_energy)
