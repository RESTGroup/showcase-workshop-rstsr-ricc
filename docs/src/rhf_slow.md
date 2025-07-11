# RHF：极简演示

- 程序：`rhf_slow.rs` ([gitee](https://gitee.com/restgroup/showcase-workshop-rstsr-ricc/blob/master/src/rhf_slow.rs), [github](https://github.com/RESTGroup/showcase-workshop-rstsr-ricc/blob/master/src/rhf_slow.rs))
- 实现内容：RHF (restricted Hartree-Fock, conventional 4c-2e)
- 不关注程序效率
- 该程序总共约 50 行

<!-- toc -->

## 1. RHF 回顾

RHF 方法是分子体系计算化学中最基础的方法。在 Bohn-Oppenheimer 近似下，假定原子核不运动，则分子体系的能量 $E^\mathsf{tot}$ 可以视为原子核互斥能 $E^\mathsf{nuc}$ 与电子能量 $E^\mathsf{elec}$ 之和：

$$
E^\mathsf{tot} = E^\mathsf{nuc} + E^\mathsf{elec}
$$

---

其中，原子核互斥能量决定于原子核的相对位置 $\bm{R}_A$ 与其电荷数 $Z_A$：

$$
E^\mathsf{nuc} = \sum_{A < B}^{n_\mathrm{atom}} \frac{Z_A Z_B}{|\bm{R}_A - \bm{R}_B|}
$$

---

Hartree-Fock 近似要求电子的波函数是 Slater 行列式。在闭壳层下 ($\alpha$ 自旋与 $\beta$ 自旋的分子轨道与能级，在空间上对应相等)，其最终表达式可以写为[^1]

$$
\begin{aligned}
E^\mathsf{elec} [\mathbf{D}] &= \sum_{\mu \nu} D_{\mu \nu} \left( h_{\mu \nu} + \frac{1}{2} J_{\mu \nu} - \frac{1}{4} K_{\mu \nu} \right) \\
&= \sum_{\mu \nu} \left( h_{\mu \nu} + \sum_{\kappa \lambda} D_{\kappa \lambda} \left( \frac{1}{2} g_{\mu \nu, \kappa \lambda} - \frac{1}{4} g_{\mu \kappa, \nu \lambda} \right) \right)
\end{aligned}
$$

其中，
- $h_{\mu \nu}$ 是 core Hamiltonian，它包含了动能 $t_{\mu \nu}$ 与电子-原子核势 $v_{\mu \nu}$。
- $g_{\mu \nu, \kappa \lambda}$ 是。
- $D_{\mu \nu}$ 是电子云密度矩阵，它是由占据轨道系数 $C_{\mu p}$ 构成；其表达式中的 $2$ 是指闭壳层下 $\alpha, \beta$ 轨道有两份贡献。

以上的量中，$h_{\mu, \nu}, t_{\mu \nu}, v_{\mu \nu}, g_{\mu \nu, \kappa \lambda}$ 是随体系确定的，$D_{\mu \nu}$ 是待求解的。记号 $E^\mathsf{elec} [\mathbf{D}]$ 代表我们将电子能量 $E^\mathrm{elec}$ 视作关于密度矩阵 $\mathbf{D}$ 的函数。

$$
\begin{aligned}
h_{\mu \nu} &= t_{\mu \nu} + v_{\mu \nu} \\
t_{\mu \nu} &= - \frac{1}{2} \int \mathrm{d} \bm{r} \, \phi_\mu (\bm{r}) \nabla^2 \phi_\nu (\bm{r}) \\
v_{\mu \nu} &= \sum_{A}^{n_\mathrm{atom}} \int \mathrm{d} \bm{r} \, \phi_\mu (\bm{r}) \frac{-Z_A}{|\bm{r} - \bm{R}_A|} \phi_\nu (\bm{r}) \\
g_{\mu \nu, \kappa \lambda} &= \int \mathrm{d} \bm{r}_1 \int \mathrm{d} \bm{r}_2 \, \phi_\mu (\bm{r}_1) \phi_\nu (\bm{r}_1) \frac{1}{|\bm{r}_1 - \bm{r}_2|} \phi_\kappa (\bm{r}_2) \phi_\lambda (\bm{r}_2) \\
D_{\mu \nu} &= 2 \sum_{i}^{n_\mathrm{occ}} C_{\mu i} C_{\nu i} \quad \text{or} \quad \mathbf{D} = 2 \mathbf{C} \mathbf{C}^\dagger
\end{aligned}
$$

[^1]: 这里的表达式可以参考 Szabo & Ostlund, eqs (3.154, 3.184)；但需要注意记号上的差异。

---

RHF 要求密度矩阵 $D_{\mu \nu}$ 对应的占据轨道系数 $C_{\mu i}$ 满足正交归一条件

$$
\sum_{\mu \nu} \mathrm{C}_{\mu i} S_{\mu \nu} C_{\nu j} = \delta_{ij} \quad \text{or} \quad \mathbf{C}^\dagger \mathbf{S} \mathbf{C} = \mathbf{I}
$$

以此正交归一条件为前提，电子能量 $E^\mathsf{elec} [\mathbf{D}]$ 取条件极小，即为体系的基态电子能量。上式中，重叠矩阵 $S_{\mu \nu}$ 是

$$
S_{\mu \nu} = \int \mathrm{d} \bm{r} \, \phi_\mu (\bm{r}) \phi_\nu (\bm{r})
$$

定义能量对密度矩阵的导数矩阵 (该矩阵也称为 Fock 矩阵)

$$
F_{\mu \nu} [\mathbf{D}] = \frac{\partial E^\mathsf{elec}}{\partial D_{\mu \nu}} = h_{\mu \nu} + J_{\mu \nu} - \frac{1}{2} K_{\mu \nu}
$$

引入 Lagrangian 乘子 $2 \bm{\varepsilon}$ (由 Koopmans 的诠释，$\bm{\varepsilon}$ 一定条件下具有轨道能的物理意义)，对能量作关于轨道系数 $\mathbf{C}$ 的条件极小，得到 Hartree-Fock-Roothaan 方程：

$$
\begin{aligned}
0 &= \frac{1}{2} \frac{\partial \left( E^\textsf{elec} - 2 \mathbf{C}^\dagger \mathbf{S} \mathbf{C} \bm{\varepsilon} \right)}{\partial \mathbf{C}^\dagger} = \frac{1}{2} \frac{\partial E^\mathsf{elec}}{\partial \mathbf{D}} \frac{\partial \mathbf{D}}{\partial \mathbf{C}^\dagger} - \mathbf{S} \mathbf{C} \bm{\varepsilon} \\
&= \mathbf{F} \mathbf{C} - \mathbf{S} \mathbf{C} \bm{\varepsilon}
\end{aligned}
$$

该问题恰好与第一类广义本征值问题等价[^2]。当作为广义本征值问题时，求解得到的系数矩阵是 $n_\mathrm{basis} \times n_\mathrm{basis}$ 即基组平方大小的；在计算密度矩阵 $D_{\mu \nu}$ 时，需要取本征值最低的 $n_\mathrm{occ}$ 个系数，重新构成 $C_{\mu i}$ 的 $n_\mathrm{basis} \times n_\mathrm{occ}$ 大小的矩阵。

[^2]: 这里第*第一类*是程序意义上的，参考 [Lapack `dsygv`](https://www.netlib.org/lapack/explore-html/d1/d39/group__hegv_gadc3e6dc69532c1233818df364b7de912.html#gadc3e6dc69532c1233818df364b7de912) (参数 `ITYPE`) 或 [SciPy `eigh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html) (参数 `type`) 程序文档。

## 2. 程序

### 2.1 输入与输出

整个程序中，涉及到的函数包括
```rust
pub fn get_energy_nuc(cint_data: &CInt) -> f64;
pub fn minimal_rhf(cint_data: &CInt) -> RHFResults
```

---

其中，输入参数 `cint_data: &CInt` 是定义了分子坐标与基组信息的类型 `libcint::CInt`。其初始化可以通过下述程序进行：
```rust
let cint_data = CInt::from_json(&mol_file);
```
这里的 `mol_file` 是定义了分子坐标与基组信息的 json 文件。该文件与 PySCF 输出的格式一致：
```python
# python generation of `mol_file`
from pyscf import gto

mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="def2-TZVP").build()
with open("h2o-tzvp.json", "w") as f:
    f.write(mol.dumps())
```

---

RHF 主程序是 `minimal_rhf`；其输出 RHFResults 定义为
```rust
pub struct RHFResults {
    pub mo_coeff: Tsr,
    pub mo_energy: Tsr,
    pub dm: Tsr,
    pub e_nuc: f64,
    pub e_elec: f64,
    pub e_tot: f64,
}
```

### 2.2 原子核互斥能计算

原子核互斥能的计算代价较小，也有多种实现方案。下面的实现方案并非是性能上最快的，但代码长度少，也比较直观。

首先，通过指标对称性，我们更改原子核互斥能的表达式为

$$
E^\mathsf{nuc} = \frac{1}{2} \sum_{A \neq B}^{n_\mathrm{atom}} \frac{Z_A Z_B}{|\bm{R}_A - \bm{R}_B|}
$$

进一步地，我们对下述矩阵作定义：

$$
\begin{aligned}
P_{AB} &= |\bm{R}_A - \bm{R}_B| \\
P'_{AB} &= \begin{cases} P_{AB} & A \neq B \\ \infty & A = B \end{cases} \\
\end{aligned}
$$

那么原子核互斥能的表达式会更加简洁：

$$
E^\mathsf{nuc} = \frac{1}{2} \sum_{A B}^{n_\mathrm{atom}} \frac{Z_A Z_B}{P'_{AB}}
$$

---

我们需要得到原子核电荷 $Z_A$。这是一个 $n_\mathrm{atom}$ 的向量。

我们可以用 `rt::asarray`，生成一个占有数据的张量：

```rust
let atom_charges = rt::asarray((cint_data.atom_charges(), &device)); // Tsr (natm, )
```

---

我们需要得到原子核坐标 $\bm{R}_A$。这是一个 $n_\mathrm{atom} \times 3$ 的矩阵。

[`CInt::atom_coords`](https://docs.rs/libcint/latest/libcint/cint/struct.CInt.html#method.atom_coords) 函数输出的是 `Vec<[f64; 3]>` 类型；它不能由 `rt::asarray` 直接读取为 f64 类型张量。
```rust
let coords = cint_data.atom_coords(); // Vec<[f64; 3]> (natm)
```
其中一种办法是，先将长度为 $n_\mathrm{atom}$ 的 `Vec<[f64; 3]>` 类型转换为 $3 n_\mathrm{atom}$ `Vec<f64>` 类型，
```rust
let coords = coords.into_iter().flatten().collect::<Vec<f64>>(); // Vec<f64> (3 * natm)
```
随后代入到 `rt::asarray` 中构成张量类型 `Tsr`，并通过 `into_shape` 重塑其形状为 $n_\mathrm{atom} \times 3$：
```rust
rt::asarray((coords, &device)).into_shape((-1, 3)) // Tsr (natm, 3)
```

另一种办法是，RSTSR 提供了函数 `into_unpack_array`；它可以将 `[f64; 3]` 类型的数据转为 `[f64]` 类型。其输入的参数是新增加出来的维度的位置 (在当前的例子中，我们希望维度增加在第 1 个位置)：
```rust
let atom_coords = rt::asarray((cint_data.atom_coords(), &device)).into_unpack_array(1);
```

---

对于 $P_{AB} = |\bm{R}_A - \bm{R}_B|$，它实际上就是两个原子之间的 Euclidean 距离。在 RSTSR 的 `rt::sci` 模块，我们有 `cdist` 函数以实现该功能 (类比 SciPy [`cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) 或 Matlab [`pdist2`](https://ww2.mathworks.cn/help/stats/pdist2.html))。该函数目前还是要求传入张量视窗；视窗可以直接从张量变量后加 `.view()` 得到。
```rust
let mut dist = rt::sci::distance::cdist((atom_coords.view(), atom_coords.view()));
```

对于 $P'_{AB} = \begin{cases} P_{AB} & A \neq B \\ \infty & A = B \end{cases}$，它相对于 $P_{AB}$ 是在对角线上的值设为无穷大。下述代码中，`diagonal_mut` 是获得矩阵 $P_{AB}$ 的对角线的可变引用，并随后调取 `fill` 以设置具体的数值。
```rust
dist.diagonal_mut(None).fill(f64::INFINITY);
```

---

最后，计算总原子核排斥能 $E^\mathsf{nuc} = \frac{1}{2} \sum_{A B}^{n_\mathrm{atom}} Z_A Z_B / P'_{AB}$ 的代码为
```rust
0.5 * ((atom_charges.i((.., None)) * &atom_charges) / dist).sum()
```
该代码用到了张量广播 (broadcast) 的特性。上述表达式在作求和之前，生成的是该张量：
$$
T_{AB} = Z_A Z_B / P'_{AB}
$$
但该数乘计算需要先进行张量的维度对齐才能进行；这是指在 $Z_{A}$ 后增加一个维度、并在 $Z_{B}$ 前增加一个维度：
$$
T_{AB} = Z_{A \square} Z_{\square B} / P'_{AB}
$$
不过对于 row-major，$Z_{\square B}$ 前面增加出来的维度在程序中是可以被识别出来的。因此，我们只需要对 $Z_A$ 作维度对齐到 $Z_{A \square}$ 即可。

在 RSTSR 中，与 NumPy 一样地，为对齐而扩充维度，可以通过基础索引中插入 `None` 实现。

### 2.3 自洽场计算

首先，我们需要获得电子积分。这些量与密度无关，可以事先存储。我们在程序 `util.rs` ([gitee](https://gitee.com/restgroup/showcase-workshop-rstsr-ricc/blob/master/src/util.rs), [github](https://github.com/RESTGroup/showcase-workshop-rstsr-ricc/blob/master/src/util.rs)) 中封装了电子积分调取函数 `intor_row_major`，可以在一个函数调用中直接给出 `Tsr` 张量类型的电子积分。

下述代码中，`hcore` 为 $h_{\mu \nu}$、`ovlp` 为 $S_{\mu \nu}$、`int2e` 为 $g_{\mu \nu, \kappa \lambda}$。
```rust
let hcore = util::intor_row_major(cint_data, "int1e_kin") + util::intor_row_major(cint_data, "int1e_nuc");
let ovlp = util::intor_row_major(cint_data, "int1e_ovlp");
let int2e = util::intor_row_major(cint_data, "int2e");
```

---

随后是自洽场循环本身。我们回顾到自洽场问题是为了求解占据轨道系数矩阵 $C_{\mu i}$ 与电子云密度矩阵 $D_{\mu \nu}$；但由于求解过程中 Fock 矩阵也是随密度有依赖关系的，因此不能一次性计算得到，而需要通过迭代求解得到。该迭代关系总结下来是：
1. 通过密度矩阵得到 Fock 矩阵 $F_{\mu \nu}$：
    $$
    F_{\mu \nu} [\mathbf{D}] = h_{\mu \nu} + J_{\mu \nu} - \frac{1}{2} K_{\mu \nu} = h_{\mu \nu} + \sum_{\kappa \lambda} D_{\kappa \lambda} \left( g_{\mu \nu, \kappa \lambda} - \frac{1}{2} g_{\mu \kappa, \nu \lambda} \right)
    $$
2. 对 Fock 矩阵求解本征问题得到轨道系数 $C_{\mu p}$，该轨道系数需要依能级自小到大地排列：
    $$
    \sum_{\nu} F_{\mu \nu} C_{\nu p} = \sum_{\nu} S_{\mu \nu} C_{\nu p} \varepsilon_p \quad \text{or} \quad \mathbf{F} \mathbf{C} = \mathbf{S} \mathbf{C} \bm{\varepsilon}
    $$
3. 取本征值最低的占据轨道数量 $n_\mathrm{occ}$ 个本征向量构成占据轨道系数 $C_{\mu i}$，进而获得密度矩阵 $D_{\mu \nu}$：
    $$
    D_{\mu \nu} = 2 \sum_{i}^{n_\mathrm{occ}} C_{\mu i} C_{\nu i} \quad \text{or} \quad \mathbf{D} = 2 \mathbf{C} \mathbf{C}^\dagger
    $$

---

自洽场既然是三行公式可以说明的问题，那么程序上也可以用三行写出来：
```rust
let mut dm = rt::zeros(([nao, nao], &device));
let mut mo_coeff = rt::zeros(([nao, nao], &device));
let mut mo_energy = rt::zeros(([nao], &device));
for _ in 0..NITER {
    let fock = &hcore + ((1.0_f64 * &int2e - 0.5_f64 * int2e.swapaxes(1, 2)) * &dm).sum_axes([-1, -2]);
    (mo_energy, mo_coeff) = rt::linalg::eigh((&fock, &ovlp)).into();
    dm = 2.0_f64 * mo_coeff.i((.., ..nocc)) % mo_coeff.i((.., ..nocc)).t();
}
```
尽管实际上该程序有 8 行，但其中 3 行是对可变变量 `dm`, `mo_coeff`, `mo_energy` 作定义，2 行是构建循环体。因此，真正用于自洽场算法实现的，确实只有 3 行。

---

```rust
let fock = &hcore + ((1.0_f64 * &int2e - 0.5_f64 * int2e.swapaxes(1, 2)) * &dm).sum_axes([-1, -2]);
```

Fock 矩阵计算中，
- `int2e.swapaxes(1, 2)` 是将 4-D 张量 $g_{\mu \nu, \kappa \lambda}$ 中的 $\nu, \kappa$ 维度作交换，得到 $g_{\mu \kappa, \nu \lambda}$：
    $$
    \texttt{T1}_{\mu \nu, \kappa \lambda} = g_{\mu \kappa, \nu \lambda}
    $$
    在 RSTSR 中，交换两个维度 (swapaxes) 或转置 (transpose) 都是没有计算与内存代价的，但可能对未来的计算效率产生影响；这与 NumPy 或 Julia 是一样的。
- `0.5_f64 * int2e.swapaxes(1, 2)` 中，我们需要明确定义 `0.5_f64` 是 f64 位浮点，而不能仅写为 `0.5`。这是由于 Rust 在处理二目运算类型推断时，左右表达式的地位不等价、以及 RSTSR 内部实现本身有关。该表达式与 `int2e.swapaxes(1, 2) * 0.5` 是等价的实现，但后者的写法可以成功地进行类型推断。
- `sum_axes([-1, -2])` 是对最后两个维度作求和。具体来说，下述代码碎片
    ```rust
    (1.0_f64 * &int2e - 0.5_f64 * int2e.swapaxes(1, 2)) * &dm
    ```
    执行的是 broadcast 数乘运算，得到了 4-D 张量：
    $$
    \texttt{T2}_{\mu \nu, \kappa \lambda} = \left( g_{\mu \nu, \kappa \lambda} - \frac{1}{2} \texttt{T1}_{\mu \nu, \kappa \lambda} \right) D_{\square \square \kappa \lambda} 
    $$
    随后对 $\texttt{T2}_{\mu \nu, \kappa \lambda}$ 的最后两个维度 $\lambda, \kappa$ 作求和：
    $$
    \texttt{T3}_{\mu \nu} = \sum_{\lambda \kappa} \texttt{T2}_{\mu \nu, \kappa \lambda}
    $$
- 最后的 Fock 矩阵则通过 $F_{\mu \nu} = h_{\mu \nu} + \texttt{T3}_{\mu \nu}$ 得到。

---

```rust
(mo_energy, mo_coeff) = rt::linalg::eigh((&fock, &ovlp)).into();
```

广义本征值求解可以通过 RSTSR 的函数 `rt::linalg::eigh` 实现。
- 该函数返回的值是 `EighResult` 类型，需要通过 `.into()` 转换为 `(Tsr, Tsr)` 的 tuple 类型。
- 这里由于是改写可变变量 `mo_energy`, `mo_coeff`，因此赋值语句左边不需要用 Rust 的 let。

---

```rust
dm = 2.0_f64 * mo_coeff.i((.., ..nocc)) % mo_coeff.i((.., ..nocc)).t();
```

密度矩阵计算中，
- `.i((.., ..nocc))` 是基础索引语句，它是指对第一个维度作全索引、第二个维度取前 $n_\mathrm{occ}$ 个指标，构成新的矩阵。具体来说，本征值求解得到的矩阵是 $n_\mathrm{basis} \times n_\mathrm{basis}$ 大小的 $C_{\mu p}$；但实际的密度矩阵需要前 $n_\mathrm{occ}$ 个本征值最低的本征向量构成的 $n_\mathrm{basis} \times n_\mathrm{occ}$ 大小的 $C_{\mu i}$。
- `.t()` 与函数 `.to_reverse_axes()` 等价；它是生成一个转置的张量视窗。
- `%` 是 RSTSR 特有的语法糖；它的意义是矩阵乘法 (或矩阵-向量乘法、向量-矩阵乘法)。RSTSR 中，不论输入的矩阵是 row-major 或 col-major，都可以正常地调用 BLAS 的 GEMM 函数，不会因为输入矩阵是否转置了而产生效率损失。输出矩阵的连续性由编译时设置的 cargo feature (或后端的默认设定) 决定。

### 2.4 RHF 能量计算

对于 RHF 方法，能量计算的代码与 Fock 矩阵的实现是非常相似的，只是在 4c-2e 双电子积分 $g_{\mu \nu, \kappa \lambda}$ 上的系数有差异。

```rust
let eng_scratch = &hcore + ((0.5_f64 * &int2e - 0.25_f64 * int2e.swapaxes(1, 2)) * &dm).sum_axes([-1, -2]);
let e_elec = (&dm * &eng_scratch).sum();
let e_tot = e_nuc + e_elec;
```
