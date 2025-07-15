# RI-RHF：高效率实现

- 程序：`ri_rhf.rs` ([gitee](https://gitee.com/restgroup/showcase-workshop-rstsr-ricc/blob/master/src/ri_rhf.rs), [github](https://github.com/RESTGroup/showcase-workshop-rstsr-ricc/blob/master/src/ri_rhf.rs))
- 实现内容：RI-RHF (restricted resolution-of-identity Hartree-Fock)
- 性能尚可
    - 体系 (H2O)<sub>10</sub> cc-pVDZ ($n_\mathrm{basis} = 240$, $n_\mathrm{aux} = 1160$)
    - 计算设备 Ryzen HX7945，16 cores，算力约 1.1 TFLOP/sec
    - OpenBLAS 1.4 sec / 16 iter (RSTSR v0.3 的矩阵-向量乘法用 GEMM 而非 GEMV 实现，J 积分较慢)
    - Faer 1.3 sec / 16 iter
- 该程序总共约 110 行
- 该程序包含 DIIS 迭代

## 目录

<!-- toc -->

## 1. RI-RHF 性能需要考虑的因素

提升 RI-RHF 主要的因素是 3c-2e 双电子积分 $g_{\mu \nu, P}$ 与 Cholesky 分解张量 $B_{P, \mu \nu}$ 具有对称性：
$$
g_{\mu \nu, P} = g_{\nu \mu, P}
$$
利用对称性可以节省一半内存，以及减小 J 积分的浮点计算量、以及 K 积分计算时的 DRAM 带宽。

> 利用对称性当然也可以减少电子积分本身的计算时间；但对于本项目的 RI-RHF 而言，我们已经假定电子积分可以全部存入内存，那么自洽过程就不需要额外计算电子积分了。同时，3c-2e 电子积分计算时间一般不大于一次 SCF 迭代，因此电子积分耗时本来就不是大头。是否节省这点计算量，对于当前项目的算法而言，意义并不大。
>
> Conventional 4c-2e 双电子积分的 RHF 方法体系稍大一些时，内存是无法存入所有电子积分的；因此每次 SCF 迭代一定会重新算电子积分，从而电子积分的耗时变得关键。这与 RI-RHF 的情形并不相同。

## 2. 3c-2e 双电子积分的获得

我们对 3c-2e 双电子积分有两个考量因素：
- 我们希望 3c-2e 双电子积分只存一半；对于 row-major 情形，我们希望存下三角部分 $\mu \geqslant \nu$。
- 考虑到后面 `solve_triangular` 函数的调用，我们希望生成 row-major 下连续的 $g_{P, \mu \nu}$，即辅助基的指标是内存上最不连续的。

libcint 给出的 3c-2e 电子积分始终是 $(\mu \nu | P)$ 形式的，必须要进行一次转置才能得到 $(P | \mu \nu)$。既然我们希望得到的是 row-major 下的 $g_{P, \mu \nu}$，那么反过来想，我们可以从 col-major 的 $g_{\nu \mu, P} = g_{\mu \nu, P}$ 作一次转置，就可以得到 row-major 下的 $g_{P, \mu \nu}$ 了。

综合这些考量因素，我们给出如下的 3c-2e 双电子积分计算代码：

```rust
use libcint::prelude::*;

let int3c2e = {
    let (out, shape) = CInt::integrate_cross("int3c2e", [cint_data, cint_data, aux_cint_data], "s2ij", None).into();
    rt::asarray((out, shape.f(), &device)).into_reverse_axes()
};
```

- [`CInt::integrate_cross`](https://docs.rs/libcint/latest/libcint/cint/struct.CInt.html#method.integrate_cross) 是可以对多个分子基组信息作交叉电子积分计算的函数，其结果是 col-major 的。参数 `"s2ij"` 表示其对称性。
- 在函数 `rt::asarray` 调用时，使用到的 `shape.f()` 是指，在构成张量时，我们希望使用 col-major 布局 (fortran-contiguous)。随后作 `into_reverse_axes` 将给出 col-major 到 row-major 的张量转置，这一步不会产生额外的内存复制。
- 最后返回得到的 `int3c2e` $g_{P, \mu \nu}$，其布局是 row-major，维度是 $n_\mathrm{aux} \times \frac{1}{2} n_\mathrm{basis} (n_\mathrm{basis} + 1)$，关于指标 $(\mu, \nu)$ 下三角存储。

整个过程中，除了电子积分本身的耗时与内存占用之外，其余部分的资源消耗是近乎为零的。

## 3. Cholesky 分解张量的获得

$$
B_{P, \mu \nu} = \sum_{Q}^{n_\mathrm{aux}} (L^{-1})_{PQ} g_{Q, \mu \nu} \quad \text{or} \quad \mathbf{B} = \mathbf{L}^{-1} \mathbf{g}^\textsf{3c}
$$

```rust
let cderi = rt::linalg::solve_triangular((int2c2e_l.view(), int3c2e, Lower));
```

这里需要留意的是，我们传入的 `int3c2e` 是占有数据的 `Tsr` 张量类型，而不是视窗引用。这意味着变量 `int3c2e` 在调用 `solve_triangular` 函数后就无法使用了。这与上一节 `ri_rhf_slow.rs` 程序的情况不同；上一节的程序的第二个参数 `int3c2e.reshape([nao * nao, naux]).t()` 是张量视窗。

这里里用到了 RSTSR 的参数重载。当第二个参数 (被求解的长方形矩阵) 是 `Tsr` 或 `TsrMut`、而非 `TsrView` 时，求解得到的结果会被写入到原来的张量中。这种情况下，只要保证传入 `solve_triangular` 存在一个维度是连续的，不论它是行连续还是列连续，`solve_triangular` 求解时都不会额外分配内存[^1]。

[^1]: 是否要分配额外的内存，这实际上依赖于后端的实现。不过当前实现了 `solve_triangular` 的后端 Faer 与 OpenBLAS 一般都不会额外分配内存。

## 4. J 积分计算

考虑到对称性，J 积分可以写为

$$
\begin{aligned}
J_{\mu \nu}
&= \sum_{P}^{n_\mathrm{aux}} B_{P, \mu \nu} \sum_{\kappa \lambda}^{n_\mathrm{basis}} B_{P, \kappa \lambda} D_{\kappa \lambda} \\
&= \sum_{P}^{n_\mathrm{aux}} B_{P, \mu \nu} \sum_{\kappa \geqslant \lambda}^{n_\mathrm{basis}} (2 - \delta_{\kappa \lambda}) B_{P, \kappa \lambda} D_{\kappa \lambda}
\end{aligned}
$$

由此，我们可以考虑构建一个缩放后的密度矩阵 $\bar{D}_{\kappa \lambda}$，并在程序实现时只取其下三角部分 $\kappa \geqslant \lambda$：

$$
\bar{D}_{\kappa \lambda} = (2 - \delta_{\kappa \lambda}) D_{\kappa \lambda}
$$

```rust
let mut dm_scaled = 2.0_f64 * dm.to_owned();
let dm_diag = dm.diagonal(None);
dm_scaled.diagonal_mut(None).assign(dm_diag);
let dm_scaled_tp = dm_scaled.pack_tril();
```

随后进行普通的矩阵-向量乘法，得到下三角的 $J_{\mu \nu}$ $(\mu \geqslant \nu)$：

```rust
let j_tp = (&cderi % dm_scaled_tp) % &cderi;
```

最后将下三角的 J 积分展开到方形矩阵：

```rust
j_tp.unpack_tril(FlagSymm::Sy)
```

## 5. K 积分计算

在这里，K 积分的计算与上一节有些许差别。

K 积分计算的核心目的仍然不变，是求取半转换的 Cholesky 分解张量时，尽量将指标辅助基指标 $P$ 与占据轨道指标 $i$ 放到一起。但是在放置顺序上，上一篇文档的程序是 $B_{i, P \mu}$；这里程序的策略则是 $B_{P, i \mu}$。

当前程序的伪代码可以写为

- 对指标 $P$ 作并行；
    - 对特定的指标 $P$，取得子下三角矩阵 $b_{\mu \nu} = B_{P, \mu \nu}$ $(\mu \geqslant \nu)$；
    - 对子下三角矩阵 $b_{\mu \nu} = b_{\nu \mu}$ 展开到方形矩阵；
    - 求取 $b_{i \mu} = \sum_{\nu} C_{\nu i} b_{\nu \mu}$
    - 对特定的指标 $P$，写入 $b_{i \mu}$ 到半转换张量 $B_{P, i \mu}$
- 将半转换张量的维度重置为 $B_{P i, \mu}$
- 通过矩阵乘法得到 $K_{\mu \nu} = \sum_{P i} B_{P i, \mu} B_{P i, \nu}$

```rust
let occ_coeff = mo_coeff.i((.., ..nocc));
let scr_xob = unsafe { rt::empty(([naux, nocc, nao], &device)) };
(0..naux).into_par_iter().for_each(|p| {
    let mut scr_xob = unsafe { scr_xob.force_mut() };
    let cderi_ob = cderi.i(p).unpack_tril(FlagSymm::Sy);
    scr_xob.i_mut(p).matmul_from(&occ_coeff.t(), &cderi_ob, 1.0, 0.0);
});
let scr_xob_flat = scr_xob.reshape([naux * nocc, nao]);
2.0_f64 * (scr_xob_flat.t() % &scr_xob_flat)
```

程序上，比较特殊的地方有

- 对指标 $P$ 并行，在 C++/OpenMP 中可以写为
    ```C++
    #pragma omp parallel for // if don't want parallel, comment this line
    for (size_t p = 0; p < naux; ++p) { ... }
    ```
    相对而言，在 Rust 下，使用 Rayon 库，语法上有些许不同，但思路是一样的：
    ```rust
    // if don't want parallel, change `into_par_iter` to `into_iter`
    (0..naux).into_par_iter().for_each(|p| { ... });
    ```
- 第一个 unsafe 是 `rt::empty` 的要求。在 Rust 中，声明一块没有初始化的内存是不安全的行为。对于 `f64` 类型而言，这个不安全性至多影响到数据安全问题，这在科学计算上不是很重要。但如果是对一个具有析构函数 (trait `Drop` implemented) 的类型不作初始化，那么很容易会出现非有效内存错误。

- 第二个 unsafe，从程序的角度来说，是强制在每个线程中，对声明为不可变的 `scr_xob` ($B_{P i, \mu}$) 取到其可变的引用。

    这从一般的程序开发角度来看，是比较危险的 unsafe。但在科学计算问题里，这是非常常见的情况。这个 unsafe 其实在 C++/OpenMP 中是声明某个变量为 shared：
    ```C++
    #pragma omp parallel for shared(scr_xob)
    ```
    如果您作为 C++ 编程者，认为这里的 omp shared 语句是可以接受的，那么这里的 unsafe 您也应该容易接受。

- 第二个 unsafe 确实是有更安全的解决方案，但它可能会导致代码变得不那么直观：
    ```rust
    let mut scr_xob_iter = scr_xob.axes_iter_mut(0);
    (0..naux).into_par_iter().zip(scr_xob_iter).for_each(|(p, mut scr_ob)| {
        let cderi_ob = cderi.i(p).unpack_tril(FlagSymm::Sy);
        scr_ob.matmul_from(&occ_coeff.t(), &cderi_ob, 1.0, 0.0);
    });
    ```
    其中，`axes_iter_mut(0)` 函数是指对第 0 个维度 (指标 $P$ 所在的维度) 返回 mutable 子张量的迭代器。这个迭代器可以用于并行的环境。
- `i_mut(p)` 与 `i` 函数一样是基础索引，但它将返回张量的可变视窗。
- `matmul_from` 与 `%` 或者 `rt::matmul` 一样，都是张量的矩阵乘法。但不同的是，`%` 与 `rt::matmul` 会为矩阵乘法的结果新开辟一片堆空间内存；但 `matmul_from` 则是在已有内存上进行矩阵乘法，且允许指定 $\mathbf{C} = \alpha \mathbf{A} \mathbf{B} + \beta \mathbf{C}$ 的系数 $\alpha$ 与 $\beta$。使用 `matmul_from` 或许从阅读上看来并不直观；但从效率与内存占用上，`matmul_from` 确实是更好的选择。

- 我们在 Rayon 并行区域内调用了矩阵乘法。在 RSTSR 中，对于 BLAS 后端，
    - 如果在 Rayon 并行区域外调用 BLAS，则会使用并行版本的 BLAS，并控实际使用的制线程数量。
    - 如果在 Rayon 并行区域内调用 BLAS，则会通过 `openblas_set_num_threads` 或 `omp_set_num_threads` 控制线程数为 1，从而线程内使用串行的 BLAS、但线程外由 Rayon 控制任务调度，达到并行的效果。
    - 对于 pthreads 并行的 OpenBLAS，如果您希望在其他地方直接使用 OpenBLAS 函数前，最好重新通过 `openblas_set_num_threads` 重新设置一下线程数量。Rayon 并行区域内调用 BLAS 可能会破坏 OpenBLAS 的全局并行参数。对于 OpenMP 并行的 OpenBLAS 应没有上述问题。

- 注意到最后实现矩阵乘法 $\mathbf{K} = 2 \mathbf{\bar{B}}^\dagger \mathbf{\bar{B}}$ 的代码
    ```rust
    2.0_f64 * (scr_xob_flat.t() % &scr_xob_flat)
    ```
    括号影响了计算顺序，即我们需要先进行 $\mathbf{\bar{B}}^\dagger \mathbf{\bar{B}}$，随后作 2 倍的乘法。如果不加括号，考虑到 `*` 与 `%` 是同优先级算符，计算顺序是会发生变化的。同时，对两个相同矩阵的转置乘法 (类似于 $\mathbf{A}^\dagger \mathbf{A}$ 或 $\mathbf{A} \mathbf{A}^\dagger$)，RSTSR 会作一步额外优化：
    - BLAS 后端会调用 SYRK 而非 GEMM 函数计算；
    - Faer 后端会使用仅输出三角矩阵的矩阵乘法模块。
    理想情况下，这类型矩阵乘法的浮点计算量相比于普通矩阵乘法，可以节省一半。
