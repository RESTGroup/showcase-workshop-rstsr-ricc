# RI-RCCSD(T)：极简到高效率

实现内容：RI-RCCSD(T) (CCSD with triplet perturbation correction)

- 程序：`ri_rccsdt_naive.rs` ([gitee](https://gitee.com/restgroup/showcase-workshop-rstsr-ricc/blob/master/src/ri_rccsdt_naive.rs), [github](https://github.com/RESTGroup/showcase-workshop-rstsr-ricc/blob/master/src/ri_rccsdt_naive.rs))
- 最简实现，无视性能
- 该程序总共约 60 行 (其中调用的函数 `prepare_intermediates` 复用了程序 `ri_rccsdt_slow.rs` 的实现)

---

- 程序：`ri_rccsdt_slow.rs` ([gitee](https://gitee.com/restgroup/showcase-workshop-rstsr-ricc/blob/master/src/ri_rccsdt_slow.rs), [github](https://github.com/RESTGroup/showcase-workshop-rstsr-ricc/blob/master/src/ri_rccsdt_slow.rs))
- 与高性能的 RI-RCCSD(T) 具有相同的算法实现与程序结构，但没有引入一些关键的程序技巧
- 该程序总共约 150 行，其中多个程序复用的函数 `prepare_intermediates` 约 60 行

---

- 程序：`ri_rccsdt.rs` ([gitee](https://gitee.com/restgroup/showcase-workshop-rstsr-ricc/blob/master/src/ri_rccsdt.rs), [github](https://github.com/RESTGroup/showcase-workshop-rstsr-ricc/blob/master/src/ri_rccsdt.rs))
- 性能较高
    - 体系 (H2O)<sub>10</sub> cc-pVDZ ($n_\mathrm{basis} = 240$, $n_\mathrm{aux} = 840$，无冻结轨道)
    - 计算设备 Ryzen HX7945，16 cores，算力约 1.1 TFLOP/sec
    - CCSD(T) 的三次微扰部分耗时约 1010 sec，CPU 性能利用率不低于 34%，可能还有提升空间
    - 本文档的所有性能分析均基于 OpenBLAS 后端讨论
- 该程序总共约 150 行 (其中调用的函数 `prepare_intermediates` 复用了程序 `ri_rccsdt_slow.rs` 的实现)
- 本程序的算法参考了 PySCF 在 CCSD(T) 上的实现

## 目录

<!-- toc -->

## 1. RI-RCCSD(T) 实现前言

CCSD(T) 方法是通过微扰的方法，在仅有基于 CCSD 得到的 $\hat T \simeq \hat T_1 + \hat T_2$ 的一次、二次激发振幅基础上，给出三次激发的能量。微扰的三次激发能量的定义并不唯一，CCSD(T) 仅仅是多种三次微扰方法的其中一种；由于其具有较好的计算量与计算精度平衡，因此作为小分子计算化学方法广为接受与使用，常被称为“黄金标准”。至于这个“黄金”是否名不副实，或者是否是一种 overkill，这因具体的计算化学问题而异。

我们在此不讨论 CCSD(T) 的理论本身。

尽管我们将实现的是闭壳层 CCSD(T) 在 RI 下的近似，但若没有进一步近似，那么 RI 近似本身并不会真的节省 CCSD(T) 的计算量。因此，下述程序的实现，基本上与普通的 CCSD(T) 一致。内存上，我们要求非常大的 $n_\mathrm{occ} n_\mathrm{vir}^3$ 大小的张量；这会比 CCSD 单次迭代的内存量要大得多 (如果不考虑到 incore DIIS 对内存的额外消耗)。

## 2. RCCSD(T) 中间量

对于 CCSD(T) 的程序实现，我们需要一些中间张量。这部分的计算相对于 CCSD(T) 本身并不大，因此就在这里作讨论。

### 2.1 记号与公式

从公式上，CCSD(T) 涉及到 6-D 张量的计算，其指标是 $(a, b, c, i, j, k)$，即 3 个非占轨道与 3 个占据轨道。在迭代时，我们的主基调是对非占轨道 $(a, b, c)$ 优先迭代[^1]；因此会希望程序用到的张量中，非占轨道往尽量放在最不连续的维度上 (row-major 下是靠前的维度)，占据轨道尽量放在最连续的维度上靠 (col-major 下是靠后的维度)。

[^1]: 这里我们对非占轨道 $(a, b, c)$ 作外循环优先迭代，而不是对占据轨道 $(i, j, k)$ 优先迭代。

    在实现对占据轨道 $(i, j, k)$ 作外循环、非占轨道 $(a, b, c)$ 作内循环的程序后，对于当前关心的体系 (5-10 个水分子)，程序效率会有显著下降。我们认为这比较可能是因为循环中，生成的 $2 n_\mathrm{vir}^3$ 大小的张量会频繁读写、占用了较多的内存带宽，且其大小显著大于 L3 缓存、带宽效率也较低。同时，其计算过程中涉及到一步 $n_\mathrm{vir}^2 \times n_\mathrm{occ}$ 与 $n_\mathrm{occ} \times n_\mathrm{vir}$ 得到 $n_\mathrm{vir}^3$ 的矩阵乘法，即两个较小的矩阵相乘得到更大的矩阵；这类矩阵乘法不太容易达到理想浮点效率。

    相对地，对非占轨道 $(a, b, c)$ 作外循环时，中间张量是 $2 n_\mathrm{occ}^3$ 浮点数大小；对于 (H2O)<sub>10</sub> 体系，这也只有 2 MB。在单个线程内部，较小的张量的 elementwise 操作与矩阵乘法，其缓存命中率较高，容易达到更高的计算效率。

为此，我们在使用分子轨道下的双电子积分、以及激发张量时，其维度顺序都与 CCSD 时稍有不同。仅针对 CCSD(T) 的实现，定义

$$
\begin{alignat*}{10}
\texttt{t1\_t} &\quad& t_{ai} &:= t_{i}^{a} \\
\texttt{t2\_t} &\quad& t_{abji} &:= t_{ij}^{ab} \\
\texttt{d\_ooo} &\quad& \Delta_{ijk} &:= \varepsilon_i + \varepsilon_j + \varepsilon_k \\
\texttt{eri\_vvov\_t} &\quad& g_{acib} &:= g_{ai}^{cb} \simeq \sum_{P} B_{ia, P} B_{bc, P} \\
\texttt{eri\_vooo\_t} &\quad& g_{aljk} &:= g_{ak}^{lj} \simeq \sum_{P} B_{ka, P} B_{lj, P} \\
\texttt{eri\_vvoo\_t} &\quad& g_{abij} &:= g_{ai}^{bj} \simeq \sum_{P} B_{ia, P} B_{bj, P} \\
\end{alignat*}
$$

需要留意，上式中的 $g_{aljk}$ 的转置顺序与 $g_{acib}$ 和 $g_{abij}$ 不同。其中最大的张量是 $g_{acib}$ 的 $n_\mathrm{occ} n_\mathrm{vir}^3$ 浮点数的 4-D 张量。

### 2.2 程序

这里的程序都是比较标准的做法。

```rust
// ri_rccsdt_slow.rs, in fn `prepare_intermediates`

let t1_t = t1.t().into_contig(RowMajor);
let t2_t = t2.transpose((2, 3, 1, 0)).into_contig(RowMajor);
let d_ooo = eo.i((.., None, None)) + eo.i((None, .., None)) + eo.i((None, None, ..));

let eri_vvov_t = unsafe { rt::empty(([nvir, nvir, nocc, nvir], &device)) };
(0..nvir).into_par_iter().for_each(|a| {
    (0..nvir).into_par_iter().for_each(|c| {
        let mut eri_vvov_t = unsafe { eri_vvov_t.force_mut() };
        eri_vvov_t.i_mut([a, c]).matmul_from(&b_ov.i((.., a)), &b_vv.i((.., c)).t(), 1.0, 0.0);
    });
});

let eri_vooo_t = unsafe { rt::empty(([nvir, nocc, nocc, nocc], &device)) };
(0..nvir).into_par_iter().for_each(|a| {
    (0..nocc).into_par_iter().for_each(|l| {
        let mut eri_vooo_t = unsafe { eri_vooo_t.force_mut() };
        eri_vooo_t.i_mut([a, l]).matmul_from(&b_oo.i(l), &b_ov.i((.., a)).t(), 1.0, 0.0);
    });
});

let eri_vvoo_t = unsafe { rt::empty(([nvir, nvir, nocc, nocc], &device)) };
(0..nvir).into_par_iter().for_each(|a| {
    (0..nvir).into_par_iter().for_each(|b| {
        let mut eri_vvoo_t = unsafe { eri_vvoo_t.force_mut() };
        eri_vvoo_t.i_mut([a, b]).matmul_from(&b_ov.i((.., a)), &b_ov.i((.., b)).t(), 1.0, 0.0);
    });
});
```

### 2.3 性能评价

| 计算变量 | FLOPs 解析式 | FLOPs 实际数值 |
|--|--|--:|
| $g_{acib}$ | $2 n_\mathrm{occ} n_\mathrm{vir}^3 n_\mathrm{aux}$ | 0.52 TFLOPs |
| $g_{aljk}$ | $2 n_\mathrm{occ}^3 n_\mathrm{vir} n_\mathrm{aux}$ | 0.04 TFLOPs |
| $g_{abij}$ | $2 n_\mathrm{occ}^2 n_\mathrm{vir}^2 n_\mathrm{aux}$ | 0.14 TFLOPs |
| 总计 | | 0.70 TFLOPs |

- 计算耗时约 1.4 sec，实际运行效率为 0.50 TFLOP/sec；
- CPU 性能利用率为 45%；
- 在 CCSD(T) 计算中，这部分 $O(N^5)$ 计算量相对而言是可以忽略不计的。

## 3. RI-RCCSD(T) 最简实现

RI-RCCSD(T) 的公式尽管不算最少 (相比于 MP2 方法)，但也相比于 RI-RCCSD 要少许多了。其表达式的复杂性与 Hartree-Fock 相当。因此，我们可以用较少的代码实现。

我们可以参考 [Shen JCTC 2019](https://dx.doi.org/10.1021/acs.jctc.8b01294) 公式 (10) 附近的表达式，不过在约定俗成上有所区别：

$$
\begin{alignat*}{10}
\texttt{wp} &\quad& W'{}_{ijk}^{abc} &= \sum_{d} g_{db}^{ia} t_{kj}^{cd} - \sum_{l} g_{kc}^{lj} t_{il}^{ab} \\
\texttt{w}  &\quad& W_{ijk}^{abc} &= \hat{P}_{ijk}^{abc} W'{}_{ijk}^{abc} = W'{}_{ijk}^{abc} + W'{}_{ikj}^{acb} + W'{}_{jik}^{bac} + W'{}_{jki}^{bca} + W'{}_{kij}^{cab} + W'{}_{kji}^{cba} \\
\texttt{v}  &\quad& V_{ijk}^{abc} &= W_{ijk}^{abc} + g_{jb}^{kc} t_i^a + g_{ia}^{kc} t_j^b + g_{ia}^{jb} t_k^c \\
\texttt{d}  &\quad& \Delta_{ijk}^{abc} &= \varepsilon_i + \varepsilon_j + \varepsilon_k - \varepsilon_a - \varepsilon_b - \varepsilon_c \\
\texttt{z} &\quad& Z_{ijk}^{abc} &= 4 W_{ijk}^{abc} + W_{jki}^{abc} + W_{kij}^{abc} - 2 W_{kji}^{abc} - 2 W_{ikj}^{abc} - 2 W_{jik}^{abc} \\
\texttt{e\_corr\_pt} &\quad& \Delta E^\textsf{(T)} &= \frac{1}{3} \sum_{ijkabc} \frac{Z_{ijk}^{abc} V_{ijk}^{abc}}{\Delta_{ijk}^{abc}}
\end{alignat*}
$$

上述 6-D 张量，以 $W_{ijk}^{abc}$ 为例，在程序中其张量对应的指标顺序是 $(a, b, c, i, j, k)$。

---

上式中，计算 $W'{}_{ijk}^{abc}$ 与 $V_{ijk}^{abc}$ 使用到转置后的分子轨道下电子积分与 CCSD 激发张量；因此为了与程序作对照，我们需要将公式改写为

$$
\begin{aligned}
W'{}_{ijk}^{abc} &= \sum_{d} g_{abid} t_{cdjk} - \sum_{l} g_{cljk} t_{abli} \\
V_{ijk}^{abc} &= W_{ijk}^{abc} + g_{bcjk} t_{ai} + g_{acik} t_{bj} + g_{abij} t_{ck}
\end{aligned}
$$

$W'{}_{ijk}^{abc}$ 的计算过程中，在指标 $(a, b, c)$ 下作外循环，其索引得到的子张量都是连续的；并且可以通过维度更改 (reshape)，不需要复制到新的张量，就可以通过矩阵乘法给出结果。这解释了我们为何要如此设计 CCSD 激发张量与分子轨道下双电子积分的指标顺序的原因。

---

RCCSD(T) 最简程序实现代码如下，核心代码总共 25 行：

```rust
// ri_rccsdt_naive.rs

let wp: Tsr = unsafe { rt::empty(([nvir, nvir, nvir, nocc, nocc, nocc], &device)) };
(0..nvir).into_par_iter().for_each(|a| {
    (0..nvir).into_par_iter().for_each(|b| {
        (0..nvir).into_par_iter().for_each(|c| unsafe {
            let mut wp = wp.force_mut();
            let wp_1 = eri_vvov_t.i((a, b)) % t2_t.i(c).reshape((nvir, -1));
            let wp_2 = t2_t.i((a, b)).t() % eri_vooo_t.i(c).reshape((nocc, -1));
            wp.i_mut([a, b, c]).assign(wp_1.reshape((nocc, nocc, nocc)) - wp_2.reshape((nocc, nocc, nocc)));
        });
    });
});
let w = wp.transpose((0, 1, 2, 3, 4, 5))
    + wp.transpose((0, 2, 1, 3, 5, 4))
    + wp.transpose((1, 0, 2, 4, 3, 5))
    + wp.transpose((1, 2, 0, 4, 5, 3))
    + wp.transpose((2, 0, 1, 5, 3, 4))
    + wp.transpose((2, 1, 0, 5, 4, 3));
let v = &w
    + t1_t.i((.., None, None, .., None, None)) * eri_vvoo_t.i((None, .., .., None, .., ..))
    + t1_t.i((None, .., None, None, .., None)) * eri_vvoo_t.i((.., None, .., .., None, ..))
    + t1_t.i((None, None, .., None, None, ..)) * eri_vvoo_t.i((.., .., None, .., .., None));
let (so, sv) = (slice!(0, nocc), slice!(nocc, nmo));
let d = -mo_energy.i((sv, None, None, None, None, None))
    - mo_energy.i((None, sv, None, None, None, None))
    - mo_energy.i((None, None, sv, None, None, None))
    + mo_energy.i((None, None, None, so, None, None))
    + mo_energy.i((None, None, None, None, so, None))
    + mo_energy.i((None, None, None, None, None, so));
let z: Tsr = 4 * &w + w.transpose((0, 1, 2, 4, 5, 3)) + w.transpose((0, 1, 2, 5, 3, 4))
    - 2 * w.transpose((0, 1, 2, 5, 4, 3))
    - 2 * w.transpose((0, 1, 2, 3, 5, 4))
    - 2 * w.transpose((0, 1, 2, 4, 3, 5));
let e_corr_pt = (z * v / &d).sum() / 3.0;
```

---

但显然，这样的代码距离能用还有不少问题：

- **内存用量庞大**。这是最主要的问题。该程序需要引入多个 $n_\mathrm{occ}^3 n_\mathrm{vir}^3$ 大小的张量；对于 (H2O)<sub>4</sub> cc-pVDZ 基组，一个这样的张量就要 26 GB；对于 (H2O)<sub>5</sub> 甚至要 100 GB。这显然是不能接受的。

- **Elementwise 操作的优化不足**。上述程序还用到了很多的 broadcast 与 transpose；这些 elementwise 运算会占用 $O(n_\mathrm{occ}^3 n_\mathrm{vir}^3) \sim O(N^6)$ 的内存带宽。单个内存中 elementwise 操作的时间代价通常比矩阵乘法中单步的乘加融合 (FMA) 代价要更大。考虑到 CCSD(T) 的主要计算量是 $2 n_\mathrm{occ}^3 n_\mathrm{vir}^4$ 即较小的 $O(N^7)$ 的矩阵乘法 FLOPs，内存带宽占用只少一个数量级，因此 CCSD(T) 的 elementwise 操作最好有较高效率的实现。

    - 这与 CCSD 的情形不同。CCSD 是较大的 $O(N^6)$ 的矩阵乘法 FLOPs，与 $O(N^4)$ 的 elementwise 操作。因此，CCSD 的实现一般只需要注重矩阵乘法本身的实现即可。

## 4. RI-RCCSD(T) 初步实现

### 4.1 固定外循环指标的 $W'{}_{ijk}^{[abc]}$ 实现

作为初步实现，我们首先要解决内存用量庞大的问题。

作为解决方法，我们对指标 $(a, b, c)$ 作并行的外循环，使得循环内部最大的张量只有 3-D $n_\mathrm{occ}^3$ 的大小，如 $W'{}_{ijk}^{[abc]}$、$W_{ijk}^{[abc]}$、$V_{ijk}^{[abc]}$ 等等。

为此，我们将计算 $W'{}_{ijk}^{[abc]}$ 的部分单独拉出来成为一个函数：

$$
W'{}_{ijk}^{[abc]} = \sum_{d} g_{[ab]i,d} t_{[c]d,jk} - \sum_{l} t_{[ab]l,i} g_{[c]l,jk}
$$

```rust
// ri_rccsdt_slow.rs

fn get_w(abc: [usize; 3], intermediates: &RCCSDTIntermediates) -> Tsr {
    let t2_t = intermediates.t2_t.as_ref().unwrap();
    let eri_vvov_t = intermediates.eri_vvov_t.as_ref().unwrap();
    let eri_vooo_t = intermediates.eri_vooo_t.as_ref().unwrap();
    let nvir = t2_t.shape()[0];
    let nocc = t2_t.shape()[2];

    let [a, b, c] = abc;

    let mut w = eri_vvov_t.i([a, b]) % t2_t.i(c).reshape([nvir, nocc * nocc]);
    w.matmul_from(&t2_t.i([a, b]).t(), &eri_vooo_t.i(c).reshape([nocc, nocc * nocc]), -1.0, 1.0);
    w.into_shape([nocc, nocc, nocc])
}
```

### 4.2 处理 $W_{ijk}^{[abc]}$ 的置换求和

我们注意到 $W_{ijk}^{[abc]}$ 是经过置换求和后得到的结果：

$$
W_{ijk}^{[abc]} = \hat{P}_{ijk}^{[abc]} W'{}_{ijk}^{[abc]} = W'{}_{ijk}^{[abc]} + W'{}_{ikj}^{[acb]} + W'{}_{jik}^{[bac]} + W'{}_{jki}^{[bca]} + W'{}_{kij}^{[cab]} + W'{}_{kji}^{[cba]}
$$

这意味着，指标 $[abc]$ 下的结果 $W_{ijk}^{[abc]}$ 依赖于其他指标 $[acb], [bca], \cdots$。

为了破除这种依赖关系，使得外循环指标 $[abc]$ 内的任务相互独立，我们的做法是
- 只对 $a \geqslant b \geqslant c$ 上超三角部分作迭代 (迭代次数是 $\frac{1}{6} n_\mathrm{vir} (n_\mathrm{vir} + 1) (n_\mathrm{vir} + 2)$ 或约 $n_\mathrm{vir}^3/6$ 次)。
- 在循环内部现场生成置换的 $W'{}_{ikj}^{[acb]}$、$W'{}_{jik}^{[bac]}$ 等张量，即对外循环中特定的 $[abc]$ 指标，我们会生成 6 次 $W'{}^{[\cdots]}_\cdots$ 的张量。

```rust
// ri_rccsdt_slow.rs

fn ccsd_t_energy_contribution(abc: [usize; 3], mol_info: &RCCSDInfo, intermediates: &RCCSDTIntermediates) -> f64 {
    // ... //
    let [a, b, c] = abc;
    let w = get_w([a, b, c], intermediates)
        + get_w([a, c, b], intermediates).transpose([0, 2, 1])
        + get_w([b, c, a], intermediates).transpose([2, 0, 1])
        + get_w([b, a, c], intermediates).transpose([1, 0, 2])
        + get_w([c, a, b], intermediates).transpose([1, 2, 0])
        + get_w([c, b, a], intermediates).transpose([2, 1, 0]);
    // ... //
}
```

### 4.3 其余的计算

其余部分的张量计算就没有非常特别的部分了。

但在能量求和时，需要注意到，我们使用了上超三角的 $a \geqslant b \geqslant c$ 的迭代，每个被迭代到的指标 $[abc]$ 的权重并不相同。

$$
\Delta E^\textsf{(T)} = \sum_{abc} \frac{1}{3} \sum_{ijk} \frac{Z_{ijk}^{abc} V_{ijk}^{abc}}{\Delta_{ijk}^{abc}} = \sum_{a \geqslant b \geqslant c} \frac{w_{abc}}{3} \sum_{ijk} \frac{Z_{ijk}^{abc} V_{ijk}^{abc}}{\Delta_{ijk}^{abc}}
$$

其中，权重 $w_{abc}$ 的数值取决于它在哪个对角线上：

$$
w_{abc} = \begin{cases}
1 & a = b = c \\
2 & a = b \neq c \; \text{ or } \; a \neq b = c \\
6 & a \neq b \neq c
\end{cases}
$$

```rust
// ri_rccsdt_slow.rs

fn ccsd_t_energy_contribution(abc: [usize; 3], mol_info: &RCCSDInfo, intermediates: &RCCSDTIntermediates) -> f64 {
    // ... //
    let v = &w
        + t1_t.i((a, .., None, None)) * eri_vvoo_t.i([b, c]).i((None, .., ..))
        + t1_t.i((b, None, .., None)) * eri_vvoo_t.i([a, c]).i((.., None, ..))
        + t1_t.i((c, None, None, ..)) * eri_vvoo_t.i([a, b]).i((.., .., None));
    let d = -(ev[[a]] + ev[[b]] + ev[[c]]) + d_ooo;
    let z = 4.0 * &w + w.transpose([1, 2, 0]) + w.transpose([2, 0, 1])
        - 2.0 * w.transpose([2, 1, 0])
        - 2.0 * w.transpose([0, 2, 1])
        - 2.0 * w.transpose([1, 0, 2]);
    let e_tsr: Tsr = (z * v) / d;

    let fac = if a == c {
        1.0 / 3.0
    } else if a == b || b == c {
        1.0
    } else {
        2.0
    };
    
    fac * e_tsr.sum()
}
```

### 4.4 性能评价

| 计算表达式 | FLOPs 解析式 | FLOPs 实际数值 |
|--|--|--:|
| $W'{}_{ijk}^{abc} \leftarrow \sum_{d} g_{abid} t_{cdjk}$ | $2 n_\mathrm{occ}^3 n_\mathrm{vir}^4$ | 296 TFLOPs |
| $W'{}_{ijk}^{abc} \leftarrow - \sum_{l} t_{abli} g_{cljk}$ | $2 n_\mathrm{occ}^4 n_\mathrm{vir}^3$ | 78 TFLOPs |
| 总计 | | 374 TFLOPs |

- 这里我们没有统计其他 $O(N^6)$ 的计算步骤；
- 计算耗时约 1603 sec，实际运行效率不低于 0.23 TFLOP/sec；
- CPU 性能利用率不低于 21%。

## 5. RI-RCCSD(T) 高性能实现
