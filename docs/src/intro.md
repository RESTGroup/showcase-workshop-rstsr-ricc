# 前言

计算化学程序开发，与其他科学计算领域类似，有**快速验证**与**高效实现**的需求。

基于 [REST](https://gitee.com/RESTGroup/rest) 平台，我们在 Rust 语言下开发了 [RSTSR](https://github.com/RESTGroup/rstsr) 数学库工具，提供便利的程序接口，应对计算化学常见高维度张量的高效运算。

作为例子，我们**快速高效**地实现了 RHF 到 RCCSD(T) 一系列方法，展示 Rust 语言下计算化学程序 (或类似的科学计算任务) 开发的可能性。

## 目录

<!-- toc -->

## 1. Rust 语言与计算化学

这里的核心问题是，**为何 Rust 是适用于计算化学程序开发的语言？**

### 1.1 程序开发的两个需求：快速验证与高效实现

计算化学程序用户比较关心程序功能是否全面、运行能力是否强大、程序是否易于使用。其中，运行能力是指是否可以高效地模拟较大的体系；这要求了程序性能必须足够高。

计算化学程序开发者还同时面临一些额外的问题：计算化学方法，特别是涉及到后自洽场、响应性质、激发态等问题的公式经常比较复杂。如何快速验证程序的正确性、并进一步作性能改善，是程序开发者面临的困难。

如何以客观的标准，评估程序语言或框架，是否能满足计算化学所需要的快速验证与高效实现？我们认为，
- **快速验证**：因程序开发者经验与习惯差异而有不同看法；我们以代码行数作为客观标准，认为**较短的代码比较契合快速开发的需求**。理想情况下，**一行公式能对应一行代码**。
- **高效实现**：在算法清晰、浮点数计算量 (FLOPs) 可以大致算出的前提下，我们可以**对比计算设备的理论浮点计算能力 (FLOP/sec) 与程序的实际浮点计算效率**，确认实现是否确实高效。

随着硬件与计算机语言的发展，早年时代表现好的程序，或许会在未来失去竞争力。但是，上面两个标准本身，不随计算机语言、硬件水平而不同；任何时代下的程序与框架应都可以此标准衡量。

### 1.2 Rust 语言：作为计算化学程序开发者的特色与不足

作为计算化学程序开发者，Rust 语言吸引我们的特色包括
- 与 C 同等级的性能；只要算法实现得当，几乎不存在因 Rust 语言本身的不足引起的性能问题；
- 现代函数式语言的表达能力、基于 trait 的泛型能力、强类型等适合于大型项目的语言特性；
- 内存安全性与无畏并行，能保证程序编写者调用失误的可能性大大下降；
- (除科学计算之外的) 程序生态丰富，在硬件层与底层应用上有诸多出色的项目。

Rust 语言的不足包括
- 作为编译语言，尽管支持增量编译，但其编译或链接时间仍然较长 (与 C++ 项目同等级，涉及到模板特化的耗时较多)；
- REPL 支持有限，难以像 Python, Matlab, Mathematica, Julia 脚本式地运行程序[^1]；
- 科学计算与 AI 生态与 GPU 支持还有待发展[^2]。

Rust 语言具有争议的地方包括
- Rust 完全否定面向对象；必须通过类似于 C#、Java 的接口 (trait) 实现多态 (polymorphism)；
- 语法上对重载 (overload) 的支持有限[^3]。

[^1]: Rust 实际上有 REPL 库 [evcxr](https://github.com/evcxr/evcxr)，可以使用但体验与 Python 还是有差距的。作为参考，C++ [xeus-cling](https://github.com/jupyter-xeus/xeus-cling) 是类似的情况。
[^2]: 一方面除了 C/C++ 外，其他所有计算机语言对 GPU 的支持事实上都不足 (Fortran 有支持但较有限)，大多都需要通过 FFI (foreign function interface) 实现。另一方面，只从 CUDA 生态而言，涉及 GPU 的代码应看作 CUDA 语言而非传统的 C/C++ (不是相同的语言)；只是大多数工具链 (toolchain) 允许 CUDA 与 C/C++ 混合编译，使用上与语法上非常接近。从工程量上来看，使用 C/C++ 不一定就比使用其他语言更少。
[^3]: 一种常见的误解是 Rust 不能重载。实际上 Rust 具有重载能力 (基于 trait)，RSTSR 也大量使用了 Rust 重载；它或许可以看作支持泛型的 C11 `_Generic`，与 Python 的 named-arguments 不同。但重载在 Rust 语言中的支持确实不足，导致现在很多 RSTSR 函数的参数必须打两个括号才能使用。

Rust 语言与其他语言的对比：
- C++ 的表达能力非常自由，但程序编写者更容易写出错误的代码；C++ 语言本身复杂，编写程序时需要考虑引用、右值引用、移动等语义。
- Python 语言尽管可以写出效率很高的程序，但需要与其他语言 (C, C++, Rust) 或特殊的 Python 方言 (如 Numba) 结合才能达到高效率。Python 语言的类型检查较弱，非常便于快速验证、但不便于大型程序开发。
- Julia 语言尽管是少有的同时实现 REPL 与编译功能的语言，但其在科学计算之外的程序生态薄弱、有影响力的项目较少。
- Mojo, Moonbit 等语言吸收了 Python 与 Rust 等语言的特性，但目前仍在发展，现阶段不适合作为主力语言。

在语言上容易产生误解的地方包括
- ~*作为编译语言，Rust 必须要用很多代码实现脚本语言同样的功能*~：这个项目会表明，在合适的数学库框架下，Rust 语言可以用不太长的代码实现计算化学关心的问题。Python 下能用 $n$ 行代码实现计算化学的算法，Rust 经常有办法在 $1.5 n$ 行以内实现；Python 下不能高效实现的算法，Rust 可以实现。
- ~*Rust 语言学习曲线陡峭*~：C++ 语言的学习难度未必低于 Rust，但这个问题因人而异。不考虑工程与架构，计算化学的算法实现所遇到的问题，大多不涉及 Rust 语法中困难的部分。尽管我们需要了解和使用 Rust 的生命周期，但我们的工作中不涉及循环链表和复杂的异步，因此并不需要对生命周期的所有细节都有所掌握。

## 2. 数学库 RSTSR

### 2.1 RSTSR 简介

该项目使用到数学库 [RSTSR](https://github.com/RESTGroup/rstsr)。

RSTSR 是 Rust 语言下，可用于处理张量存储与索引、以及对矩阵作线性代数运算的库。
- 功能与 API 对标 NumPy 与 SciPy，可以原生地在 Rust 语言下开发科学计算程序。对于习惯 NumPy 的用户，可以参考 [NumPy-RSTSR 对照表](https://restgroup.github.io/rstsr-book/zh-hans/docs/numpy-cheatsheet) 以快速熟悉 RSTSR 的使用。
- 支持多后端。目前 RSTSR 支持 OpenBLAS 与 [Faer](https://github.com/sarah-quinones/faer-rs/) (纯 Rust 的数值代数库) 后端。未来会引入 MKL 与 CUDA 等其他后端。
- 涉及到矩阵线性代数的部分，性能由后端决定；其余的部分尽可能充分利用张量连续性与并行，性能较高。
- 同时支持行优先与列优先。

### 2.2 所有权问题

RSTSR 在使用上与 NumPy 有很多相似之处，但有一个非常大的区别在于数据所有权。

Rust 有非常严格的所有权规则；除非您的数据类型是 RC (reference counting) 智能指针，否则必须要区分数据是占有的还是引用的、对于引用类型还需要指定其生命周期。

RSTSR 的所有权规则参考自 Rust 库 `ndarray`；其简要说明可以参考 [用户文档](https://restgroup.github.io/rstsr-book/zh-hans/docs/fundamentals/structure_and_ownership)。调用的函数的过程中，
- 如果是不改变张量数据的操作 (比如索引、转置等)，函数签名中没有 `into_` 前缀的将返回引用类型张量视窗 (如函数 `i`, `transpose`, `swapaxes`)；若有 `into_` 前缀的则返回 `self` 自身的类型 (如函数 `into_reverse_axes`)。
- 张量维度更改 (reshape) 是可能产生新数据的操作。`reshape` 将返回占有或视窗的张量 (copy-on-write 类型)；`into_shape` 将返回占有数据的张量。调用 `reshape` 或 `into_shape` 时，如果原先的张量数据足够连续，可以避免数据的复制，从而以近乎为零的开销实现维度更改。

## 3. 约定俗成

### 3.1 记号

- 本项目统一使用行优先 (row-major)。
- 本项目仅考虑闭壳层基态，不引入赝势 (ECP)。占据轨道数严格等于实际体系电子数的一半。
- 本项目仅考虑实函数。

指标规则为

| 意义 | 英文名称 | 公式指标 | 数量 | 数量程序变量 | 程序变量代号 |
|--|--|--|--|--|--|
| 占据轨道 | occupied | $n_\mathrm{occ}$ | $i, j, k, l$ | `nocc` | `o` |
| 非占轨道 | virtual | $n_\mathrm{vir}$ | $a, b, c, d$ | `nvir` | `v` |
| 所有分子轨道 | orbital | $n_\mathrm{orb}$ | $p, q, r$ | `nmo` |
| (原子轨道) 基组 | basis | $n_\mathrm{basis}$ | $\mu, \nu, \kappa, \lambda$ | `nao` | `b` |
| (原子轨道) 辅助基组 | auxiliary | $n_\mathrm{aux}$ | $P, Q, R$ | `naux` | `x` |
| 原子核 | atom | $A, B$ | $n_\mathrm{atom}$ | `natm` |

### 3.2 张量与后端类型

本工作中，为了方便起见，后端类型声明为 `DeviceTsr`。若用户可以通过 cargo feature 指定 Faer 后端或 OpenBLAS 后端。

```rust
// prelude.rs
#[cfg(not(feature = "use_openblas"))]
pub type DeviceTsr = DeviceFaer;
#[cfg(feature = "use_openblas")]
pub type DeviceTsr = DeviceOpenBLAS;
```

我们总是在 64 位浮点数、固定后端下进行开发；为了简化代码，张量类型则分别定义为
- 占有张量 `Tsr`
- 不可变张量视窗 `TsrView`
- 可变张量视窗 `TsrMut`

我们会在非常少数的情景下，使用固定维度张量 (相对于动态维度而言)，因此保留维度泛型参数。

```rust
// prelude.rs
pub type Tsr<D = IxD> = Tensor<f64, DeviceTsr, D>;
pub type TsrView<'a, D = IxD> = TensorView<'a, f64, DeviceTsr, D>;
pub type TsrMut<'a, D = IxD> = TensorMut<'a, f64, DeviceTsr, D>;
```

## 4. 必要的性能分析背景

### 4.1 简单矩阵代数的 FLOPs 分析

在本项目中，我们会对 FLOPs 作简单分析。

- 对于矩阵乘法 ($\mathbf{A} \in \mathbb{F}^{m \times k}$，$\mathbf{B} \in \mathbb{F}^{k \times n}$，$\mathbf{C} \in \mathbb{F}^{m \times n}$，其中 $\mathbb{F}$ 代表数据的类型)

    $$
    C_{ij} = \sum_a A_{ia} B_{aj} \quad \text{or} \quad \mathbf{C} = \mathbf{A} \mathbf{B}
    $$

    则该计算的 FLOPs 总量是 $2 m n k$；其中的 $2$ 系数来源与一次加法与一次乘法。

- 对于矩阵乘法 

    $$
    C_{ij} = \sum_a A_{ia} B_{ja} \quad \text{or} \quad \mathbf{C} = \mathbf{A} \mathbf{A}^\dagger
    $$

    则该计算的 FLOPs 近似为 $n^2 k$ (由于 $C_{ij} = C_{ji}$，因此只需要作一半矩阵乘法，剩下一半通过转置得到)。

- 对于上三角或下三角矩阵 $\mathbf{A} \in \mathbb{F}^{n \times n}$ 以及长方形矩阵 $\mathbf{B} \in \mathbb{F}^{n \times m}$，作下述三角矩阵乘法或求解

    $$
    \begin{aligned}
    \mathbf{C} &= \mathbf{A} \mathbf{B} \quad \text{(triangular matrix multiply)} \\
    \mathbf{C} &= \mathbf{A}^{-1} \mathbf{B} \quad \text{(triangular matrix solve)} \\
    \end{aligned}
    $$

    则该计算的 FLOPs 近似为 $n^2 m$。

在以矩阵乘法为性能瓶颈的问题中，其他 FLOPs 通常是可以作为小量忽略的。但也要留意，除了矩阵乘法外，
- 在理想的程序实现、以及特定的算法下，内存带宽也会占用计算资源；
- 不理想的程序实现，会导致各种性能损耗，包括但不限于多余的内存资源分配与复制、低效率的指针索引、并行问题等等。

### 4.2 计算设备的理想性能

本文档着重于 CPU。CPU 厂商通常不会直接将 64 位浮点数矩阵乘法峰值能力标在产品说明上，我们通常需要自己作一些简单的换算。GPU 也可以作类似的分析，不过其峰值性能通常标注于产品说明，性能的比较会更加直观。

对于矩阵乘法问题，CPU 计算设备的理想性能是

$$
\text{Ideal GFLOP/sec} = n_\textsf{core} n_\textsf{FMA} n_\textsf{SIMD} n_\textsf{unit} f_\textsf{CPU}
$$

- $n_\textsf{core}$ 物理内核数 (关闭超线程下的最大有效线程数)；
- $n_\textsf{FMA}$ 乘加融合系数，固定为 2；
- $n_\textsf{SIMD}$ 单个指令允许最大的数据大小 (对于 64 位浮点，x86 AVX-512 为 8、AVX2 为 4，ARM Neon 为 2)；
- $n_\textsf{unit}$ 为 FMA 指令通道数 (AMD Zen AVX-512 为 1、AVX2 为 2，Intel Xeon 与 ARM Neon 通常为 2)；
- $f_\textsf{CPU}$ 为满负荷运行时，以 GHz 为单位的 CPU 频率。

本项目主要在个人电脑 AMD 7945HX 上测试。该设备的理想性能是 1.1 TFLOP/sec 或 1100 GFLOP/sec。

## 5. 前言结语

这份文档希望讨论程序的具体细节，以追求比较理想的程序性能。Rust 语言不仅能帮助我们较容易地达到理想性能，而且其具有较高的开发效率的潜力，适合计算化学的程序开发。

但也要留意，程序性能的提升经常只是改进；它当然也很重要，但真正的突破还需要依靠计算化学的算法、方法、或认知论的发展。
