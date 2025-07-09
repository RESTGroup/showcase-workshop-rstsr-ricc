# REST Workshop：Rust 计算化学程序开发演示

**Rust 计算化学程序开发：极短 RHF 到高效 RCCSD(T) 的简易实现**

该仓库以 RHF 与 RCCSD(T) 为例子，用于演示 Rust 语言下，计算化学程序的**快速开发**与**高性能实现**的一种策略与工程实践。

该仓库将使用 [RSTSR](https://github.com/RESTGroup/rstsr) ([REST](https://gitee.com/RESTGroup/REST) 子项目) 作为数学库工具，[libcint](https://github/sunqm/libcint) (Rust [binding and wrapper](https://github.com/ajz34/libcint-rs)) 作为电子积分引擎。我们将展示 RSTSR 在单一语言框架 (Rust) 下，可以同时兼顾**开发效率**与**运行效率**。

> 这里的**快速开发**，会因程序开发者经验与习惯差异而有不同看法；我们以代码行数作为客观标准，认为较短的代码比较契合快速开发的需求。
>
> 这里的**高性能**以个人电脑或单节点设备的运行性能作为评估标准。我们不考虑许多工程要求，包括有限的内存、硬盘空间、契合计算化学程序架构等等；引入这些工程要求将会增加代码复杂性，不适合演示用途。
>
> 即使该仓库只是演示用途，相同算法、给足内存资源的前提下，程序的运行效率应已达到或超过目前主流计算化学程序。

**目前该仓库尚未完成。**

待完成目标：
- [x] CLI 与二进制文件
- [ ] 标准编译与运行流程 (Github Action)
- [ ] 说明文档
