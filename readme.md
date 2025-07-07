# Rust 语言计算化学开发演示：极短 RHF 与高效 RI-RCCSD 的简易实现

该仓库以 RHF 与 RI-RCCSD 为例子，用于演示 Rust 语言下，计算化学程序的**快速开发**与**高性能实现**的一种策略与工程实践。

该仓库将使用 [RSTSR](https://github.com/RESTGroup/rstsr) ([REST](https://gitee.com/RESTGroup/REST) 子项目) 作为数学库工具，[libcint](https://github/sunqm/libcint) (Rust [binding and wrapper](https://github.com/ajz34/libcint-rs)) 作为电子积分引擎。我们将展示 RSTSR 在单一语言框架 (Rust) 下，可以同时兼顾**开发效率**与**运行效率**。

> 这里的**高效**，是在不考虑许多工程要求的情况下的高性能。工程要求包括有限的内存、硬盘空间、契合计算化学程序架构等等；引入这些工程要求将会增加代码复杂性，不适合演示用途。

**目前该仓库尚未完成。**

待完成目标：
- [ ] CLI 与二进制文件
- [ ] 标准编译与运行流程 (Github Action)
- [ ] 说明文档
