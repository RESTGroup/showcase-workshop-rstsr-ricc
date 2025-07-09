# REST Workshop：Rust 计算化学程序开发

该仓库以 RHF 与 RCCSD(T) 为例子，用于演示 Rust 语言下，计算化学程序的**快速开发**与**高性能实现**的一种策略与工程实践。

该仓库将使用 [RSTSR](https://github.com/RESTGroup/rstsr) ([REST](https://gitee.com/RESTGroup/REST) 子项目) 作为数学库工具，[libcint](https://github/sunqm/libcint) (Rust [binding and wrapper](https://github.com/ajz34/libcint-rs)) 作为电子积分引擎。我们将展示 RSTSR 在单一语言框架 (Rust) 下，可以同时兼顾**开发效率**与**运行效率**。