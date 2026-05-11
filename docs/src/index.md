# REST Workshop：Rust 计算化学程序开发

该仓库以 RHF 与 RCCSD(T) 为例子，用于演示 Rust 语言下，计算化学程序的**快速开发**与**高性能实现**的一种策略与工程实践。

该仓库将使用 [RSTSR](https://github.com/RESTGroup/rstsr) ([REST](https://gitee.com/RESTGroup/REST) 子项目) 作为数学库工具，[libcint](https://github/sunqm/libcint) (Rust [binding and wrapper](https://github.com/ajz34/libcint-rs)) 作为电子积分引擎。我们将展示 RSTSR 在单一语言框架 (Rust) 下，可以同时兼顾**开发效率**与**运行效率**。

## 编译与测试

```bash
cargo build
cargo test --lib -- playground_ri_ccsdt --exact --nocapture
```

该项目由于开发过程也需要关注效率，因此在 debug 下仍然开启了 `opt-level=2` 优化选项。上述编译后的程序效率与 release 没有明显差异。

上述程序测试的是水分子 def2-TZVP 基组下的 RI-RCCSD(T) 能量计算。

该编译流程会从 github 上下载电子积分库 libcint 并作静态链接 (static linking)。如果 github 下载较慢，可以先下载好 libcint 的源码，并声明到环境变量 `CINT_SRC`。
