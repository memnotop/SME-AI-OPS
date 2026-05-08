# SME1 AI 运算库与两个 benchmark

本目录只保留 SME1 AI 运算库源码和两个 benchmark：

- `bench_sme_full_timing`：全面 REF/SME 时间对比。
- `verify_sme_ai_timing`：逐个库函数运行验证，并输出每个库函数的结果和耗时。

本库只使用 SME1，不使用 SME2。当前微内核面向 `cntw=16`，也就是 `512-bit SME vector length`。

## 目录结构

```text
sme_ai_library/
├── README.md
├── build.sh
├── include/
│   └── sme_ai_ops.h
├── src/
│   ├── sme_ai_ops.c
│   └── sme1_gemm_tile_16x16.S
├── test_bench/
│   ├── bench_sme_full_timing.c
│   └── verify_sme_ai_timing.c
├── run_qemu_full_bench.sh
├── run_qemu_verify_timing.sh
└── build/
    ├── libsme_ai_ops.a
    ├── bench_sme_full_timing
    └── verify_sme_ai_timing
```

## 构建

```bash
cd /home/liumingjian/dachuang/jiaofu/sme_ai_library
bash build.sh
```

构建后只生成：

- `build/libsme_ai_ops.a`
- `build/bench_sme_full_timing`
- `build/verify_sme_ai_timing`

## 运行

在真实目标 CPU 上运行：

```bash
cd /home/liumingjian/dachuang/jiaofu/sme_ai_library
./build/bench_sme_full_timing
./build/verify_sme_ai_timing
```

在当前 x86 主机上通过 QEMU 运行：

```bash
cd /home/liumingjian/dachuang/jiaofu/sme_ai_library
./run_qemu_full_bench.sh
./run_qemu_verify_timing.sh
```

## 两个 benchmark 的区别

- `bench_sme_full_timing` 用于性能对比。它把同一个高层算子分别跑 REF 路径和 SME 路径，输出 `REF_ms`、`SME_ms`、`SPEEDUP`、误差和 checksum。
- `verify_sme_ai_timing` 用于逐函数验证。它不计算 speedup，而是逐个调用公开库函数，输出 `OK/FAIL`、运行次数、输出元素数量、耗时和 checksum。

更完整的交付结构和库函数说明见上一层目录：

```text
/home/liumingjian/dachuang/jiaofu/README.md
```
