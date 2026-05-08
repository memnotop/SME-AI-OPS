# llama.cpp SME1 Q2_K 优化说明

本文说明 `~/dachuang/llama.cpp_ref` 与 `~/dachuang/llama.cpp_sme` 的差异、`mini.gguf` 的模型特点，以及 SME1 如何加速 llama.cpp。

## 相关目录

```text
~/dachuang/
├── llama.cpp_ref/          # 参考源码，未加入本次自研 SME1 Q2_K 路径
├── llama.cpp_sme/          # SME 源码，加入本次自研 SME1 Q2_K 路径
├── kmodules/
│   └── mini.gguf           # 原始测试模型
└── jiaofu/
    └── llama.cpp/          # 最终交付目录，含 REF/SME 二进制和 mini.gguf
```

最终交付时使用：

```text
~/dachuang/jiaofu/llama.cpp/bin/ref/llama-bench
~/dachuang/jiaofu/llama.cpp/bin/ref/llama-cli
~/dachuang/jiaofu/llama.cpp/bin/sme/llama-bench
~/dachuang/jiaofu/llama.cpp/bin/sme/llama-cli
```

## mini.gguf 模型分析

`mini.gguf` 是 TinyLlama 1.1B Chat 的 GGUF 模型：

| 项目 | 数值 |
| --- | --- |
| 架构 | `llama` |
| 模型名 | `tinyllama_tinyllama-1.1b-chat-v1.0` |
| GGUF version | `3` |
| 文件量化类型 | `Q2_K Medium` |
| tensor 数量 | `201` |
| Transformer 层数 | `22` |
| context length | `2048` |
| embedding length | `2048` |
| feed-forward length | `5632` |
| attention heads | `32` |
| KV heads | `4` |

tensor 类型分布：

| tensor 类型 | 数量 | 说明 |
| --- | ---: | --- |
| `Q2_K` | 45 | 本次 SME1 优化主要覆盖的权重量化类型 |
| `Q3_K` | 110 | 仍走 llama.cpp 原有非自研 SME 路径 |
| `F32` | 45 | norm、scale 等浮点参数 |
| `Q6_K` | 1 | 少量量化权重 |

典型 `Q2_K` tensor 包括：

- `token_embd.weight`
- `blk.*.attn_q.weight`
- `blk.*.attn_k.weight`

因此，SME 优化不是覆盖整个模型的所有算子，而是重点优化 `Q2_K` 权重参与的矩阵乘路径。这个模型有 45 个 `Q2_K` tensor，适合展示 Q2_K 路径优化。

## 修改了哪些代码

### 1. `llama.cpp_sme/build_aarch64.sh`

作用：构建 SME 版本的 `llama-bench` 和 `llama-cli`。

主要修改：

- 增加 `-march=armv9-a+sve+sme`，让编译器允许生成 SME/SVE 相关代码。
- 增加 `$ORIGIN` RPATH，让交付目录中的可执行文件优先加载同目录动态库。
- 构建目标包含 `llama-bench` 和 `llama-cli`，不只构建 benchmark。

### 2. `llama.cpp_ref/build_aarch64.sh`

作用：构建 REF 版本。

主要修改：

- 同样增加 `$ORIGIN` RPATH，便于交付目录独立运行。
- 不添加 `+sme` 编译标志，因此不会启用本次 SME1 优化代码。

### 3. `llama.cpp_sme/ggml/src/ggml-cpu/repack.cpp`

作用：决定某个 tensor 在矩阵乘时使用哪种 repack/GEMM 路径。

本次增加逻辑：

```cpp
if (ggml_cpu_has_sme()) {
    if (cur->ne[1] % 8 == 0) {
        return &q2_K_8x8_q8_K;
    }
}
```

含义：

- 当前 tensor 类型是 `Q2_K` 时，如果运行环境支持 SME，并且矩阵行数满足 8 对齐，就选择 `q2_K_8x8_q8_K` 路径。
- 这个路径最终会调用 `arch/arm/repack.cpp` 中的 SME1 实现。

### 4. `llama.cpp_sme/ggml/src/ggml-cpu/arch/arm/repack.cpp`

这是本次优化的核心文件。

新增内容：

- `ggml_arm_q2k_use_sme()`：运行时判断是否启用 SME Q2_K 路径。
- `ggml_arm_sme_begin()`：执行 `smstart`，进入 SME streaming mode。
- `ggml_arm_sme_end()`：执行 `smstop`，退出 SME streaming mode。
- `ggml_arm_sme_smopa_dot8_4x8()`：使用 SME1 `smopa` 指令做 8-bit signed outer product。
- `ggml_gemv_q2_K_8x8_q8_K()`：Q2_K x Q8_K 的向量乘/小 batch 路径。
- `ggml_gemm_q2_K_8x8_q8_K()`：Q2_K x Q8_K 的矩阵乘路径。
- `ggml_arm_q2k_get_meta_cache()`：缓存 Q2_K 的 scale/min 元数据，减少重复解包开销。

本文件只使用 SME1 指令：

- `smstart`
- `smopa`
- `smstop`

没有使用 SME2 指令。

说明：llama.cpp 上游源码中可能保留 KleidiAI 相关的 `sme2` 文件名或字符串，但本次构建没有开启 `GGML_CPU_KLEIDIAI`，本交付的自研 Q2_K 优化路径也没有调用 SME2 指令。最终以 `libggml-cpu.so` 反汇编中出现的 `smstart/smopa/smstop` 为准。

## SME1 如何加速

llama.cpp 推理中的大部分耗时来自矩阵乘。对 `Q2_K` 权重来说，计算过程可以理解为：

1. 模型权重以 `Q2_K` 形式存储，每个权重只占约 2 bit，同时带有 scale/min 元数据。
2. 运行时输入激活会被量化成 `Q8_K`。
3. 原始计算需要做 `Q2_K x Q8_K` 的点积，再乘 scale/min 还原为 FP32 结果。
4. SME 版本把 2-bit 权重解出为 8-bit 小块，把输入激活也组织为 8-bit 小块。
5. 使用 SME1 `smopa` 一次做外积累加，把多个 int8 乘加结果累加到 ZA 矩阵寄存器。
6. 最后把 int32 累加结果取出，乘以 `Q2_K` 和 `Q8_K` 的缩放系数，得到 FP32 输出。

`smopa` 的优势在于它面向矩阵外积设计，比普通标量循环更适合做一小块矩阵乘。当前实现使用 8 列权重块和 4 行激活块，对应 `q2_K_8x8_q8_K` 路径。

## 启用条件

SME 路径需要同时满足：

- 编译时为 AArch64 且开启 `__ARM_FEATURE_SME`。
- SME 版二进制运行在支持 SME1 的 CPU 上；本交付的 SME 版是面向 SME 目标 CPU 的专用构建。
- `ggml_cpu_has_sme()` 返回 true。注意在当前单后端构建中，这个函数主要反映编译时 SME 后端是否启用，因此不要把 SME 版二进制拿到不支持 SME 的 CPU 上运行。
- `ggml_cpu_get_sve_cnt() >= 32`，即向量长度至少 32 字节；目标 CPU 为 512-bit 时 `cntb=64`。
- tensor 类型为 `Q2_K`。
- tensor 行数满足 8 对齐。
- 没有设置 `GGML_Q2K_SME=0`。

关闭 SME Q2_K 路径：

```bash
GGML_Q2K_SME=0 ./bin/sme/llama-bench -m models/mini.gguf -p 16 -n 8 -r 3 -t 4 -ngl 0 --no-warmup
```

## 如何确认 SME 指令存在

可以用交叉工具链反汇编 SME 版本核心库：

```bash
/home/liumingjian/y_software/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-objdump -d \
  /home/liumingjian/dachuang/jiaofu/llama.cpp/bin/sme/libggml-cpu.so.0.9.8 | rg "smstart|smopa|smstop"
```

能看到 `smstart`、`smopa`、`smstop`，说明 SME1 优化代码已经编译进 `libggml-cpu.so`。

## 如何重新编译

REF 版本：

```bash
cd /home/liumingjian/dachuang/llama.cpp_ref
./build_aarch64.sh
```

SME 版本：

```bash
cd /home/liumingjian/dachuang/llama.cpp_sme
./build_aarch64.sh
```

编译后主要产物位于：

```text
llama.cpp_ref/aarch64build/bin/llama-bench
llama.cpp_ref/aarch64build/bin/llama-cli
llama.cpp_sme/aarch64build/bin/llama-bench
llama.cpp_sme/aarch64build/bin/llama-cli
```

## 运行建议

目标 CPU 上建议先运行：

```bash
cd /home/liumingjian/dachuang/jiaofu/llama.cpp
THREADS=2 PROMPT_TOKENS=1 GEN_TOKENS=0 REPEAT=1 ./scripts/run_target_bench_mini.sh
VERSION=sme PROMPT="Hello." N_PREDICT=16 ./scripts/run_target_cli_mini.sh
```

如果需要做性能对照：

```bash
./bin/ref/llama-bench -m models/mini.gguf -p 16 -n 8 -r 3 -t 4 -ngl 0 --no-warmup
./bin/sme/llama-bench -m models/mini.gguf -p 16 -n 8 -r 3 -t 4 -ngl 0 --no-warmup
GGML_Q2K_SME=0 ./bin/sme/llama-bench -m models/mini.gguf -p 16 -n 8 -r 3 -t 4 -ngl 0 --no-warmup
```

第三条命令用于在同一个 SME 二进制内关闭自研 Q2_K 路径，帮助确认性能差异是否来自本次 SME1 优化。
