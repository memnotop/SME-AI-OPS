# llama.cpp REF/SME 交付说明

本目录提供两套 AArch64 llama.cpp 可执行环境：

- `bin/ref/`：参考版本，不包含本次自研 SME1 Q2_K 优化。
- `bin/sme/`：SME 版本，包含本次自研 SME1 Q2_K 优化。

两套目录各自包含 `llama-bench`、`llama-cli` 和对应动态库，避免 REF/SME 混用同名 `libggml-cpu.so`。

## 目录结构

```text
llama.cpp/
├── README.md
├── bin/
│   ├── ref/
│   │   ├── llama-bench
│   │   ├── llama-cli
│   │   └── lib*.so*
│   └── sme/
│       ├── llama-bench
│       ├── llama-cli
│       └── lib*.so*
├── doc/
│   └── SME_Q2K_OPTIMIZATION.md
├── logs/
├── models/
│   └── mini.gguf
└── scripts/
    ├── run_qemu_cli_help.sh
    ├── run_qemu_mini_gguf.sh
    ├── run_target_bench_mini.sh
    └── run_target_cli_mini.sh
```

## 在目标 CPU 上运行

运行 `llama-bench` 对比：

```bash
cd /home/liumingjian/dachuang/jiaofu/llama.cpp
./scripts/run_target_bench_mini.sh
```

运行 SME 版 `llama-cli`：

```bash
cd /home/liumingjian/dachuang/jiaofu/llama.cpp
VERSION=sme PROMPT="Hello, introduce yourself briefly." N_PREDICT=16 ./scripts/run_target_cli_mini.sh
```

运行 REF 版 `llama-cli`：

```bash
VERSION=ref PROMPT="Hello." N_PREDICT=16 ./scripts/run_target_cli_mini.sh
```

## 在 x86 主机上用 QEMU 冒烟测试

QEMU 只用于确认程序能启动、模型能加载、输出格式正常，不用于最终性能结论。

```bash
cd /home/liumingjian/dachuang/jiaofu/llama.cpp
./scripts/run_qemu_cli_help.sh
./scripts/run_qemu_mini_gguf.sh
```

`run_qemu_mini_gguf.sh` 会把日志写入 `logs/`。

## 常用参数

- `THREADS=2`：设置线程数。
- `PROMPT_TOKENS=1`：`llama-bench` 的 prompt token 数。
- `GEN_TOKENS=0`：`llama-bench` 的生成 token 数。
- `REPEAT=1`：`llama-bench` 重复次数。
- `VERSION=sme/ref`：选择 `llama-cli` 的 SME 或 REF 版本。
- `GGML_Q2K_SME=0`：只对 SME 版本有效，用于临时关闭自研 Q2_K SME 路径。

示例：

```bash
THREADS=4 PROMPT_TOKENS=16 GEN_TOKENS=8 REPEAT=3 ./scripts/run_target_bench_mini.sh
GGML_Q2K_SME=0 ./bin/sme/llama-bench -m models/mini.gguf -p 16 -n 8 -r 3 -t 4 -ngl 0 --no-warmup
```

详细源码修改和 SME 加速原理见 [doc/SME_Q2K_OPTIMIZATION.md](doc/SME_Q2K_OPTIMIZATION.md)。
