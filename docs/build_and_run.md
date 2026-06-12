# 构建与运行说明

## 一、默认工具路径

脚本默认使用：

```text
/home/liumingjian/y_software/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu
/home/liumingjian/y_software/qemu-master/build/qemu-aarch64
```

如需替换，可设置环境变量：

```bash
export TOOLCHAIN_ROOT=/path/to/arm-gnu-toolchain
export QEMU=/path/to/qemu-aarch64
export QEMU_AARCH64=/path/to/qemu-aarch64
```

## 二、构建 SME AI 运算库

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/sme_ai_library
bash build.sh
```

输出：

```text
build/libsme_ai_ops.a
build/bench_sme_full_timing
build/verify_sme_ai_timing
```

## 三、运行逐函数验证

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/sme_ai_library
./run_qemu_verify_timing.sh
```

所有 case 应输出 `OK`。

## 四、运行 benchmark

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/sme_ai_library
./run_qemu_full_bench.sh
```

输出 `REF_ms`、`SME_ms` 和 `SPEEDUP`。

## 五、运行 llama.cpp 冒烟测试

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/llama.cpp
./scripts/run_qemu_cli_help.sh
./scripts/run_qemu_mini_gguf.sh
```

## 六、运行最小示例

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/examples
./build_demo_linear.sh
./run_qemu_demo_linear.sh
```

## 七、一键验收

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0
./run_all_checks.sh
```

快速模式：

```bash
RUN_BENCH=0 RUN_LLAMA=0 ./run_all_checks.sh
```
