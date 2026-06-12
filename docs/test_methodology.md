# 测试方法说明

## 一、测试目标

本项目测试分为四类：

1. 构建测试：确认源码能重新生成静态库和测试程序。
2. 指令测试：确认关键二进制中包含 SME1 指令。
3. 正确性测试：逐项调用公开 API，检查状态和 checksum。
4. 路径测试：运行 REF/SME benchmark 和 llama.cpp 冒烟测试。

## 二、逐函数验证

脚本：

```bash
sme_ai_library/run_qemu_verify_timing.sh
```

程序：

```text
sme_ai_library/build/verify_sme_ai_timing
```

输出字段：

| 字段 | 含义 |
| --- | --- |
| `CASE` | API 或测试项名称 |
| `OK` | 是否通过 |
| `ITERS` | 重复次数 |
| `ELEMS` | 输出元素数量 |
| `QEMU_ms` | QEMU 包裹后的墙钟时间 |
| `CHECKSUM` | 输出校验和 |

## 三、benchmark

脚本：

```bash
sme_ai_library/run_qemu_full_bench.sh
```

输出字段：

| 字段 | 含义 |
| --- | --- |
| `CASE` | 高层算子名称 |
| `REF_ms` | 参考路径耗时 |
| `SME_ms` | SME 路径耗时 |
| `SPEEDUP` | `REF_ms / SME_ms` |

## 四、误差口径

Linear、MLP、SwiGLU、Attention、Conv2D 等高层算子会比较 REF 与 SME 输出，输出最大绝对误差和最大相对误差。由于部分路径存在 FP16 输入和 FP32 累加/输出转换，结题说明中应同时报告 `max_abs` 和 `max_rel`，避免只看绝对误差。

## 五、QEMU 数据说明

QEMU 可用于确认：

- 程序能够启动；
- SME 指令语义能够被模拟；
- 数值检查能够通过；
- 调用链能够打通。

QEMU 不适合直接作为真实 CPU 性能结论。真实性能需要目标 CPU 上的 wall time、perf 或 PMU 数据。
