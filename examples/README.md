# 最小用户示例

本目录展示如何在第三方 C 程序中链接 `libsme_ai_ops.a`，并调用 `sme_ai_linear_sme_packed_f16f32`。

## 一、构建示例

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0/examples
./build_demo_linear.sh
```

脚本会生成：

```text
examples/build/demo_linear
```

## 二、通过 QEMU 运行

```bash
./run_qemu_demo_linear.sh
```

预期输出包含：

```text
DEMO_LINEAR PASS
```

## 三、示例说明

`demo_linear.c` 完成以下流程：

1. 初始化 `sme_ai_workspace`。
2. 初始化 `sme_ai_packed_linear_weight`。
3. 调用 `sme_ai_linear_ref_f16f32` 得到参考结果。
4. 调用 `sme_ai_linear_sme_packed_f16f32` 得到 SME 结果。
5. 比较两者最大绝对误差。
6. 释放 packed weight 和 workspace。

该示例不是 benchmark，只用于说明库函数调用方式。
