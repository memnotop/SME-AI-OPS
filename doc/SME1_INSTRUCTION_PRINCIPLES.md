# SME1 主要指令与原理说明

本文总结本交付中实际使用到的 SME1 相关指令，并解释它们为什么适合 AI 计算。涉及的代码位置主要有：

- `jiaofu/sme_ai_library/src/sme1_gemm_tile_16x16.S`
- `jiaofu/sme_ai_library/src/sme_ai_ops.c`
- `llama.cpp_sme/ggml/src/ggml-cpu/arch/arm/repack.cpp`

本交付只使用 SME1，不使用 SME2。核心计算指令是 `FMOPA` 和 `SMOPA`。

## 一、SME1 的核心概念

SME 是 Arm 的 Scalable Matrix Extension，重点用于矩阵乘、卷积、attention 等 AI 运算。它和普通 NEON/SVE 的主要区别是：SME 不只使用向量寄存器 `Z`，还提供一个二维矩阵累加器 `ZA`。

### 1. `Z` 向量寄存器

`Z0`、`Z1` 等是 SVE/SME 使用的向量寄存器。向量长度不是写死的，而是由硬件决定。

在本项目的目标 CPU 上，SME vector length 是 512 bit：

- `cntb = 64`：一个向量有 64 字节。
- `cntw = 16`：一个向量有 16 个 32-bit word。
- 对 FP32 来说，一个向量正好可以放 16 个 float。

### 2. `P` 谓词寄存器

`P0`、`P1` 等是 predicate register，用于控制哪些向量 lane 参与运算。

例如：

```asm
ptrue p0.s
```

表示把 `p0` 中所有 32-bit lane 都设为 active。后续 `fmopa` 只会使用 active 的 lane。

再例如：

```asm
whilelt p0.b, xzr, x9
```

表示按 byte lane 生成谓词：从 0 到 `x9 - 1` 的 lane 有效，超过范围的 lane 无效。这样可以处理不满整个向量长度的小块数据。

### 3. `ZA` 矩阵累加器

`ZA` 是 SME 的二维矩阵寄存器，用于保存矩阵乘的累加结果。

普通向量指令通常一次只处理一维向量，例如：

```text
C[i] += A[i] * B[i]
```

SME 的外积指令可以一次处理二维结果，例如：

```text
C[row][col] += A[row] * B[col]
```

这就是 `FMOPA` 和 `SMOPA` 的核心思想：把一个列向量和一个行向量做外积，直接累加到 `ZA` 矩阵中。

当 `cntw = 16` 时，`ZA.S` 可以自然表示一个 16x16 的 FP32/INT32 累加 tile，非常适合本项目的 16x16 GEMM 微内核。

## 二、进入和退出 SME 模式

### 1. `smstart`

示例：

```asm
smstart
```

作用：进入 SME streaming mode。

SME 的 `ZA` 矩阵寄存器和部分 streaming 语义需要在 streaming mode 下使用。本项目中，执行 SME 微内核前会先调用：

```c
__asm__ volatile("smstart");
```

对应位置：

- `jiaofu/sme_ai_library/src/sme_ai_ops.c`
- `llama.cpp_sme/ggml/src/ggml-cpu/arch/arm/repack.cpp`

### 2. `smstop`

示例：

```asm
smstop
```

作用：退出 SME streaming mode。

进入 streaming mode 后必须在合适位置退出，否则会影响后续普通代码的运行状态。本项目在每段 SME 计算结束后调用：

```c
__asm__ volatile("smstop");
```

### 3. 为什么要成对出现

`smstart` 和 `smstop` 可以理解为 SME 计算区间的边界：

```text
smstart
    使用 ZA 和 SME 外积指令进行矩阵计算
smstop
```

这样做的好处是：

- SME 相关状态只在需要时打开。
- 普通 C 代码、libc 函数和其他非 SME 代码不长期处于 streaming mode。
- 更容易定位 SME 代码范围。

## 三、清空 ZA 累加器

### `zero {za}`

示例：

```asm
zero {za}
```

作用：把整个 `ZA` 矩阵累加器清零。

矩阵乘一般形式是：

```text
C = A x B
```

实现时通常写成累加：

```text
C = 0
for k:
    C += A[:, k] x B[k, :]
```

`zero {za}` 就对应这里的 `C = 0`。如果不清零，`ZA` 中可能保留上一次计算的结果，当前 tile 就会出错。

本项目用法：

```asm
ptrue p0.s
zero {za}
```

先设置谓词，再清空 `ZA`，然后进入 K 维循环。

## 四、FP32 外积累加：`FMOPA`

### 1. 指令形式

本项目中使用：

```asm
fmopa za0.s, p0/m, p0/m, z0.s, z1.s
```

含义：

- `z0.s`：左侧 FP32 向量，作为矩阵 A 的一个列向量。
- `z1.s`：右侧 FP32 向量，作为矩阵 B 的一个行向量。
- `p0/m`：谓词控制哪些 lane 参与计算。
- `za0.s`：结果累加到 `ZA` 的 FP32 tile 中。

### 2. 数学含义

如果 `z0.s` 中有 16 个 FP32：

```text
a0, a1, ..., a15
```

`z1.s` 中有 16 个 FP32：

```text
b0, b1, ..., b15
```

那么一次 `FMOPA` 可以理解为执行：

```text
for i in 0..15:
    for j in 0..15:
        ZA[i][j] += a[i] * b[j]
```

也就是一次 16x16 外积累加。

### 3. 在本项目中的用途

代码位置：

```text
jiaofu/sme_ai_library/src/sme1_gemm_tile_16x16.S
```

核心循环：

```asm
ptrue p0.s
zero {za}

1:
    cbz x3, 2f
    ldr z0, [x0]
    ldr z1, [x1]
    fmopa za0.s, p0/m, p0/m, z0.s, z1.s
    add x0, x0, #64
    add x1, x1, #64
    sub x3, x3, #1
    b 1b
```

这段代码实现的是一个 16x16 GEMM tile：

```text
C[16][16] += A[16][K] x B[K][16]
```

每次循环处理一个 K：

- `ldr z0, [x0]` 读取 A 的 16 个 FP32。
- `ldr z1, [x1]` 读取 B 的 16 个 FP32。
- `fmopa` 做一次 16x16 外积并累加到 `ZA`。
- `x0` 和 `x1` 每次增加 64 字节，因为 16 个 FP32 正好是 64 字节。

### 4. 为什么 `FMOPA` 适合 GEMM

普通标量实现 16x16 tile 的一个 K 步需要：

```text
16 x 16 = 256 次乘加
```

`FMOPA` 把这 256 个输出元素的乘加组织成一次矩阵外积操作，由硬件矩阵累加器完成。这样能减少循环开销、减少寄存器搬运，并更充分利用矩阵计算单元。

## 五、INT8 外积累加：`SMOPA`

### 1. 指令形式

本项目中使用：

```asm
smopa za0.s, p0/m, p1/m, z0.b, z1.b
```

以及：

```asm
smopa za0.s, p0/m, p1/m, z2.b, z3.b
```

含义：

- `z0.b`、`z2.b`：左侧 signed int8 数据，来自运行时量化后的激活。
- `z1.b`、`z3.b`：右侧 signed int8 数据，来自解包后的量化权重。
- `p0/m`、`p1/m`：分别控制左、右输入中哪些 byte lane 有效。
- `za0.s`：结果以 signed int32 形式累加到 `ZA`。

### 2. 数学含义

`SMOPA` 是 signed integer matrix outer product accumulate。

可以把它理解为：

```text
ZA_int32[row][col] += int8_left[row] * int8_right[col]
```

实际硬件会按照 SME 的规则把 byte 输入组织成 32-bit 累加单元，所以最终结果保存在 `ZA.S` 形式的 int32 tile 中。

### 3. 在 llama.cpp 中的用途

代码位置：

```text
llama.cpp_sme/ggml/src/ggml-cpu/arch/arm/repack.cpp
```

核心片段：

```asm
zero {za}
whilelt p0.b, xzr, x9
whilelt p1.b, xzr, x10
ld1b { z0.b }, p0/z, [a_lo]
ld1b { z1.b }, p1/z, [w_lo]
smopa za0.s, p0/m, p1/m, z0.b, z1.b
ld1b { z2.b }, p0/z, [a_hi]
ld1b { z3.b }, p1/z, [w_hi]
smopa za0.s, p0/m, p1/m, z2.b, z3.b
```

这里服务于 `Q2_K x Q8_K` 矩阵乘：

- `Q2_K` 是 2-bit 权重量化格式。
- `Q8_K` 是 8-bit 激活量化格式。
- 计算前会把 2-bit 权重解包成 int8 小块。
- `SMOPA` 对 int8 数据做外积累加。
- 累加结果是 int32。
- 最后再乘以 scale/min 等量化参数，还原成 FP32 输出。

### 4. 为什么 `SMOPA` 适合量化模型

量化模型中，权重和激活通常是 int8 或更低 bit 宽。矩阵乘的核心变成：

```text
int32_acc += int8_activation * int8_weight
```

`SMOPA` 正是为这种 int8 到 int32 的矩阵外积累加设计的。它适合：

- LLM 中的量化权重矩阵乘。
- `Q2_K/Q8_K`、`Q4_K/Q8_K` 等量化格式的核心点积。
- batch 较小但矩阵维度较大的推理场景。

本项目的 llama.cpp SME 优化重点就是让 `mini.gguf` 中的 `Q2_K` tensor 尽可能进入这条 `SMOPA` 路径。

## 六、谓词和加载/存储指令

### 1. `ptrue`

示例：

```asm
ptrue p0.s
```

作用：把指定粒度的所有 lane 设置为 active。

在 512-bit 向量长度下：

- `ptrue p0.s` 表示 16 个 FP32 lane 全部有效。
- 后续 `fmopa` 会使用完整 16x16 tile。

### 2. `whilelt`

示例：

```asm
whilelt p0.b, xzr, x9
```

作用：生成一个“从 0 到 x9-1 有效”的谓词。

它适合处理不是整向量长度的数据。例如 llama.cpp 的 Q2_K 路径中，左侧和右侧 byte 数不一定都使用完整 64 字节，因此用 `whilelt` 精确控制有效 lane。

### 3. `ldr z0, [x0]`

示例：

```asm
ldr z0, [x0]
```

作用：从内存中读取一个完整向量到 `z0`。

在 `cntw=16` 时，一次读取：

```text
16 x FP32 = 64 字节
```

AI 库中的 FP32 GEMM 微内核用它读取 A 和 B 的 16 个元素。

### 4. `ld1b`

示例：

```asm
ld1b { z0.b }, p0/z, [a_lo]
```

作用：按 byte 从内存加载到向量寄存器，受 predicate 控制。

- `p0/z` 表示 inactive lane 会被置零。
- 适合加载 int8 量化数据。

llama.cpp 的 `SMOPA` 路径使用 `ld1b` 读取 int8 激活和 int8 权重。

### 5. `str za[...]`

示例：

```asm
str za[w12, #0], [x2]
```

作用：把 `ZA` 中的一段结果存回内存。

AI 库中，`FMOPA` 完成 16x16 tile 累加后，需要把 `ZA` 中的 FP32 结果写到输出矩阵 `C`。

### 6. `st1w`

示例：

```asm
st1w { za0h.s[w12, #0] }, p2, [x11]
```

作用：把 `ZA` 中的 32-bit word 按谓词控制写回内存。

llama.cpp 的 `SMOPA` 路径用它把 int32 累加结果取出，随后在 C++ 中乘以 scale/min，得到最终 FP32 输出。

## 七、查询向量长度：`cntb` 和 `cntw`

### 1. `cntb`

示例：

```asm
cntb x0
```

作用：返回当前 streaming vector length 中包含多少字节。

目标 CPU 为 512-bit 时：

```text
cntb = 512 / 8 = 64
```

### 2. `cntw`

示例：

```asm
cntw x0
```

作用：返回当前向量中包含多少个 32-bit word。

目标 CPU 为 512-bit 时：

```text
cntw = 512 / 32 = 16
```

### 3. 为什么本项目关注 `cntw=16`

AI 库中的 SME1 微内核设计为 16x16 tile：

```text
A tile: 16 x K
B tile: K x 16
C tile: 16 x 16
```

当 `cntw=16` 时，一个向量正好表示 16 个 FP32 元素，`FMOPA` 正好对应 16x16 外积累加。如果硬件只给出 `cntw=4` 或 `cntw=8`，当前这个 16x16 微内核就不匹配，需要重写 tile 大小。

## 八、两条核心路径对比

| 路径 | 主要指令 | 输入数据 | 累加结果 | 用途 |
| --- | --- | --- | --- | --- |
| SME AI 库 GEMM | `FMOPA` | FP32 向量，通常由 FP16 输入转换/打包得到 | FP32 `ZA.S` | Linear、MLP、Attention、Conv2D 的 GEMM 部分 |
| llama.cpp Q2_K | `SMOPA` | int8 激活和 int8 解包权重 | INT32 `ZA.S` | `Q2_K x Q8_K` 量化矩阵乘 |

简单理解：

- `FMOPA` 面向浮点矩阵乘。
- `SMOPA` 面向量化整数矩阵乘。
- 二者都使用 `ZA` 做二维累加，因此比普通一维向量循环更接近矩阵乘本身。

## 九、完整计算流程示意

### 1. SME AI 库 FP32 GEMM

```text
FP16 输入/权重
    ↓ 转换和打包
FP32 A tile + FP32 B tile
    ↓ smstart
zero {za}
    ↓ K 循环
FMOPA: ZA += A_vector outer B_vector
    ↓
store ZA to output C
    ↓ smstop
FP32 输出
```

### 2. llama.cpp Q2_K 量化矩阵乘

```text
Q2_K 权重 + Q8_K 激活
    ↓ 解包 Q2_K 权重为 int8 小块
int8 weight + int8 activation
    ↓ smstart
zero {za}
    ↓
SMOPA: INT32_ZA += int8_activation outer int8_weight
    ↓
store INT32_ZA
    ↓ 乘 scale/min 做反量化
FP32 输出
    ↓ smstop
```

## 十、和普通实现相比的优势

### 1. 更适合矩阵乘

普通标量或普通向量实现经常围绕一维向量展开，而矩阵乘本质是二维累加。SME 的 `ZA` 直接提供二维累加器，减少了大量手动维护多个 accumulator 的代码。

### 2. 更高的数据复用

一次 `FMOPA` 或 `SMOPA` 会把一个左向量和一个右向量组合成多个输出元素。左向量元素会被复用于多个列，右向量元素会被复用于多个行。

### 3. 更适合 AI 推理中的大矩阵

LLM 和 CNN 中大量算子最终都可以归结为 GEMM：

- Linear
- MLP
- Attention 的 Q/K/V/O 投影
- im2col 后的 Conv2D
- 量化权重矩阵乘

因此 SME1 的外积累加指令能直接作用于这些热点计算。

## 十一、注意事项

- 本交付的 SME 版本面向支持 SME1 的目标 CPU。
- `FMOPA` 路径依赖 `cntw=16` 的 512-bit SME 长度。
- llama.cpp 的 `SMOPA` 路径主要覆盖 `Q2_K` tensor，不代表所有 tensor 都使用 SME。
- QEMU 可以用于验证程序能运行和指令能被模拟，但不能代表真实硬件性能。
- 如果要临时关闭 llama.cpp 中的 Q2_K SME 路径，可以使用：

```bash
GGML_Q2K_SME=0 ./bin/sme/llama-bench -m models/mini.gguf -p 16 -n 8 -r 3 -t 4 -ngl 0 --no-warmup
```
