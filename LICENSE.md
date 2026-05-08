# 许可与第三方组件说明

本发布包用于大学生创新训练项目终检交付和后续教学科研验证。

## 一、自研部分

以下内容为本项目整理和实现的交付内容：

- `sme_ai_library/include/sme_ai_ops.h`
- `sme_ai_library/src/sme_ai_ops.c`
- `sme_ai_library/src/sme1_gemm_tile_16x16.S`
- `sme_ai_library/test_bench/bench_sme_full_timing.c`
- `sme_ai_library/test_bench/verify_sme_ai_timing.c`
- `sme_ai_library/*.sh`
- 根目录中文发布说明文件
- `doc/SME1_INSTRUCTION_PRINCIPLES.md`

这些内容可用于本项目验收、复现实验、课程展示和后续科研开发。若需对外公开发布，应由项目组和指导教师根据学校相关管理要求另行确认许可方式。

## 二、第三方组件

本发布包包含第三方开源项目或其构建产物：

- `llama.cpp/` 中的 `llama-bench`、`llama-cli`、`libllama`、`libggml*` 等来自 llama.cpp 及 ggml 生态的构建产物。
- `llama.cpp/bin/*/libstdc++*`、`libgcc_s*`、`libgomp*` 等来自 GNU 工具链运行库。
- `llama.cpp/models/mini.gguf` 是用于功能验证的 GGUF 示例模型文件。

第三方组件应遵守其原项目或原发行方的许可证、模型许可证和使用条款。本发布包仅将其作为项目复现和验收所需的运行环境一并打包。

## 三、使用边界

- 本发布包不用于商业发行。
- 本发布包中的 QEMU 测试结论仅用于说明程序可运行、调用链可打通和数值结果可验证。
- 对真实目标 CPU 的性能结论，应以目标硬件实测数据为准。
