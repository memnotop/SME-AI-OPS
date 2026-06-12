# 性能结果说明

## 一、当前可引用结果

当前发布包建议引用一键验收报告中的最终运行结果，而不是散落在历史目录中的旧日志。

生成方式：

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0
./run_all_checks.sh
```

报告位置：

```text
acceptance_reports/<时间戳>/acceptance_report.md
```

benchmark 日志位置：

```text
acceptance_reports/<时间戳>/logs/
```

## 二、可展示指标

| 指标 | 含义 |
| --- | --- |
| API 验证通过数量 | 说明公开 API 均能运行 |
| `REF_ms` / `SME_ms` | 说明同一算子的两条路径可对照 |
| `SPEEDUP` | QEMU 环境下的参考比值 |
| `MAX_ABS` / `MAX_REL` | REF 与 SME 结果误差 |
| llama.cpp `t/s` | 示例模型加载和 benchmark 输出 |

## 三、答辩建议

可以强调：

- SME 路径已经能被调用；
- 公开 API 均有测试；
- llama.cpp Q2_K 路径已接入；
- 发布包能一键生成验收报告。

需要避免：

- 把 QEMU wall time 当成飞腾/目标 CPU 的确定性能提升；
- 把 Q2_K 路径接入说成全模型所有量化格式均已优化；
- 把工具链配置说成已完成 GCC/LLVM 后端改造。

## 四、后续真机测试建议

目标 CPU 上建议补充：

```bash
./sme_ai_library/build/verify_sme_ai_timing
./sme_ai_library/build/bench_sme_full_timing
./llama.cpp/scripts/run_target_bench_mini.sh
```

并记录：

- CPU 型号和 SME/SVE 能力；
- 线程数；
- vector length；
- 每个算子的平均耗时；
- perf/PMU 热点。
