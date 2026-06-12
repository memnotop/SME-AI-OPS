# 后续开发说明

本项目当前以结题交付为主要目标。后续如继续开发，建议遵守以下原则：

## 一、修改代码后必须运行

```bash
cd /home/liumingjian/dachuang/SME-AI-OPS-v1.0
./run_all_checks.sh
```

现场快速检查可运行：

```bash
RUN_BENCH=0 RUN_LLAMA=0 ./run_all_checks.sh
```

## 二、新增 API 要同步更新

若新增或删除 `sme_ai_ops.h` 中的公开 API，需要同步更新：

- `API_TEST_REPORT_MAPPING.md`
- `docs/api_reference.md`
- `sme_ai_library/test_bench/verify_sme_ai_timing.c`
- `sme_ai_library/run_qemu_verify_timing.sh`

## 三、新增 benchmark 要同步更新

若新增高层 benchmark，需要同步更新：

- `sme_ai_library/test_bench/bench_sme_full_timing.c`
- `sme_ai_library/run_qemu_full_bench.sh`
- `API_TEST_REPORT_MAPPING.md`
- `docs/performance_summary.md`

## 四、更新固定文件后刷新校验

```bash
find . \( -path './.git' -o -path './acceptance_reports' -o -path './sme_ai_library/build' -o -path './llama.cpp/logs' -o -path './examples/build' \) -prune -o -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
```

注意：`SHA256SUMS` 不校验可再生成的 build、logs 和 acceptance reports。
