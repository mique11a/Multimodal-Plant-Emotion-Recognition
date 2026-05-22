# analysis_outputs

该目录只保留最终可展示、可复核的测试结果和图表。

## 文件说明

- `pressure_dashboard.png`: 压测总览图，适合直接汇报
- `pressure_confusion_heatmap.png`: 窗口级混淆矩阵图
- `pressure_dashboard.svg`: 总览图的矢量版
- `pressure_report.md`: 压测结论文字版
- `pressure_summary.json`: 聚合后的核心统计
- `pressure_summary.csv`: 聚合统计表
- `pressure_file_metrics.csv`: 文件级压测明细
- `local_smoke/`: 本地推理 smoke case

## 当前结论

- 文件级准确率: 三类均为 `100%`
- 窗口级纯度:
  - `Normal`: `100.00%`
  - `Touch`: `83.23%`
  - `Light`: `98.99%`
- 主要混淆:
  - `Touch -> Normal`
  - `Light -> Touch`

## 关联脚本

- 单文件推理: `python test.py --csv analysis_outputs/local_smoke/raw_test_touch_000.csv`
- 批量压测: `python test_res.py`
