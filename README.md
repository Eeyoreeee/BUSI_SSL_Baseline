# BUSI_SSL_Baseline

BUSI 乳腺超声二值病灶分割任务的半监督学习基线仓库。

本项目当前重点不是直接堆复杂方法，而是先建立一个可运行、可复现、便于后续扩展和对比的实验基线，并在此基础上逐步增强训练能力。

## 当前状态

当前仓库已经历以下阶段：

- `v0.1.0`：首个可运行、可复现的 baseline（基线版本）
- `v0.1.1`：文档与仓库整理，不改核心实验定义
- 当前代码主线：在 baseline 基础上加入学习率调度器与低标注监督实验配置，建议整理后发布为 `v0.2.0`

## 项目目标

本仓库当前阶段的目标是建立一个：

- 结构清晰
- 训练流程可复现
- 便于比较实验改动
- 方便后续扩展 SSL（半监督学习）策略

的研究基线。

## 当前已支持内容

当前代码已经覆盖：

- 数据清单生成
- 固定数据划分生成
- supervised（监督学习）训练流程
- minimal SSL（最小半监督学习）训练流程
- EMA（指数滑动平均）teacher-student（教师-学生）伪标签训练框架
- evaluation（评估）、logging（日志）与 checkpoint（检查点）保存
- warmup + cosine 学习率调度策略
- 低标注比例监督实验配置，如 `1/8` 标注子集
- AutoDL 运行说明 notebook

## Repository Structure

当前仓库仍保留 AutoDL 环境下的部分目录结构，后续版本会继续整理。当前大致结构如下：

```text
.
├─ README.md
├─ BUSI_SSL_Baseline_AutoDL_Annotated_Runbook.ipynb
├─ autodl-fs/
│  └─ busi_ssl_baseline/
│     └─ busi_ssl_baseline/
│        ├─ busi_seg/
│        │  ├─ analysis/
│        │  ├─ data/
│        │  ├─ engine/
│        │  ├─ logging/
│        │  ├─ losses/
│        │  ├─ models/
│        │  ├─ ssl/
│        │  └─ utils/
│        ├─ configs/
│        ├─ data_meta/
│        ├─ tools/
│        ├─ train_sup.py
│        └─ train_ssl.py
```

## Supported Experiments

当前版本支持的实验范围主要包括：

- BUSI 二值病灶分割
- DeepLabV3Plus + ResNet50 基线模型
- 全监督训练
- 部分标注场景下的最小半监督训练
- 不同比例标注子集划分，如 `1/2`、`1/4`、`1/8`
- 基于 warmup + cosine 的训练日程控制

## Quick Start

### 1. Install dependencies

```bash
pip install -r autodl-fs/busi_ssl_baseline/busi_ssl_baseline/requirements.txt
```

### 2. Prepare metadata

```bash
python autodl-fs/busi_ssl_baseline/busi_ssl_baseline/tools/prepare_busi_manifest.py
python autodl-fs/busi_ssl_baseline/busi_ssl_baseline/tools/make_splits.py
```

### 3. Run supervised training

```bash
python autodl-fs/busi_ssl_baseline/busi_ssl_baseline/train_sup.py --config autodl-fs/busi_ssl_baseline/busi_ssl_baseline/configs/experiments/sup_full.yaml
```

### 4. Run low-label supervised training

```bash
python autodl-fs/busi_ssl_baseline/busi_ssl_baseline/train_sup.py --config autodl-fs/busi_ssl_baseline/busi_ssl_baseline/configs/experiments/sup_subset_1of8.yaml
```

### 5. Run SSL training

```bash
python autodl-fs/busi_ssl_baseline/busi_ssl_baseline/train_ssl.py --config autodl-fs/busi_ssl_baseline/busi_ssl_baseline/configs/base_ssl.yaml
```

## Versioning Plan

本项目当前采用渐进式版本管理：

- `v0.1.x`：文档、路径、结构、复现性与工程整理
- `v0.2.x`：新增实验配置、训练策略与基础方法扩展
- `v0.3.x`：引入更明确的方法增强与系统化对比实验
- `v1.0.0`：形成较完整、稳定的主实验框架

## Recommended Next Release

建议将当前主线整理后发布为：

### `v0.2.0`

建议定义为：

> 在 `v0.1.1` 基础上，引入 epoch-level（按 epoch 生效）的 warmup + cosine 学习率调度器，并补充低标注监督实验配置，作为从“可运行基线”过渡到“实验能力增强版”的首个版本。

## Roadmap

下一步建议包括：

- 整理仓库目录结构，去除环境路径痕迹
- 合并重复 README
- 统一配置中的路径写法
- 清理 notebook 输出与冗余文件
- 补充 `CHANGELOG.md`
- 将当前主线整理并发布为 `v0.2.0`
- 扩展更丰富的 SSL 对比实验

## Notes

当前版本仍然更强调工程基线的建立与实验可复现性，而不是最终方法性能的定稿。后续版本将逐步引入更系统的实验对照与方法增强。

## License

待补充。