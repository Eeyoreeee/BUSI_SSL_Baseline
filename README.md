# BUSI_SSL_Baseline

BUSI 乳腺超声二值病灶分割任务的半监督学习基线仓库。

本项目当前重点不是直接堆复杂方法，而是先建立一个可运行、可复现、便于后续扩展和对比的实验基线。

## 当前版本

当前版本为 **v0.1.0**。

该版本定义为：首个可运行、可复现的 baseline（基线版本），完成了从数据准备、数据划分、监督训练、最小半监督训练，到评估与运行说明的基本闭环，作为后续所有实验改动的对照起点。

目前已包含：

- 数据清单生成
- 固定数据划分生成
- supervised（监督学习）训练流程
- minimal SSL（最小半监督学习）训练流程
- EMA（指数滑动平均）teacher-student（教师-学生）伪标签训练框架
- evaluation（评估）、logging（日志）与 checkpoint（检查点）保存
- AutoDL 运行说明 notebook

## 项目目标

本仓库当前阶段的目标是建立一个：

- 结构清晰
- 训练流程可复现
- 便于比较实验改动
- 方便后续扩展 SSL 策略

的研究基线。

## Repository Structure

当前仓库仍保留 AutoDL 环境下的目录结构，后续版本会继续整理。当前大致结构如下：

~~~text
.
├─ BUSI_SSL_Baseline_AutoDL_Annotated_Runbook.ipynb
├─ README.md
├─ autodl-fs/
│  └─ busi_ssl_baseline/
│     └─ busi_ssl_baseline/
│        ├─ busi_seg/
│        ├─ configs/
│        ├─ data_meta/
│        ├─ tools/
│        ├─ train_sup.py
│        └─ train_ssl.py
~~~

## Supported Experiments

当前版本支持的实验范围主要包括：

- BUSI 二值病灶分割
- DeepLabV3Plus + ResNet50 基线模型
- 全监督训练
- 部分标注场景下的最小半监督训练
- 不同比例标注子集划分，如 1/2、1/4、1/8

## Quick Start

### 1. Install dependencies

~~~bash
pip install -r autodl-fs/busi_ssl_baseline/busi_ssl_baseline/requirements.txt
~~~

### 2. Prepare metadata

~~~bash
python autodl-fs/busi_ssl_baseline/busi_ssl_baseline/tools/prepare_busi_manifest.py
python autodl-fs/busi_ssl_baseline/busi_ssl_baseline/tools/make_splits.py
~~~

### 3. Run supervised training

~~~bash
python autodl-fs/busi_ssl_baseline/busi_ssl_baseline/train_sup.py --config autodl-fs/busi_ssl_baseline/busi_ssl_baseline/configs/experiments/sup_full.yaml
~~~

### 4. Run SSL training

~~~bash
python autodl-fs/busi_ssl_baseline/busi_ssl_baseline/train_ssl.py --config autodl-fs/busi_ssl_baseline/busi_ssl_baseline/configs/experiments/ssl_tau095.yaml
~~~

## Versioning Plan

本项目当前采用渐进式版本管理：

- v0.1.x：文档、路径、结构、复现性与工程整理
- v0.2.x：新增实验配置、策略分支与基础方法扩展
- v0.3.x：引入更明确的方法增强与系统化对比实验
- v1.0.0：形成较完整、稳定的主实验框架

## Roadmap

下一步计划包括：

- 整理仓库目录结构，去除环境路径痕迹
- 合并重复 README
- 统一配置中的路径写法
- 清理 notebook 输出与冗余文件
- 增加更清晰的实验记录与版本说明
- 扩展更丰富的 SSL 对比实验

## Notes

当前版本更强调工程基线的建立，而不是最终方法性能的定稿。后续版本将逐步引入更系统的实验对照与方法增强。

## License

待补充。
