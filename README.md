# BUSI_SSL_Baseline

BUSI 二值病灶分割任务的半监督学习基线仓库。

## Version

当前首个版本定义为 `v0.1.0`。

该版本为首个可运行、可复现的 baseline（基线版本），包含：
- 数据清单生成
- 数据划分生成
- 监督训练流程
- 最小 EMA（指数滑动平均） teacher-student（教师-学生） 半监督训练流程
- 评估、日志与 checkpoint（检查点）保存
- AutoDL 运行 notebook

该版本的目标是建立清晰、稳定、可对照的实验起点。