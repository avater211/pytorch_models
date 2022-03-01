## 1、介绍
本工具用于统计模型中运行的所有信息，并上传到superset网页展示。

***模型的信息包括：***
- 软件信息：pytorch框架及其依赖库的版本信息
- 硬件信息：mlu型号，cpu型号
- 模型超参：batch_size，opt_level，extra_params
- 性能数据：latency_stats，throughput

***superset网页链接***
> http://dataview.cambricon.com/superset/dashboard/18/

> http://dataview.cambricon.com/superset/dashboard/14/

## 2、运行步骤
### 2.1 环境配置
> pip install git+http://gitlab.software.cambricon.com/liangfan/cndb

### 2.2 运行命令

启动展示性能数据的脚本

> bash launch.sh 0

启动展示精度数据的脚本

> bash launch.sh 1

