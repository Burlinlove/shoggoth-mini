# Shoggoth Mini
![系统演示](assets/media/system.gif)

Shoggoth Mini 是一个软体触手机器人，使用 [SpiRobs](https://arxiv.org/pdf/2303.09861) 设计，通过强化学习和 GPT-4o 的组合进行控制。阅读完整的博客文章请点击[这里](https://www.matthieulc.com/posts/shoggoth-mini/)。

## 快速开始

### 1. 安装

首先，确保您已安装 Python 3.10+ 和 [Poetry](https://python-poetry.org/docs/#installation)。

克隆项目仓库并安装依赖项：
```bash
git clone https://github.com/mlecauchois/shoggoth-mini
cd shoggoth-mini
poetry install
```

接下来，`lerobot` 是一个关键依赖项，需要从源代码安装：
```bash
git clone https://github.com/huggingface/lerobot.git
pip install -e ./lerobot
```

最后，激活虚拟环境以运行后续命令：
```bash
eval "$(poetry env activate)"
```

### 2. 硬件配置

使用 `lerobot` 中的脚本查找驱动板的 USB 端口：
```bash
python lerobot/scripts/find_motors_bus_port.py
```

获得端口后，配置每个电机（ID 为 1、2 和 3），将 `DRIVER_BOARD_USB_PORT` 替换为您找到的端口：
```bash
python lerobot/scripts/configure_motor.py \
  --port DRIVER_BOARD_USB_PORT \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```

### 3. 电机校准

通过使用方向键调整电机直到触手尖端笔直，然后按 Enter 键保存来校准电机：
```bash
python -m shoggoth_mini calibrate --config shoggoth_mini/configs/default_hardware.yaml
```

### 4. 运行编排器

运行主编排器应用程序，它集成了所有系统组件：
```bash
python -m shoggoth_mini orchestrate \
  --config shoggoth_mini/configs/default_orchestrator.yaml \
  --hardware-config shoggoth_mini/configs/default_hardware.yaml \
  --perception-config shoggoth_mini/configs/default_perception.yaml \
  --control-config shoggoth_mini/configs/default_control.yaml
```

## 硬件复制

要完整复制和设置机器人，请按照 [ASSEMBLY.md](ASSEMBLY.md) 中的步骤进行。所有 3D 打印资源都包含在仓库中。总成本应少于 200 美元。

## 软件复制

### 1. 硬件设置

测试电机连接：
```bash
python -m shoggoth_mini primitive "<yes>" --config shoggoth_mini/configs/default_hardware.yaml
```

校准电机：
```bash
python -m shoggoth_mini calibrate --config shoggoth_mini/configs/default_hardware.yaml
```

### 2. 手动控制

使用触控板控制：
```bash
python -m shoggoth_mini trackpad --config shoggoth_mini/configs/default_hardware.yaml
```

用呼吸模式测试空闲运动：
```bash
python -m shoggoth_mini idle --duration 10 \
  --hardware-config shoggoth_mini/configs/default_hardware.yaml \
  --control-config shoggoth_mini/configs/default_control.yaml
```

### 3. 强化学习管道

生成 MuJoCo XML 模型：
```bash
python -m shoggoth_mini generate-xml --output-path assets/simulation/tentacle.xml
```

训练强化学习模型：
```bash
python -m shoggoth_mini rl train --config shoggoth_mini/configs/default_rl_training.yaml
```

使用 Tensorboard 监控：
```bash
tensorboard --logdir=./
```

在仿真中评估强化学习模型：
```bash
mjpython -m shoggoth_mini rl evaluate ./results/ppo_tentacle_XXXXXXX/models/best_model.zip --config shoggoth_mini/configs/default_rl_training.yaml --num-episodes 10 --render
```

### 4. 视觉管道

记录用于立体三角测量的校准图像。调整暂停间隔以有时间改变图案方向：
```bash
python -m shoggoth_mini record stereo-calibration --num-pairs 20 --interval 3
```

使用您刚刚记录的图像计算立体三角测量校准参数，按照 `notebooks/3d_triangulation.ipynb` 下的 DeepLabCut 笔记本中的步骤进行。

记录标注视频。此命令将记录 6 对（每个摄像头一个）10 秒视频：
```bash
python -m shoggoth_mini record annotation --duration 60 --chunk-duration 10
```

使用 k-means 提取代表性帧：
```bash
python -m shoggoth_mini extract-frames video.mp4 output_frames/ 100
```

我使用 [roboflow](https://roboflow.com/) 标注这些图像，它有很好的自动标注功能，可以避免在高置信度图像上浪费时间。

生成合成训练数据（我使用 [Segment Anything 演示](https://segment-anything.com/demo) 提取触手尖端）：
```bash
# 使用默认值的基本用法
python -m shoggoth_mini synthetic-images assets/synthetic/objects assets/synthetic/backgrounds --num-images 1000
```

在合成图像上训练视觉模型：
```bash
python -m shoggoth_mini vision train dataset.yaml --config shoggoth_mini/configs/default_vision_training.yaml
```

然后在视觉训练配置中更改 `base_model` 以指向最佳模型检查点，并继续在真实图像上训练。

评估训练的模型：
```bash
python -m shoggoth_mini vision evaluate model.pt dataset.yaml --config shoggoth_mini/configs/default_vision_training.yaml
```

对单个图像进行推理：
```bash
python -m shoggoth_mini vision predict model.pt image.jpg --output prediction.jpg --confidence 0.5 --config shoggoth_mini/configs/default_vision_training.yaml
```

调试立体视觉和三角测量：
```bash
python -m shoggoth_mini debug-perception --config shoggoth_mini/configs/default_perception.yaml
```

### 5. 完整系统

在真实机器人上运行闭环强化学习模型和视觉模型：
```bash
python -m shoggoth_mini.control.closed_loop \
  --control-config shoggoth_mini/configs/default_control.yaml \
  --perception-config shoggoth_mini/configs/default_perception.yaml \
  --hardware-config shoggoth_mini/configs/default_hardware.yaml
```

运行完整的编排器：
```bash
python -m shoggoth_mini orchestrate \
  --config shoggoth_mini/configs/default_orchestrator.yaml \
  --hardware-config shoggoth_mini/configs/default_hardware.yaml \
  --perception-config shoggoth_mini/configs/default_perception.yaml \
  --control-config shoggoth_mini/configs/default_control.yaml
```

## 已知问题

- 控制有时会因未知原因导致电机进入无限滚动/展开状态。在这种情况下，对我有效的方法是通过将电缆展开到最大值，然后重新卷起来进行重置。这有时需要打开机器人来解开电线。我还没有时间修复这个问题，如果您解决了，请提交 PR！
- `orchestrator/orchestrator.py`、`control/closed_loop.py` 和 `control/idle.py` 是重度随意编码的，只进行了轻微重构。
- 使用 `control/closed_loop.py` 进行推理会导致 Y 轴上的跟踪偏移，与仿真相比。

## 可能的扩展

- 增加 GPT4o 层的鲁棒性或从头开始训练（例如 [Moshi-like](https://kyutai.org/assets/pdfs/Moshi.pdf)）
- 给它一个声音（但要尽可能非人类！）
- 训练更多强化学习策略（例如抓取和持有复杂物体）
- 使用直驱电机减少噪音
- 添加更多触手并让它爬行
- 抛弃 2D 投影控制以解锁更具表现力的策略

## 引用

```bibtex
@misc{lecauchois2025shoggothmini,
  author = {Le Cauchois, Matthieu B.},
  title = {Shoggoth Mini: Expressive and Functional Control of a Soft Tentacle Robot},
  howpublished = "\url{https://github.com/mlecauchois/shoggoth-mini}",
  year = {2025}
}
``` 