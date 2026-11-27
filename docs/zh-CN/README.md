# linker-sim 快速入门

linker-sim 在普通电脑的 CPU 上用 MuJoCo 4 仿真一台 Linker 工作站 LW-v0（两台 AR5 机械臂、两只 O6 灵巧手、桌子和相机），不需要 GPU。它的机器人模型文件、驱动模型、控制器限位和物理参数都与公司的 GPU 仿真器 linker-lab 相同，行为与真机一致。

linker-sim 首先是一个 Python 库，用来在上面开发课程和产品。它自带两个基于这个库的示例：手动操作机器人的 **playground**，以及运行预训练策略的 **gallery**。

本页是中文快速入门，内容与英文文档对应：安装、检查、playground、gallery、第一节课。完整的接口说明见英文的 [API 参考](../api.md)，两个查看器的区别见 [viewers](../viewers.md)，电脑要求见 [requirements](../requirements.md)。英文文档是准确的版本；两者不一致时以英文为准。

## 1. 安装

需要：搭载 Apple Silicon 的 Mac，或 x86_64 / ARM64 的 Linux 电脑；约 2 GB 磁盘空间；一个浏览器。仿真本身不需要 GPU。不支持 Intel 芯片的 Mac（MuJoCo 4 没有对应版本），本版本也不支持 Windows。

```bash
git clone https://github.com/linker-bot/linker-sim
cd linker-sim
./install.sh
```

`install.sh` 会在需要时把 uv（Python 包管理器）安装到 `~/.local/bin`，再把 Python 3.12 和所有依赖按 `uv.lock` 中锁定的版本安装到本目录的 `.venv` 中。除了 uv 及其缓存，不会改动本目录以外的任何东西。最后它会运行 `linker-sim doctor`（见下一节）。

如果安装程序安装了 uv，当前终端还找不到它（`uv: command not found`）：打开一个新终端，或运行 `source ~/.local/bin/env`。

LW-v0 的机器人模型文件随安装一起装好：`uv.lock` 把它们锁定为来自 [linker-sim-assets](https://github.com/linker-bot/linker-sim-assets) 的软件包 `linker-lw-v0`。

## 2. 检查

```bash
uv run linker-sim doctor
```

它检查已安装的版本，在 CPU 上运行 Warp 和 torch，并仿真 LW-v0 一秒钟。最后一项检查类似于：

```text
[ok] simulation: LW-v0 1.0.0, MuJoCo 4.0.1: home pose held to 9.7e-07 rad; 1 s simulated at 15x real time on one thread
All checks passed.
```

其中的实时倍数（这里是 15×）表示这台电脑仿真工作站比真实时间快多少倍。大于约 4× 时，所有功能应该都能以实时速度运行（在更慢的电脑上测量之前，这只是估计）。

所有命令都通过 `uv run` 在本目录的环境中运行；也可以在每个终端中先执行一次 `source .venv/bin/activate`，之后省去 `uv run`。

## 3. Playground：手动操作机器人

```bash
uv run linker-sim play
```

它会打印一个地址，例如 `http://127.0.0.1:8080/`。用浏览器打开，几秒后出现工作站，右手下方的桌上有一个 5 cm 的方块。终端中 Studio 自己的消息（"Model changed: the browser page reloads automatically."、浏览器连接的日志）是正常的。**先在场景中点击一下**，让页面接收键盘输入。然后：

| 按键 | 作用 |
| --- | --- |
| I / K、J / L、U / O | 把当前机械臂（开始时是右臂）的手掌目标向前 / 后、左 / 右、上 / 下移动（相对于你的视角） |
| Shift + 上述按键 | 改为旋转目标 |
| Tab | 切换机械臂 |
| 1 / 2 / 3 / 4 | 手：张开 / 握紧 / 捏取 / 拇指对掌 |
| R | 复位 |
| 鼠标 | 旋转和移动视角 |

机械臂像真机一样在真实控制器的限位内跟随目标。面板显示机械臂的工具位姿、手掌离目标还有多远、限位介入的次数、接触和仿真速度。在终端按 Ctrl+C 停止。

另一个查看器中的同一个 playground：

```bash
uv run linker-sim play --viewer viser
```

在 viser 中每按一次键，目标移动一小步（按住会重复），还可以用鼠标拖动目标的手柄。两个查看器的区别见 [viewers](../viewers.md)。

**在远程 Linux 电脑上工作？** 页面只对那台电脑本身开放。用 SSH 转发端口：`ssh -L 8080:127.0.0.1:8080 <电脑>`，然后在你自己的电脑上打开 `http://127.0.0.1:8080/`。

## 4. Gallery：预训练策略

```bash
uv run linker-sim gallery list
uv run linker-sim gallery info ycb-grasp
uv run linker-sim gallery run ycb-grasp            # 或加 --viewer viser
```

在 `ycb-grasp` 中，从面板的列表里选择 31 个物体之一，按 **Drop**：物体落在桌面左半部分的随机位置；停稳后按 **Grasp**。左臂和左手把它抓起并举高 15 cm，面板显示是否抓取成功。**Reset** 清空桌面。

**它的权重仅限研究用途（非商业用途）。** `gallery info` 显示各项许可。可以随产品发布的重新训练的策略将在后续版本中提供。

## 5. 第一个脚本

把下面的代码保存为本目录下的 `first.py`，用 `uv run python first.py` 运行：

```python
import numpy as np

from linker_sim import hand
from linker_sim.workstation import Station, arm_action_cfg

station = Station(arm_left=arm_action_cfg("left", command="pose", frame="palm"))
pos, quat = station.palm_pose("left")  # 手掌点，工作站坐标系
goal = np.concatenate(station.to_base("left", pos + (0, 0, 0.05), quat))  # 向上 5 cm
for _ in range(60):  # 仿真一秒：60 个决策步
    station.step(arm_left=goal, hand_left=hand.PRESETS["close"])
print("手掌上移了", station.palm_pose("left")[0][2] - pos[2], "m")
```

几个基本约定：
- 单位是米、弧度、秒、千克。四元数的顺序是 x, y, z, w。
- **工作站坐标系**：x 指向桌子前方，z 向上，桌面在原点下方 0.5 m。
- **手掌点**：手掌前方 3.5 cm 处的一个点。
- **位姿指令在各自机械臂的基座坐标系中给出**，与真实控制器一致，所以要先用 `to_base` 转换。
- 每次 `step()` 是一个**决策步**（1/60 s）。指令是目标而不是瞬移：机械臂在真实的速度限位内向目标运动，手的电机也按自己的速度闭合。

## 6. 第一节课：抓起一个盒子

[`examples/pick_up_a_box.py`](../../examples/pick_up_a_box.py) 是一节基于 API 的小课程，英文的 [lesson](../lesson.md) 一步一步地构建它。学生把自己的一个盒子交给 gallery 中的抓取策略（`ycb-grasp`），用面板上的按钮（或键 1–3）运行：把盒子放到桌上、抓取、复位。面板显示场景的状态、手掌点到盒子的距离以及盒子的高度。

```bash
uv run python examples/pick_up_a_box.py                  # MuJoCo Studio：打开它打印的地址
uv run python examples/pick_up_a_box.py --viewer viser   # 在 viser 中运行同一节课
uv run python examples/pick_up_a_box.py --check          # 不开查看器：放下、抓取并检查是否抬起
```

它展示了课程最常用的几部分 API：
- 按策略的方式描述一个物体：网格、质量（`objects.mass_properties`）、表面上的 512 个点、包围盒和放置姿态。
- 在自己的代码中使用 gallery 条目的策略和场景：`gallery.load`、`gallery.fetch`。
- 读取手掌点和物体的位姿：`palm_pose`、`pose()`。
- 一个在两个查看器中都能运行的 `Panel`：文字、按钮、标记。

为什么用策略而不是让手按固定脚本闭合？O6 的电机闭合较慢，且只能用真实的力量，因此在固定位置简单地闭合手掌很少能握住物体：手掌去哪里、每根手指闭合多少都要适应物体。gallery 的策略正是为此在数千个物体上训练的。

面板只需写一次：

```python
from linker_sim import viewer
from linker_sim.viewer import Panel
from linker_sim.workstation import Station

station = Station()
panel = Panel("我的课程")
panel.text(lambda: f"仿真时间：{station.time:.1f} s")
panel.button("复位", station.reset, key="r")
# viewer.run(station, panel)                  # MuJoCo Studio
# viewer.run(station, panel, viewer="viser")  # 同一个面板在 viser 中
```

`viewer.run` 以实时速度推进仿真，每个决策步调用一次 `step()`。所有回调（按钮、文字、`step`）都与仿真在同一个线程中、在两个决策步之间运行，因此回调可以直接读取和控制工作站，无需加锁。

**可以读取 MuJoCo 的数据，但不支持写入。** `station.model` 和 `station.data` 是 MuJoCo 的 `MjModel` 和 `MjData`，API 中没有的信息可以从中读取；写入会绕过驱动模型，其效果在任何版本中都可能改变。

## 遇到问题

| 现象 | 处理 |
| --- | --- |
| `install.sh: linker-sim needs a Mac with Apple Silicon` | 不支持 Intel 芯片的 Mac |
| `glibc ... is too old` | 使用更新的 Linux：x86_64 上 Ubuntu 20.04 及以上，ARM64 上 22.04 及以上 |
| 下载返回 `404` | release 是私有的：按第 1 节手动下载 |
| `doctor` 仿真检查失败或低于 1× | 电脑太慢或太忙：关闭其他程序后重试，并反馈结果 |
| 页面一直空白，或 Studio 显示 "Connecting…" | 最多等一分钟（Studio 要下载整个模型，约 290 MB）；或改用 `--viewer viser` |
| 页面很卡，Studio 的按键延迟数秒 | 浏览器没有用 GPU 绘制：查看 `chrome://gpu`，或改用 `--viewer viser` |
| 按键没有反应 | 先在场景中点击一下。Studio 中可能有另一个标签页占用了面板：关闭它，或点 "Steal control" |
| 指定 `--port N` 后 Studio 提示 `Port N is already in use` | 换一个端口，或去掉 `--port`，自动使用从 8080 开始的第一个空闲端口。viser 会自动换到下一个空闲端口，请看它打印的地址 |

反馈问题时，请附上 `uv run linker-sim doctor` 的输出。测试一台电脑时要记录哪些信息，见 [requirements](../requirements.md#testing-a-computer)。
