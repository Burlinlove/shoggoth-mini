# 组装指南

## 零件清单

### 电子器件
- **1个** Waveshare 串行总线舵机驱动板  
  [Amazon](https://www.amazon.fr/dp/B0CJ6TP3TP)

- **3个** Waveshare 20kg.cm 总线舵机电机（包含安装螺丝）
  [Amazon](https://www.amazon.fr/dp/B0CDC587BQ)

- **1个** 60W 12V 5A 电源适配器  
  [Amazon](https://www.amazon.fr/dp/B001W3UYLY)

- **1根** USB-C 到 USB-C 数据线（用于舵机驱动板）
- **1根** USB Micro 到 USB-C 数据线（用于摄像头）

- **1个** 同步3D立体USB摄像头（3840×1080P，4MP，60fps，130° FOV）  
  [AliExpress](https://www.aliexpress.com/item/1005006353765708.html)

> **注意：** 如果使用不同的摄像头，您可能需要相应调整3D打印零件和物体检测模型。


### 硬件
- **2个** M2×3×3 热嵌件  
  [Amazon](https://www.amazon.fr/dp/B0CS6XJSSL)

- **7个** M2.5×3.5×4 热嵌件  
  [Amazon](https://www.amazon.fr/dp/B0CS6YVJYD)

- **2个** M2 × 6mm 螺丝

- **2个** M2.5 × 10mm 螺丝

- **3个** M2.5 × 6mm 螺丝

- **0.5mm 21kg 钓鱼线**（稍后我们将剪成3段80cm长的线）

### 3D打印材料
- **eTPU-95A 线材**（用于打印SpiRobs）  
  [eSUN 3D](https://esun3d.fr/products/esun-etpu-95a-gris-grey-1-75-mm-1-kg)

- **PLA 线材**（用于刚性结构零件）

## 打印

### 圆顶 - PLA
<div align="center">
<img src="assets/media/assembly/print_dome.png" width="50%">
</div>

[下载STEP文件](assets/hardware/printing/dome.step)

### 底板 - PLA
<div align="center">
<img src="assets/media/assembly/print_plate.png" width="50%">
</div>

[下载STEP文件](assets/hardware/printing/plate.step)

### 滚轮罩 - PLA (3个)
<div align="center">
<img src="assets/media/assembly/print_roller_cover.png" width="50%">
</div>

[下载STEP文件](assets/hardware/printing/roller_cover.step)

### 滚轮 - PLA (3个)
<div align="center">
<img src="assets/media/assembly/print_roller.png" width="50%">
</div>

[下载STEP文件](assets/hardware/printing/roller.step)

### 尖头 - PLA
<div align="center">
<img src="assets/media/assembly/print_spike.png" width="50%">
</div>

[下载STEP文件](assets/hardware/printing/spike.step)

### 加厚SpiRobs触手 - eTPU
<div align="center">
<img src="assets/media/assembly/print_thick_spirobs.png" width="50%">
</div>

[下载3MF文件](assets/hardware/printing/thick_spirobs.3mf) | [下载STEP文件](assets/hardware/printing/thick_spirobs.step)

> **注意：** 这是3线SpiRobs的改进版本，具有更厚的脊柱以增加刚性和稳定性。

## 组装说明

### 底板准备和舵机驱动板安装

使用电烙铁在底板上的驱动板安装位置安装4个M2.5热嵌件。

<div align="center">
<img src="assets/media/assembly/plate_1.jpeg" width="50%">
</div>

使用电烙铁在圆顶固定位置安装3个M2.5热嵌件。

<div align="center">
<img src="assets/media/assembly/plate_2.jpeg" width="50%">
</div>

使用2个M2.5 × 10mm螺丝将舵机驱动板安装到底板上（或者，您可以使用所有四个角安装孔）。如果需要，可以使用PCB支柱来抬高驱动板。

<div align="center">
<img src="assets/media/assembly/plate_3.jpeg" width="50%">
</div>

将驱动板固定到底板上之前安装的嵌件位置。

<div align="center">
<img src="assets/media/assembly/plate_4.jpeg" width="45%" style="display: inline-block; margin: 0 10px;">
<img src="assets/media/assembly/plate_5.jpeg" width="45%" style="display: inline-block; margin: 0 10px;">
</div>

### 摄像头安装

使用电烙铁在圆顶的摄像头固定位置安装2个M2热嵌件。

<div align="center">
<img src="assets/media/assembly/camera_1.jpeg" width="50%">
</div>

将摄像头放置在圆顶内的安装凹槽中，并用2个M2 × 6mm螺丝固定到已安装的嵌件中。

<div align="center">
<img src="assets/media/assembly/camera_2.jpeg" width="50%">
</div>

### 电机组装和安装

拆下每个Waveshare电机顶部的两个螺丝，将电机插入滚轮罩中。

<div align="center">
<img src="assets/media/assembly/motor_1.jpeg" width="50%">
</div>

重新安装螺丝，将电机固定在滚轮罩内。

<div align="center">
<img src="assets/media/assembly/motor_2.jpeg" width="50%">
</div>

将滚轮连接到每个电机组件上。

<div align="center">
<img src="assets/media/assembly/motor_3.jpeg" width="50%">
</div>

对其他两个舵机电机重复此过程。在继续之前，根据下图分配电机ID。使用贴纸标识每个电机。为了与控制软件正确配合，每个电机必须相对于驱动板位置分配正确的ID。驱动板应位于电缆开口附近，与大部分操作发生的摄像头相对。

要使用LeRobot代码库分配电机ID：

1. 将每个电机单独连接到驱动板
2. 将电源线连接到驱动板，USB-C线连接到电脑
3. 查找驱动板的USB端口：
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
python lerobot/scripts/find_motors_bus_port.py
```

4. 配置每个电机ID（对电机1、2、3重复操作）：
```bash
python lerobot/scripts/configure_motor.py \
  --port DRIVER_BOARD_USB_PORT \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```

配置完成后，将每个电机插入底板上的指定插槽。

<div align="center">
<img src="assets/media/assembly/motor_4.jpeg" width="50%">
</div>

使用Waveshare电机附带的螺丝将电机固定到底板上（建议每个电机使用2个螺丝）。

<div align="center">
<img src="assets/media/assembly/motor_5.jpeg" width="50%">
</div>

组装现在应该如下图所示。

<div align="center">
<img src="assets/media/assembly/motor_6.jpeg" width="50%">
</div>

安装钓鱼线滚轮。首先将一个螺丝部分拧入滚轮，以帮助在轮子上定位。磁性螺丝刀在这一步特别有用。

<div align="center">
<img src="assets/media/assembly/motor_7.jpeg" width="50%">
</div>

完全固定滚轮到轮子上。

<div align="center">
<img src="assets/media/assembly/motor_8.jpeg" width="50%">
</div>

确保正确对齐：
- 剩余的滚轮螺丝孔与轮子螺丝孔对齐
- 滚轮中的钓鱼线穿孔与滚轮罩中的穿孔对齐

根据需要使用螺丝刀旋转电机/滚轮组件以实现正确对齐。

<div align="center">
<img src="assets/media/assembly/motor_9.jpeg" width="50%">
</div>

安装钓鱼线（这是最具挑战性的步骤之一）。在每根80cm钓鱼线的一端创建一个小弯曲以便于插入。

<div align="center">
<img src="assets/media/assembly/motor_10.jpeg" width="50%">
</div>

将钓鱼线穿过两个对齐的孔。

<div align="center">
<img src="assets/media/assembly/motor_11.jpeg" width="50%">
</div>

弯曲钓鱼线并将其插入滚轮的下螺丝孔，与已安装的螺丝相对。

<div align="center">
<img src="assets/media/assembly/motor_12.jpeg" width="50%">
</div>

安装另一个Waveshare电机螺丝来固定钓鱼线。确保螺丝完全与电机轮螺纹啮合。

<div align="center">
<img src="assets/media/assembly/motor_13.jpeg" width="50%">
</div>

通过用力拉拽来测试钓鱼线的牢固性。钓鱼线应保持牢固固定。这种方法虽然看起来粗糙，但提供了可靠的钓鱼线固定。

<div align="center">
<img src="assets/media/assembly/motor_14.jpeg" width="50%">
</div>

对其余两根钓鱼线和电机重复此过程。

<div align="center">
<img src="assets/media/assembly/motor_15.jpeg" width="50%">
</div>

按顺序连接电机：电机2连接到电机3，电机3连接到电机1，电机1连接到驱动板。

<div align="center">
<img src="assets/media/assembly/motor_16.jpeg" width="50%">
</div>

### 最终组装

开始最终组装，将三根线缆（摄像头、驱动板数据线和驱动板电源线）穿过圆顶开口。将摄像头线缆连接到摄像头。

<div align="center">
<img src="assets/media/assembly/cables_1.jpeg" width="50%">
</div>

将数据线和电源线连接到驱动板。

<div align="center">
<img src="assets/media/assembly/cables_2.jpeg" width="50%">
</div>

将每根钓鱼线穿过圆顶顶部相应的孔。请特别注意以下几点：
- 钓鱼线2（前腱致动器）必须穿过小隧道以防止拉紧时与摄像头干扰
- 每个钓鱼线ID必须对应正确的圆顶孔
- 将钓鱼线布线在摄像头线缆下方以防止操作期间干扰

放置圆顶并拉紧线缆和钓鱼线。使用底板中的穿孔从下方验证正确的线缆布线。

<div align="center">
<img src="assets/media/assembly/cables_3.jpeg" width="45%" style="display: inline-block; margin: 0 10px;">
<img src="assets/media/assembly/cables_4.jpeg" width="45%" style="display: inline-block; margin: 0 10px;">
</div>
<div align="center">
<img src="assets/media/assembly/cables_5.jpeg" width="50%">
</div>

用3个M2.5 × 6mm螺丝固定圆顶。

<div align="center">
<img src="assets/media/assembly/cables_6.jpeg" width="50%">
</div>

将钓鱼线穿过触手。确保如下图所示的正确孔对齐，然后将每根钓鱼线完全穿过所有孔。为了获得最佳效果，一次处理一根钓鱼线。

<div align="center">
<img src="assets/media/assembly/cables_7.jpeg" width="50%">
</div>

完成后，如图所示拉紧所有钓鱼线。

<div align="center">
<img src="assets/media/assembly/cables_8.jpeg" width="50%">
</div>

在用结固定钓鱼线之前，使用校准脚本去除多余的松弛。80cm的钓鱼线长度超过静止长度要求，这有三个重要原因：

1. **控制范围**：软件需要每个方向大约一整圈（≈11cm）的钓鱼线移动来实现正确的触手关节运动和松弛管理
2. **维护访问**：额外长度允许圆顶检查和内部工作，无需解开结和重复钓鱼线安装过程
3. **重新校准**：提供调整能力以实现最佳静止位置校准

运行校准脚本进行初始预卷绕（稍后我们将微调静止位置）：

```bash
python -m shoggoth_mini calibrate --config shoggoth_mini/configs/default_hardware.yaml
```

使用方向键（左、上、右）来卷绕电机并去除松弛。如需要，按空格键反转卷绕方向。请注意，机器人控制器期望您使用脚本开始时出现的方向进行卷绕。不要在不进行必要软件/配置更改的情况下改变卷绕方向（我认为它在代码中还不是处处可配置的）。

留下大约10cm的钓鱼线长度用于打结。

<div align="center">
<img src="assets/media/assembly/cables_9.jpeg" width="50%">
</div>

为了最佳的钓鱼线固定，避免直接打结。相反，用每根钓鱼线创建一个环，并将其重新穿过最后一个孔。这种方法提供更好的张力分布，并减少操作期间松动的风险。拉紧每根钓鱼线以消除松弛。完成此步骤后，触手应保持直立。

<div align="center">
<img src="assets/media/assembly/cables_10.jpeg" width="45%" style="display: inline-block; margin: 0 10px;">
<img src="assets/media/assembly/cables_11.jpeg" width="45%" style="display: inline-block; margin: 0 10px;">
</div>

打一个牢固的结并用力拉拽测试。修剪多余的钓鱼线长度，至少留下3cm以备需要时解结。

<div align="center">
<img src="assets/media/assembly/cables_12.jpeg" width="50%">
</div>

将剩余的钓鱼线长度塞入尖头触手尖端，并将尖头插入触手的第一段。

### 最终校准

执行最终触手校准以确保所有钓鱼线都处于正确的静止长度。运行校准脚本并使用方向键调整，直到触手直立：

```bash
python -m shoggoth_mini calibrate --config shoggoth_mini/configs/default_hardware.yaml
```

此脚本将校准数据保存到 `assets/hardware/calibration/tentacle_calibration.json`，控制软件稍后会自动加载此文件。

**重要校准注意事项：**
- 避免在校准期间过度放松钓鱼线（除非圆顶也被拉开以保持张力）
- 防止过度拉紧线缆（见下方对比图片）

<table align="center">
<tr>
<td align="center">
<img src="assets/media/assembly/final_1.jpeg" width="400">
<br>
<em>正确的线缆张力</em>
</td>
<td align="center">
<img src="assets/media/assembly/final_2.jpeg" width="400">
<br>
<em>过度拉紧 - 避免这种情况</em>
</td>
</tr>
</table>

正确校准的触手尖端应表现出"鸡头"行为：当您扭转和弯曲触手主体时，尖端保持相对稳定。

<div align="center">
<img src="assets/media/assembly/final_3.gif" width="50%">
</div>

完成！ 