#!/usr/bin/env python3
"""自动演示触手控制 - 绕过键盘问题"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math

def auto_demo_tentacle_control():
    """自动演示触手运动，无需键盘输入"""
    
    # 加载模型
    model_path = "assets/simulation/tentacle.xml"
    print(f"🚀 加载模型: {model_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 控制参数
    baseline_lengths = np.array([0.25, 0.25, 0.25])
    
    print("\n🎬 自动演示模式:")
    print("   ⏯️  触手将自动执行各种运动模式")
    print("   👁️  观察触手的弯曲和运动")
    print("   ⏱️  每个动作持续3秒")
    print("   🔄  总演示时长约30秒")
    print("   ESC - 可以随时退出\n")
    
    # 预定义的运动模式
    movement_patterns = [
        {"name": "向上弯曲", "cursor": (0.0, 0.6), "duration": 3},
        {"name": "向右下弯曲", "cursor": (0.5, -0.4), "duration": 3},
        {"name": "向左下弯曲", "cursor": (-0.5, -0.4), "duration": 3},
        {"name": "向右弯曲", "cursor": (0.7, 0.0), "duration": 3},
        {"name": "向左弯曲", "cursor": (-0.7, 0.0), "duration": 3},
        {"name": "圆周运动", "cursor": "circle", "duration": 5},
        {"name": "波浪运动", "cursor": "wave", "duration": 5},
        {"name": "回到中心", "cursor": (0.0, 0.0), "duration": 2},
    ]
    
    def convert_cursor_to_tendon_lengths(cursor_x, cursor_y):
        """将2D光标转换为腱绳长度"""
        magnitude = np.sqrt(cursor_x**2 + cursor_y**2)
        if magnitude > 0.01:
            effect = magnitude * 0.3  # 强控制效果
            
            # 根据方向调整腱绳长度
            tendon_lengths = baseline_lengths.copy()
            tendon_lengths[0] -= cursor_y * effect      # 上下控制
            tendon_lengths[1] -= (cursor_x * 0.866 - cursor_y * 0.5) * effect  # 右下
            tendon_lengths[2] -= (-cursor_x * 0.866 - cursor_y * 0.5) * effect # 左下
            
            return np.clip(tendon_lengths, 0.12, 0.34)
        else:
            return baseline_lengths
    
    def get_tip_position():
        """获取触手尖端位置"""
        try:
            tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
            if tip_site_id >= 0:
                return data.site_xpos[tip_site_id].copy()
            else:
                return data.site_xpos[-1].copy() if model.nsite > 0 else np.zeros(3)
        except:
            return np.zeros(3)
    
    # 启动可视化
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("🎬 开始自动演示...")
            
            total_start_time = time.time()
            current_pattern_index = 0
            pattern_start_time = time.time()
            
            while viewer.is_running() and current_pattern_index < len(movement_patterns):
                current_time = time.time()
                pattern = movement_patterns[current_pattern_index]
                elapsed_in_pattern = current_time - pattern_start_time
                
                # 计算当前光标位置
                if pattern["cursor"] == "circle":
                    # 圆周运动
                    angle = (elapsed_in_pattern / pattern["duration"]) * 2 * math.pi
                    cursor_x = 0.5 * math.cos(angle)
                    cursor_y = 0.5 * math.sin(angle)
                elif pattern["cursor"] == "wave":
                    # 波浪运动
                    cursor_x = 0.6 * math.sin(elapsed_in_pattern * 2)
                    cursor_y = 0.4 * math.cos(elapsed_in_pattern * 3)
                else:
                    # 固定位置
                    target_x, target_y = pattern["cursor"]
                    # 平滑过渡
                    progress = min(elapsed_in_pattern / 1.0, 1.0)  # 1秒过渡时间
                    cursor_x = target_x * progress
                    cursor_y = target_y * progress
                
                # 转换为腱绳长度
                tendon_lengths = convert_cursor_to_tendon_lengths(cursor_x, cursor_y)
                data.ctrl[:] = tendon_lengths
                
                # 运行仿真
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # 显示状态信息
                tip_pos = get_tip_position()
                print(f"\r🎯 {pattern['name']}: 光标[{cursor_x:5.2f},{cursor_y:5.2f}] | "
                      f"尖端[{tip_pos[0]:5.2f},{tip_pos[1]:5.2f},{tip_pos[2]:5.2f}] | "
                      f"腱绳[{data.ctrl[0]:.2f},{data.ctrl[1]:.2f},{data.ctrl[2]:.2f}] | "
                      f"进度{elapsed_in_pattern:.1f}s/{pattern['duration']}s", end="")
                
                # 检查是否该切换到下一个模式
                if elapsed_in_pattern >= pattern["duration"]:
                    current_pattern_index += 1
                    pattern_start_time = current_time
                    print()  # 换行
                    if current_pattern_index < len(movement_patterns):
                        next_pattern = movement_patterns[current_pattern_index]
                        print(f"\n🔄 切换到: {next_pattern['name']}")
                
                # 控制帧率
                time.sleep(model.opt.timestep)
            
            print(f"\n✅ 自动演示完成! 总时长: {time.time() - total_start_time:.1f}秒")
            print("   你应该看到了触手的各种弯曲和运动！")
            
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        print("💡 提示: 在macOS上使用 'mjpython auto_demo_control.py'")

if __name__ == "__main__":
    auto_demo_tentacle_control()
