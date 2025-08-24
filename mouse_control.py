#!/usr/bin/env python3
"""鼠标控制触手脚本 - 通过鼠标位置控制"""

import mujoco
import mujoco.viewer
import numpy as np
import time

def mouse_tentacle_control():
    """通过鼠标位置控制触手"""
    
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
    mouse_x, mouse_y = 0.0, 0.0
    window_center_x, window_center_y = 400, 300  # 假设的窗口中心
    control_sensitivity = 0.003  # 鼠标灵敏度
    
    print("\n🖱️ 鼠标控制说明:")
    print("   🎯 移动鼠标控制触手方向")
    print("   📍 窗口中心 = 触手中性位置")
    print("   ⬆️⬇️ 鼠标上下 = 触手上下弯曲")
    print("   ⬅️➡️ 鼠标左右 = 触手左右弯曲")
    print("   🔄 实时响应鼠标位置")
    print("   ESC - 退出\n")
    
    def mouse_callback(button, action, x, y):
        """鼠标回调函数"""
        nonlocal mouse_x, mouse_y
        
        # 将屏幕坐标转换为控制坐标
        # x, y 是鼠标在窗口中的像素位置
        relative_x = (x - window_center_x) * control_sensitivity
        relative_y = (window_center_y - y) * control_sensitivity  # Y轴翻转
        
        mouse_x = np.clip(relative_x, -0.8, 0.8)
        mouse_y = np.clip(relative_y, -0.8, 0.8)
        
        # 显示鼠标位置（每10次更新显示一次，避免刷屏）
        if abs(mouse_x) > 0.05 or abs(mouse_y) > 0.05:
            print(f"🖱️ 鼠标控制: 像素({x},{y}) → 光标[{mouse_x:.3f},{mouse_y:.3f}]")
    
    def convert_cursor_to_tendon_lengths(cursor_x, cursor_y):
        """将2D光标转换为腱绳长度"""
        magnitude = np.sqrt(cursor_x**2 + cursor_y**2)
        if magnitude > 0.01:
            effect = magnitude * 0.25  # 控制效果强度
            
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
            # 设置鼠标回调
            viewer.user_mouse_button_callback = mouse_callback
            
            print("🎬 开始鼠标控制...")
            print("💡 在MuJoCo窗口内移动鼠标来控制触手")
            
            step_count = 0
            last_update_time = time.time()
            
            while viewer.is_running():
                # 转换鼠标位置为腱绳长度
                tendon_lengths = convert_cursor_to_tendon_lengths(mouse_x, mouse_y)
                data.ctrl[:] = tendon_lengths
                
                # 运行仿真
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # 显示状态信息
                step_count += 1
                current_time = time.time()
                if current_time - last_update_time > 1.0:  # 每秒更新一次状态
                    tip_pos = get_tip_position()
                    print(f"📊 状态: 鼠标控制[{mouse_x:5.2f},{mouse_y:5.2f}] | "
                          f"尖端[{tip_pos[0]:5.2f},{tip_pos[1]:5.2f},{tip_pos[2]:5.2f}] | "
                          f"腱绳[{data.ctrl[0]:.2f},{data.ctrl[1]:.2f},{data.ctrl[2]:.2f}]")
                    last_update_time = current_time
                
                # 控制帧率
                time.sleep(model.opt.timestep)
                
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        print("💡 提示: 在macOS上使用 'mjpython mouse_control.py'")

if __name__ == "__main__":
    mouse_tentacle_control()
