#!/usr/bin/env python3
"""简化版触手手动控制脚本 - 适合快速测试"""

import mujoco
import mujoco.viewer
import numpy as np
import time

def simple_tentacle_control():
    """简单的触手控制演示"""
    
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
    
    # 控制参数 (增强版本)
    baseline_lengths = np.array([0.25, 0.25, 0.25])
    cursor_x, cursor_y = 0.0, 0.0
    control_speed = 0.05  # 增加控制速度
    
    print("\n🎮 简化控制说明:")
    print("   WASD - 控制触手方向")
    print("   SPACE - 重置到中心")
    print("   ESC - 退出")
    print("   程序将显示实时控制状态\n")
    
    # 键盘状态
    keys = {'w': False, 's': False, 'a': False, 'd': False}
    
    def key_callback(key, scancode, action, mods):
        """简化的键盘处理"""
        nonlocal cursor_x, cursor_y, keys
        
        key_down = (action == 1) or (action == 2)  # PRESS or REPEAT
        
        # 更新按键状态
        if key == 87: keys['w'] = key_down      # W
        elif key == 83: keys['s'] = key_down    # S
        elif key == 65: keys['a'] = key_down    # A
        elif key == 68: keys['d'] = key_down    # D
        
        # 功能键
        if action == 1:  # PRESS only
            if key == 32:  # SPACE - 重置
                cursor_x, cursor_y = 0.0, 0.0
                print("🎯 重置到中心")
            elif key == 256:  # ESC - 退出
                print("👋 退出...")
                return False
    
    # 启动可视化
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.user_key_callback = key_callback
            
            print("🎬 开始交互控制...")
            step_count = 0
            
            while viewer.is_running():
                # 根据按键更新光标位置
                if keys['w']: cursor_y += control_speed
                if keys['s']: cursor_y -= control_speed
                if keys['a']: cursor_x -= control_speed
                if keys['d']: cursor_x += control_speed
                
                # 限制范围
                cursor_x = np.clip(cursor_x, -0.8, 0.8)
                cursor_y = np.clip(cursor_y, -0.8, 0.8)
                
                # 简单的腱绳控制转换 (增强效果)
                magnitude = np.sqrt(cursor_x**2 + cursor_y**2)
                if magnitude > 0.01:
                    effect = magnitude * 0.15  # 增加控制强度
                    
                    # 根据方向调整腱绳长度
                    data.ctrl[0] = baseline_lengths[0] - cursor_y * effect      # 上下控制
                    data.ctrl[1] = baseline_lengths[1] - (cursor_x * 0.866 - cursor_y * 0.5) * effect  # 右下
                    data.ctrl[2] = baseline_lengths[2] - (-cursor_x * 0.866 - cursor_y * 0.5) * effect # 左下
                else:
                    data.ctrl[:] = baseline_lengths
                
                # 确保在有效范围内
                data.ctrl[:] = np.clip(data.ctrl, 0.12, 0.34)
                
                # 运行仿真
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # 显示状态信息
                step_count += 1
                if step_count % 30 == 0:  # 每30步显示一次
                    # 获取尖端位置
                    try:
                        tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
                        if tip_site_id >= 0:
                            tip_pos = data.site_xpos[tip_site_id]
                        else:
                            tip_pos = data.site_xpos[-1] if model.nsite > 0 else [0, 0, 0]
                        
                        print(f"\r🎮 光标:[{cursor_x:5.2f},{cursor_y:5.2f}] | "
                              f"尖端:[{tip_pos[0]:5.2f},{tip_pos[1]:5.2f},{tip_pos[2]:5.2f}] | "
                              f"腱绳:[{data.ctrl[0]:.2f},{data.ctrl[1]:.2f},{data.ctrl[2]:.2f}]", end="")
                    except:
                        pass
                
                # 控制帧率
                time.sleep(model.opt.timestep)
                
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        print("💡 提示: 在macOS上使用 'mjpython simple_manual_control.py'")

if __name__ == "__main__":
    simple_tentacle_control()
