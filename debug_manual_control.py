#!/usr/bin/env python3
"""调试版触手手动控制脚本 - 带有详细反馈"""

import mujoco
import mujoco.viewer
import numpy as np
import time

def debug_tentacle_control():
    """带调试信息的触手控制"""
    
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
    cursor_x, cursor_y = 0.0, 0.0
    control_speed = 0.08  # 更高的控制速度
    
    print("\n🎮 调试控制说明:")
    print("   WASD - 控制触手方向")
    print("   SPACE - 重置到中心")
    print("   ESC - 退出")
    print("   🔍 控制台会显示按键检测状态\n")
    
    # 键盘状态和调试
    keys = {'w': False, 's': False, 'a': False, 'd': False}
    last_key_time = 0
    key_press_count = 0
    
    def key_callback(key, scancode, action, mods):
        """增强的键盘处理with调试"""
        nonlocal cursor_x, cursor_y, keys, last_key_time, key_press_count
        
        key_down = (action == 1) or (action == 2)  # PRESS or REPEAT
        current_time = time.time()
        
        # 调试：显示所有按键事件
        if action == 1:  # 只在按下时显示
            key_press_count += 1
            print(f"🔍 按键检测 #{key_press_count}: key={key}, action={action}")
        
        # 更新按键状态
        key_detected = False
        if key == 87:      # W
            keys['w'] = key_down
            key_detected = True
            if action == 1: print("⬆️ W键被按下!")
        elif key == 83:    # S
            keys['s'] = key_down  
            key_detected = True
            if action == 1: print("⬇️ S键被按下!")
        elif key == 65:    # A
            keys['a'] = key_down
            key_detected = True
            if action == 1: print("⬅️ A键被按下!")
        elif key == 68:    # D
            keys['d'] = key_down
            key_detected = True
            if action == 1: print("➡️ D键被按下!")
        
        # 功能键
        if action == 1:  # PRESS only
            if key == 32:  # SPACE - 重置
                cursor_x, cursor_y = 0.0, 0.0
                print("🎯 重置到中心")
                key_detected = True
            elif key == 256:  # ESC - 退出
                print("👋 退出...")
                key_detected = True
                return False
        
        if key_detected:
            last_key_time = current_time
    
    # 启动可视化
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.user_key_callback = key_callback
            
            print("🎬 开始调试控制...")
            print("💡 请确保MuJoCo窗口有焦点后按WASD键")
            step_count = 0
            last_control_update = time.time()
            
            while viewer.is_running():
                current_time = time.time()
                
                # 根据按键更新光标位置
                control_changed = False
                old_cursor_x, old_cursor_y = cursor_x, cursor_y
                
                if keys['w']: 
                    cursor_y += control_speed
                    control_changed = True
                if keys['s']: 
                    cursor_y -= control_speed
                    control_changed = True
                if keys['a']: 
                    cursor_x -= control_speed
                    control_changed = True
                if keys['d']: 
                    cursor_x += control_speed
                    control_changed = True
                
                # 限制范围
                cursor_x = np.clip(cursor_x, -0.8, 0.8)
                cursor_y = np.clip(cursor_y, -0.8, 0.8)
                
                # 如果控制发生变化，显示详细信息
                if control_changed and current_time - last_control_update > 0.1:
                    print(f"🎯 光标更新: [{old_cursor_x:.3f},{old_cursor_y:.3f}] → [{cursor_x:.3f},{cursor_y:.3f}]")
                    last_control_update = current_time
                
                # 增强的腱绳控制转换
                magnitude = np.sqrt(cursor_x**2 + cursor_y**2)
                if magnitude > 0.01:
                    effect = magnitude * 0.25  # 大幅增加控制强度!
                    
                    # 根据方向调整腱绳长度
                    old_ctrl = data.ctrl.copy()
                    data.ctrl[0] = baseline_lengths[0] - cursor_y * effect      # 上下控制
                    data.ctrl[1] = baseline_lengths[1] - (cursor_x * 0.866 - cursor_y * 0.5) * effect  # 右下
                    data.ctrl[2] = baseline_lengths[2] - (-cursor_x * 0.866 - cursor_y * 0.5) * effect # 左下
                    
                    # 如果腱绳控制发生明显变化，显示
                    if np.any(np.abs(data.ctrl - old_ctrl) > 0.01):
                        print(f"🦾 腱绳更新: [{old_ctrl[0]:.3f},{old_ctrl[1]:.3f},{old_ctrl[2]:.3f}] → [{data.ctrl[0]:.3f},{data.ctrl[1]:.3f},{data.ctrl[2]:.3f}]")
                else:
                    data.ctrl[:] = baseline_lengths
                
                # 确保在有效范围内
                data.ctrl[:] = np.clip(data.ctrl, 0.12, 0.34)
                
                # 运行仿真
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # 显示状态信息
                step_count += 1
                if step_count % 60 == 0:  # 每60步显示一次
                    # 获取尖端位置
                    try:
                        tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
                        if tip_site_id >= 0:
                            tip_pos = data.site_xpos[tip_site_id]
                        else:
                            tip_pos = data.site_xpos[-1] if model.nsite > 0 else [0, 0, 0]
                        
                        print(f"📊 状态: 光标[{cursor_x:5.2f},{cursor_y:5.2f}] | 尖端[{tip_pos[0]:5.2f},{tip_pos[1]:5.2f},{tip_pos[2]:5.2f}] | 腱绳[{data.ctrl[0]:.2f},{data.ctrl[1]:.2f},{data.ctrl[2]:.2f}]")
                    except:
                        pass
                
                # 控制帧率
                time.sleep(model.opt.timestep)
                
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        print("💡 提示: 在macOS上使用 'mjpython debug_manual_control.py'")

if __name__ == "__main__":
    debug_tentacle_control()
