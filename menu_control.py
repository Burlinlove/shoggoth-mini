#!/usr/bin/env python3
"""菜单驱动的触手控制 - 适合Mac触控板用户"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math

def menu_tentacle_control():
    """通过菜单选择控制触手动作"""
    
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
    current_action = "center"
    action_start_time = time.time()
    
    # 预定义的动作菜单
    action_menu = {
        "1": {"name": "向上弯曲", "cursor": (0.0, 0.7), "type": "static"},
        "2": {"name": "向下弯曲", "cursor": (0.0, -0.6), "type": "static"},
        "3": {"name": "向左弯曲", "cursor": (-0.7, 0.0), "type": "static"},
        "4": {"name": "向右弯曲", "cursor": (0.7, 0.0), "type": "static"},
        "5": {"name": "右上弯曲", "cursor": (0.5, 0.5), "type": "static"},
        "6": {"name": "右下弯曲", "cursor": (0.5, -0.5), "type": "static"},
        "7": {"name": "左上弯曲", "cursor": (-0.5, 0.5), "type": "static"},
        "8": {"name": "左下弯曲", "cursor": (-0.5, -0.5), "type": "static"},
        "9": {"name": "圆周运动", "cursor": "circle", "type": "dynamic"},
        "0": {"name": "波浪运动", "cursor": "wave", "type": "dynamic"},
        " ": {"name": "回到中心", "cursor": (0.0, 0.0), "type": "static"},
        "r": {"name": "随机运动", "cursor": "random", "type": "dynamic"},
    }
    
    def print_menu():
        """打印控制菜单"""
        print("\n" + "="*50)
        print("🎮 触手控制菜单")
        print("="*50)
        print("📋 静态动作:")
        print("   1️⃣ - 向上弯曲       2️⃣ - 向下弯曲")
        print("   3️⃣ - 向左弯曲       4️⃣ - 向右弯曲")
        print("   5️⃣ - 右上弯曲       6️⃣ - 右下弯曲")
        print("   7️⃣ - 左上弯曲       8️⃣ - 左下弯曲")
        print("\n🔄 动态动作:")
        print("   9️⃣ - 圆周运动       0️⃣ - 波浪运动")
        print("   R - 随机运动")
        print("\n⚙️ 控制:")
        print("   SPACE - 回到中心    ESC - 退出")
        print("   H - 显示此菜单")
        print("="*50)
        print("💡 按对应数字键执行动作，动作会立即生效！")
        print("🎯 当前动作: 中心位置")
    
    def convert_cursor_to_tendon_lengths(cursor_x, cursor_y):
        """将2D光标转换为腱绳长度"""
        magnitude = np.sqrt(cursor_x**2 + cursor_y**2)
        if magnitude > 0.01:
            effect = magnitude * 0.3  # 强控制效果
            
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
    
    def get_current_cursor_position(action, elapsed_time):
        """根据当前动作和时间计算光标位置"""
        if action not in action_menu:
            return 0.0, 0.0
        
        action_info = action_menu[action]
        cursor = action_info["cursor"]
        
        if action_info["type"] == "static":
            return cursor[0], cursor[1]
        elif cursor == "circle":
            # 圆周运动，5秒一圈
            angle = (elapsed_time / 5.0) * 2 * math.pi
            return 0.5 * math.cos(angle), 0.5 * math.sin(angle)
        elif cursor == "wave":
            # 波浪运动
            return (0.6 * math.sin(elapsed_time * 2), 
                   0.4 * math.cos(elapsed_time * 3))
        elif cursor == "random":
            # 随机运动，每2秒换一个随机目标
            seed = int(elapsed_time / 2.0)
            np.random.seed(seed)
            return (np.random.uniform(-0.6, 0.6), 
                   np.random.uniform(-0.6, 0.6))
        else:
            return 0.0, 0.0
    
    # 显示初始菜单
    print_menu()
    
    # 键盘状态
    last_key_press = 0
    
    def key_callback(key, scancode, action_type, mods):
        """键盘回调函数"""
        nonlocal current_action, action_start_time, last_key_press
        
        if action_type == 1:  # PRESS only
            current_time = time.time()
            
            # 防止重复按键
            if current_time - last_key_press < 0.1:
                return
            last_key_press = current_time
            
            # 数字键 1-9
            if 49 <= key <= 57:  # ASCII 1-9
                key_char = str(key - 48)
                if key_char in action_menu:
                    current_action = key_char
                    action_start_time = current_time
                    print(f"\n🎯 切换到: {action_menu[key_char]['name']}")
                    
            # 数字键 0
            elif key == 48:  # ASCII 0
                current_action = "0"
                action_start_time = current_time
                print(f"\n🎯 切换到: {action_menu['0']['name']}")
                
            # 空格键
            elif key == 32:  # SPACE
                current_action = " "
                action_start_time = current_time
                print(f"\n🎯 切换到: 回到中心")
                
            # R键
            elif key == 82 or key == 114:  # R or r
                current_action = "r"
                action_start_time = current_time
                print(f"\n🎯 切换到: 随机运动")
                
            # H键 - 帮助
            elif key == 72 or key == 104:  # H or h
                print_menu()
                
            # ESC键
            elif key == 256:  # ESC
                print("\n👋 退出...")
                return False
            
            else:
                # 显示未识别的按键（调试用）
                print(f"\n🔍 按键: {key} (按H查看菜单)")
    
    # 启动可视化
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.user_key_callback = key_callback
            
            print("\n🎬 菜单控制已启动!")
            print("💡 请确保MuJoCo窗口有焦点，然后按数字键选择动作")
            print("🎯 当前动作: 中心位置")
            
            step_count = 0
            last_status_time = time.time()
            
            while viewer.is_running():
                current_time = time.time()
                elapsed_in_action = current_time - action_start_time
                
                # 计算当前光标位置
                cursor_x, cursor_y = get_current_cursor_position(current_action, elapsed_in_action)
                
                # 转换为腱绳长度
                tendon_lengths = convert_cursor_to_tendon_lengths(cursor_x, cursor_y)
                data.ctrl[:] = tendon_lengths
                
                # 运行仿真
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # 定期显示状态信息
                if current_time - last_status_time > 2.0:  # 每2秒显示一次
                    tip_pos = get_tip_position()
                    action_name = action_menu.get(current_action, {}).get("name", "未知动作")
                    print(f"📊 {action_name}: 光标[{cursor_x:5.2f},{cursor_y:5.2f}] | "
                          f"尖端[{tip_pos[0]:5.2f},{tip_pos[1]:5.2f},{tip_pos[2]:5.2f}] | "
                          f"腱绳[{data.ctrl[0]:.2f},{data.ctrl[1]:.2f},{data.ctrl[2]:.2f}]")
                    last_status_time = current_time
                
                # 控制帧率
                time.sleep(model.opt.timestep)
                
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        print("💡 提示: 在macOS上使用 'mjpython menu_control.py'")

if __name__ == "__main__":
    menu_tentacle_control()
