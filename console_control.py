#!/usr/bin/env python3
"""控制台输入控制触手 - 完全绕过MuJoCo键盘事件"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import threading
import sys

class TentacleController:
    def __init__(self):
        self.model_path = "assets/simulation/tentacle.xml"
        self.model = None
        self.data = None
        self.viewer = None
        self.running = False
        
        # 控制参数
        self.baseline_lengths = np.array([0.25, 0.25, 0.25])
        self.current_action = "center"
        self.action_start_time = time.time()
        
        # 动作定义
        self.actions = {
            "1": {"name": "向上弯曲", "cursor": (0.0, 0.7)},
            "2": {"name": "向下弯曲", "cursor": (0.0, -0.6)},
            "3": {"name": "向左弯曲", "cursor": (-0.7, 0.0)},
            "4": {"name": "向右弯曲", "cursor": (0.7, 0.0)},
            "5": {"name": "右上弯曲", "cursor": (0.5, 0.5)},
            "6": {"name": "右下弯曲", "cursor": (0.5, -0.5)},
            "7": {"name": "左上弯曲", "cursor": (-0.5, 0.5)},
            "8": {"name": "左下弯曲", "cursor": (-0.5, -0.5)},
            "9": {"name": "圆周运动", "cursor": "circle"},
            "0": {"name": "波浪运动", "cursor": "wave"},
            "r": {"name": "随机运动", "cursor": "random"},
            "c": {"name": "回到中心", "cursor": (0.0, 0.0)},
            "q": {"name": "退出", "cursor": "quit"}
        }
    
    def load_model(self):
        """加载MuJoCo模型"""
        try:
            print(f"🚀 加载模型: {self.model_path}")
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            print("✅ 模型加载成功!")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def print_menu(self):
        """打印控制菜单"""
        print("\n" + "="*60)
        print("🎮 触手控制台控制 - 支持Mac触控板!")
        print("="*60)
        print("📋 可用指令:")
        print("   1 - 向上弯曲        2 - 向下弯曲")
        print("   3 - 向左弯曲        4 - 向右弯曲")
        print("   5 - 右上弯曲        6 - 右下弯曲")
        print("   7 - 左上弯曲        8 - 左下弯曲")
        print()
        print("   9 - 圆周运动        0 - 波浪运动")
        print("   r - 随机运动        c - 回到中心")
        print("   q - 退出程序        h - 显示菜单")
        print()
        print("💡 使用方法:")
        print("   1. 在此控制台输入数字或字母")
        print("   2. 按回车键执行")
        print("   3. 触手会立即响应您的指令!")
        print("="*60)
        print(f"🎯 当前动作: {self.get_current_action_name()}")
        print("请输入指令 (输入 'h' 显示菜单): ", end="", flush=True)
    
    def get_current_action_name(self):
        """获取当前动作名称"""
        return self.actions.get(self.current_action, {}).get("name", "中心位置")
    
    def convert_cursor_to_tendon_lengths(self, cursor_x, cursor_y):
        """将2D光标转换为腱绳长度"""
        magnitude = np.sqrt(cursor_x**2 + cursor_y**2)
        if magnitude > 0.01:
            effect = magnitude * 0.35  # 很强的控制效果
            
            tendon_lengths = self.baseline_lengths.copy()
            tendon_lengths[0] -= cursor_y * effect      # 上下控制
            tendon_lengths[1] -= (cursor_x * 0.866 - cursor_y * 0.5) * effect  # 右下
            tendon_lengths[2] -= (-cursor_x * 0.866 - cursor_y * 0.5) * effect # 左下
            
            return np.clip(tendon_lengths, 0.12, 0.34)
        else:
            return self.baseline_lengths
    
    def get_tip_position(self):
        """获取触手尖端位置"""
        try:
            tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
            if tip_site_id >= 0:
                return self.data.site_xpos[tip_site_id].copy()
            else:
                return self.data.site_xpos[-1].copy() if self.model.nsite > 0 else np.zeros(3)
        except:
            return np.zeros(3)
    
    def get_current_cursor_position(self):
        """根据当前动作计算光标位置"""
        if self.current_action not in self.actions:
            return 0.0, 0.0
        
        cursor = self.actions[self.current_action]["cursor"]
        elapsed_time = time.time() - self.action_start_time
        
        if isinstance(cursor, tuple):
            return cursor[0], cursor[1]
        elif cursor == "circle":
            # 圆周运动，4秒一圈
            angle = (elapsed_time / 4.0) * 2 * math.pi
            return 0.6 * math.cos(angle), 0.6 * math.sin(angle)
        elif cursor == "wave":
            # 波浪运动
            return (0.7 * math.sin(elapsed_time * 1.5), 
                   0.5 * math.cos(elapsed_time * 2.5))
        elif cursor == "random":
            # 随机运动，每1.5秒换一个随机目标
            seed = int(elapsed_time / 1.5)
            np.random.seed(seed)
            return (np.random.uniform(-0.7, 0.7), 
                   np.random.uniform(-0.7, 0.7))
        else:
            return 0.0, 0.0
    
    def input_handler(self):
        """处理控制台输入的线程"""
        while self.running:
            try:
                user_input = input().strip().lower()
                
                if user_input == 'h':
                    self.print_menu()
                elif user_input == 'q':
                    print("👋 退出程序...")
                    self.running = False
                    break
                elif user_input in self.actions:
                    old_action = self.current_action
                    self.current_action = user_input
                    self.action_start_time = time.time()
                    action_name = self.actions[user_input]["name"]
                    
                    print(f"🎯 切换: {self.actions.get(old_action, {}).get('name', '中心位置')} → {action_name}")
                    print("请输入下一个指令 (输入 'h' 显示菜单): ", end="", flush=True)
                elif user_input == '':
                    print("请输入指令 (输入 'h' 显示菜单): ", end="", flush=True)
                else:
                    print(f"❓ 未识别的指令: '{user_input}' (输入 'h' 查看可用指令)")
                    print("请输入指令: ", end="", flush=True)
                    
            except (EOFError, KeyboardInterrupt):
                print("\n👋 程序被中断，退出...")
                self.running = False
                break
            except Exception as e:
                print(f"❌ 输入处理错误: {e}")
    
    def run(self):
        """运行主程序"""
        if not self.load_model():
            return
        
        self.print_menu()
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.viewer = viewer
                self.running = True
                
                # 启动输入处理线程
                input_thread = threading.Thread(target=self.input_handler, daemon=True)
                input_thread.start()
                
                print("🎬 控制台控制已启动!")
                print("💡 MuJoCo窗口已打开，触手会根据您的控制台输入做出反应")
                print("🎯 当前动作: 中心位置")
                print("请输入指令 (输入 'h' 显示菜单): ", end="", flush=True)
                
                step_count = 0
                last_status_time = time.time()
                
                while self.running and viewer.is_running():
                    current_time = time.time()
                    
                    # 计算当前光标位置
                    cursor_x, cursor_y = self.get_current_cursor_position()
                    
                    # 转换为腱绳长度
                    tendon_lengths = self.convert_cursor_to_tendon_lengths(cursor_x, cursor_y)
                    self.data.ctrl[:] = tendon_lengths
                    
                    # 运行仿真
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    
                    # 定期显示状态信息（不干扰用户输入）
                    if current_time - last_status_time > 3.0:  # 每3秒显示一次状态
                        tip_pos = self.get_tip_position()
                        action_name = self.get_current_action_name()
                        
                        # 在新行显示状态，不影响输入提示
                        print(f"\n📊 状态更新 - {action_name}: 光标[{cursor_x:5.2f},{cursor_y:5.2f}] | "
                              f"尖端[{tip_pos[0]:5.2f},{tip_pos[1]:5.2f},{tip_pos[2]:5.2f}] | "
                              f"腱绳[{self.data.ctrl[0]:.2f},{self.data.ctrl[1]:.2f},{self.data.ctrl[2]:.2f}]")
                        print("请输入指令: ", end="", flush=True)
                        last_status_time = current_time
                    
                    # 控制帧率
                    time.sleep(self.model.opt.timestep)
                
                print("\n✅ 控制台控制session完成!")
                
        except Exception as e:
            print(f"❌ 运行错误: {e}")
            self.running = False

def main():
    """主函数"""
    controller = TentacleController()
    controller.run()

if __name__ == "__main__":
    main()
