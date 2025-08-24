#!/usr/bin/env python3
"""手动控制触手机器人的交互式脚本"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

# 导入项目模块
try:
    from shoggoth_mini.control.geometry import convert_2d_cursor_to_target_lengths
    from shoggoth_mini.common.constants import MOTOR_NAMES
except ImportError:
    print("警告: 无法导入项目模块，使用简化控制模式")
    MOTOR_NAMES = ["1", "2", "3"]


@dataclass
class ControlState:
    """控制状态"""
    cursor_x: float = 0.0
    cursor_y: float = 0.0
    max_magnitude: float = 0.8
    control_speed: float = 0.02
    auto_center: bool = False
    paused: bool = False
    show_info: bool = True


class ManualTentacleController:
    """手动触手控制器"""
    
    def __init__(self, xml_path: str = "assets/simulation/tentacle.xml"):
        self.xml_path = xml_path
        self.model = None
        self.data = None
        self.viewer = None
        self.state = ControlState()
        self.running = False
        
        # 控制参数
        self.baseline_lengths = np.array([0.25, 0.25, 0.25], dtype=np.float32)
        self.actuator_low = np.array([0.12, 0.12, 0.12], dtype=np.float32)
        self.actuator_high = np.array([0.34, 0.34, 0.34], dtype=np.float32)
        
        # 键盘状态
        self.key_states = {
            'w': False, 's': False, 'a': False, 'd': False,
            'up': False, 'down': False, 'left': False, 'right': False,
            'space': False, 'r': False, 'c': False, 'h': False
        }
        
        # 历史记录
        self.tip_history = []
        self.max_history = 1000
        
    def load_model(self):
        """加载MuJoCo模型"""
        try:
            print(f"🚀 加载MuJoCo模型: {self.xml_path}")
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.data = mujoco.MjData(self.model)
            
            print(f"✅ 模型加载成功!")
            print(f"   - 刚体数: {self.model.nbody}")
            print(f"   - 关节数: {self.model.njnt}")
            print(f"   - 执行器数: {self.model.nu}")
            print(f"   - 腱绳数: {self.model.ntendon}")
            
            # 初始化控制
            self.data.ctrl[:] = self.baseline_lengths
            mujoco.mj_forward(self.model, self.data)
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def get_cursor_position(self) -> Tuple[float, float]:
        """获取当前光标位置"""
        return self.state.cursor_x, self.state.cursor_y
    
    def set_cursor_position(self, x: float, y: float):
        """设置光标位置"""
        self.state.cursor_x = np.clip(x, -self.state.max_magnitude, self.state.max_magnitude)
        self.state.cursor_y = np.clip(y, -self.state.max_magnitude, self.state.max_magnitude)
    
    def update_control_from_keyboard(self):
        """根据键盘状态更新控制"""
        if self.state.paused:
            return
            
        # 计算移动
        dx = dy = 0.0
        speed = self.state.control_speed
        
        # WASD 控制
        if self.key_states['w'] or self.key_states['up']:
            dy += speed
        if self.key_states['s'] or self.key_states['down']:
            dy -= speed
        if self.key_states['a'] or self.key_states['left']:
            dx -= speed
        if self.key_states['d'] or self.key_states['right']:
            dx += speed
        
        # 更新光标位置
        new_x = self.state.cursor_x + dx
        new_y = self.state.cursor_y + dy
        self.set_cursor_position(new_x, new_y)
    
    def convert_to_tendon_lengths(self, cursor_pos: Tuple[float, float]) -> np.ndarray:
        """将2D光标位置转换为腱绳长度"""
        try:
            # 尝试使用项目的转换函数
            cursor_array = np.array(cursor_pos, dtype=np.float32)
            return convert_2d_cursor_to_target_lengths(
                cursor_array,
                self.baseline_lengths,
                self.actuator_low,
                self.actuator_high,
                self.state.max_magnitude
            )
        except:
            # 简化版本的转换
            return self._simple_conversion(cursor_pos)
    
    def _simple_conversion(self, cursor_pos: Tuple[float, float]) -> np.ndarray:
        """简化的控制转换（备用方案）"""
        x, y = cursor_pos
        
        # 简单的线性映射
        tendon_lengths = self.baseline_lengths.copy()
        
        # 根据光标位置调整腱绳长度
        magnitude = np.sqrt(x*x + y*y)
        if magnitude > 0.01:
            effect_scale = magnitude / self.state.max_magnitude * 0.1
            
            # 简单的方向映射
            tendon_lengths[0] -= y * effect_scale  # 上下
            tendon_lengths[1] -= (x * 0.866 - y * 0.5) * effect_scale  # 右下
            tendon_lengths[2] -= (-x * 0.866 - y * 0.5) * effect_scale  # 左下
        
        # 确保在有效范围内
        return np.clip(tendon_lengths, self.actuator_low, self.actuator_high)
    
    def get_tip_position(self) -> np.ndarray:
        """获取触手尖端位置"""
        try:
            # 查找尖端site
            tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
            if tip_site_id >= 0:
                return self.data.site_xpos[tip_site_id].copy()
        except:
            pass
        
        # 备用方案：使用最后一个site
        if self.model.nsite > 0:
            return self.data.site_xpos[-1].copy()
        else:
            return np.zeros(3)
    
    def update_physics(self):
        """更新物理仿真"""
        if self.state.paused:
            return
            
        # 更新键盘控制
        self.update_control_from_keyboard()
        
        # 自动回中
        if self.state.auto_center and not any(self.key_states.values()):
            decay = 0.95
            self.state.cursor_x *= decay
            self.state.cursor_y *= decay
        
        # 转换光标位置到腱绳长度
        cursor_pos = self.get_cursor_position()
        tendon_lengths = self.convert_to_tendon_lengths(cursor_pos)
        
        # 设置控制命令
        self.data.ctrl[:] = tendon_lengths
        
        # 运行物理步骤
        mujoco.mj_step(self.model, self.data)
        
        # 记录尖端历史
        tip_pos = self.get_tip_position()
        self.tip_history.append(tip_pos.copy())
        if len(self.tip_history) > self.max_history:
            self.tip_history.pop(0)
    
    def print_status(self):
        """打印当前状态"""
        if not self.state.show_info:
            return
            
        cursor_x, cursor_y = self.get_cursor_position()
        tip_pos = self.get_tip_position()
        magnitude = np.sqrt(cursor_x*cursor_x + cursor_y*cursor_y)
        
        print(f"\r🎮 光标: [{cursor_x:6.3f}, {cursor_y:6.3f}] | "
              f"强度: {magnitude:5.3f} | "
              f"尖端: [{tip_pos[0]:6.3f}, {tip_pos[1]:6.3f}, {tip_pos[2]:6.3f}] | "
              f"腱绳: [{self.data.ctrl[0]:.3f}, {self.data.ctrl[1]:.3f}, {self.data.ctrl[2]:.3f}]", 
              end="")
    
    def print_help(self):
        """打印帮助信息"""
        print("\n" + "="*60)
        print("🎮 触手手动控制帮助")
        print("="*60)
        print("📋 控制方式:")
        print("   WASD 或 方向键  - 控制触手方向")
        print("   鼠标拖拽        - 直接控制光标位置 (如果支持)")
        print()
        print("⚙️ 功能键:")
        print("   SPACE  - 暂停/继续仿真")
        print("   R      - 重置到中心位置")
        print("   C      - 开启/关闭自动回中")
        print("   H      - 显示/隐藏此帮助信息")
        print("   +/-    - 增加/减少控制速度")
        print("   ESC    - 退出程序")
        print()
        print("📊 当前设置:")
        print(f"   控制速度: {self.state.control_speed:.3f}")
        print(f"   最大范围: {self.state.max_magnitude:.3f}")
        print(f"   自动回中: {'开启' if self.state.auto_center else '关闭'}")
        print(f"   显示信息: {'开启' if self.state.show_info else '关闭'}")
        print("="*60)
    
    def handle_key_callback(self, key, scancode, action, mods):
        """键盘回调函数"""
        key_down = (action == 1) or (action == 2)  # PRESS or REPEAT
        
        # 更新键盘状态
        key_map = {
            87: 'w',      # W
            83: 's',      # S  
            65: 'a',      # A
            68: 'd',      # D
            265: 'up',    # UP
            264: 'down',  # DOWN
            263: 'left',  # LEFT
            262: 'right', # RIGHT
        }
        
        if key in key_map:
            self.key_states[key_map[key]] = key_down
        
        # 功能键（只在按下时触发）
        if action == 1:  # PRESS
            if key == 32:  # SPACE
                self.state.paused = not self.state.paused
                print(f"\n{'⏸️ 暂停' if self.state.paused else '▶️ 继续'} 仿真")
                
            elif key == 82:  # R
                self.state.cursor_x = 0.0
                self.state.cursor_y = 0.0
                print(f"\n🎯 重置到中心位置")
                
            elif key == 67:  # C
                self.state.auto_center = not self.state.auto_center
                print(f"\n🎯 自动回中: {'开启' if self.state.auto_center else '关闭'}")
                
            elif key == 72:  # H
                self.print_help()
                
            elif key == 61:  # +
                self.state.control_speed = min(0.1, self.state.control_speed + 0.005)
                print(f"\n⬆️ 控制速度: {self.state.control_speed:.3f}")
                
            elif key == 45:  # -
                self.state.control_speed = max(0.001, self.state.control_speed - 0.005)
                print(f"\n⬇️ 控制速度: {self.state.control_speed:.3f}")
                
            elif key == 256:  # ESC
                self.running = False
                print(f"\n👋 退出程序...")
    
    def run(self):
        """运行交互式控制"""
        if not self.load_model():
            return
        
        print("\n🎯 启动手动触手控制...")
        self.print_help()
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 设置键盘回调
                viewer.user_key_callback = self.handle_key_callback
                
                self.running = True
                step_counter = 0
                
                while self.running and viewer.is_running():
                    # 更新物理
                    self.update_physics()
                    
                    # 同步显示
                    viewer.sync()
                    
                    # 每10步打印一次状态
                    step_counter += 1
                    if step_counter % 10 == 0:
                        self.print_status()
                    
                    # 控制帧率
                    time.sleep(self.model.opt.timestep)
                
                print(f"\n✅ 手动控制session完成!")
                
        except Exception as e:
            print(f"❌ 运行错误: {e}")
            print(f"   提示: 在macOS上使用 mjpython manual_tentacle_control.py")


def main():
    """主函数"""
    print("🦾 MuJoCo触手手动控制器")
    print("   作者: AI Assistant")
    print("   版本: 1.0")
    
    # 解析命令行参数
    xml_path = "assets/simulation/tentacle.xml"
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    
    # 创建并运行控制器
    controller = ManualTentacleController(xml_path)
    controller.run()


if __name__ == "__main__":
    main()
