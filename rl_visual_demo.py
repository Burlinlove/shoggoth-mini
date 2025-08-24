#!/usr/bin/env python3
"""RL环境可视化演示"""

from shoggoth_mini.training.rl.environment import TentacleTargetFollowingEnv
from shoggoth_mini.configs.rl_training import RLEnvironmentConfig
import numpy as np
import time

def demo_rl_visualization():
    """演示RL环境可视化"""
    print("🧠 RL环境可视化演示")
    print("=" * 40)
    
    # 创建配置
    config = RLEnvironmentConfig()
    
    # 创建环境（启用human渲染模式）
    env = TentacleTargetFollowingEnv(config=config, render_mode="human")
    
    print("✅ RL环境已创建")
    print("🎯 目标: 控制触手跟随红色目标球")
    print("🎮 演示: 将执行预编程的控制序列")
    print("📊 注意观察目标位置和触手尖端的运动")
    
    try:
        # 重置环境
        obs, info = env.reset()
        
        # 预设的控制序列：画圆形轨迹
        circle_actions = []
        n_steps = 20
        for i in range(n_steps):
            angle = 2 * np.pi * i / n_steps
            x = 0.5 * np.cos(angle)
            y = 0.5 * np.sin(angle)
            circle_actions.append([x, y])
        
        print(f"\n🎬 开始演示 - 尝试画圆形轨迹...")
        
        total_reward = 0
        for step, action in enumerate(circle_actions):
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            total_reward += reward
            
            # 获取状态信息
            tip_pos = info.get('tip_position', [0, 0, 0])
            target_pos = info.get('target_position', [0, 0, 0]) 
            distance = info.get('distance', 0)
            
            print(f"   步骤 {step+1:2d}: 动作=[{action[0]:5.2f}, {action[1]:5.2f}] "
                  f"距离={distance:.3f}m 奖励={reward:.3f}")
            
            # 慢一点，让人能看清
            time.sleep(0.2)
            
            if terminated or truncated:
                print("   Episode结束")
                break
        
        print(f"\n📊 演示完成:")
        print(f"   - 总奖励: {total_reward:.3f}")
        print(f"   - 执行步数: {step+1}")
        
    except Exception as e:
        print(f"❌ 演示错误: {e}")
        print("   提示: 确保使用 mjpython rl_visual_demo.py")
    finally:
        env.close()

if __name__ == "__main__":
    demo_rl_visualization()
