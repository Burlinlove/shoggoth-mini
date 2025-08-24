#!/usr/bin/env python3
"""测试强化学习环境"""

import numpy as np
from shoggoth_mini.training.rl.environment import TentacleTargetFollowingEnv
from shoggoth_mini.configs.rl_training import RLEnvironmentConfig

def test_rl_environment():
    """测试RL环境基本功能"""
    
    print("🚀 Creating RL environment...")
    
    # 创建环境配置
    config = RLEnvironmentConfig()
    
    # 创建环境
    env = TentacleTargetFollowingEnv(config=config, render_mode=None)
    
    print(f"✅ Environment created successfully!")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Max episode steps: {env._max_episode_steps}")
    print(f"   - Time per step: {env.time_per_step:.3f}s")
    
    # 重置环境
    print(f"\n🔄 Resetting environment...")
    obs, info = env.reset()
    print(f"   - Initial observation shape: {obs.shape}")
    print(f"   - Target position: {info.get('target_trajectory', [{}])[0].get('position', 'N/A')}")
    
    # 测试随机动作
    print(f"\n🎮 Testing random actions...")
    total_reward = 0
    
    for step in range(10):
        # 随机2D动作（光标位置）
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 获取信息
        tip_pos = info.get('tip_position', [0, 0, 0])
        target_pos = info.get('target_position', [0, 0, 0])
        distance = info.get('distance', 0)
        
        if step % 3 == 0:
            print(f"   Step {step}:")
            print(f"     - Action (2D cursor): [{action[0]:.3f}, {action[1]:.3f}]")
            print(f"     - Tip position: [{tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f}]")
            print(f"     - Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            print(f"     - Distance to target: {distance:.3f}m")
            print(f"     - Reward: {reward:.3f}")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step}")
            break
    
    print(f"\n📊 Episode summary:")
    print(f"   - Total reward: {total_reward:.3f}")
    print(f"   - Final distance: {info.get('distance', 0):.3f}m")
    
    # 测试固定动作序列
    print(f"\n🎯 Testing fixed action sequence...")
    obs, info = env.reset()
    
    # 测试四个方向的移动
    test_actions = [
        [0.5, 0.0],    # 右
        [0.0, 0.5],    # 上  
        [-0.5, 0.0],   # 左
        [0.0, -0.5],   # 下
        [0.0, 0.0],    # 中心
    ]
    
    for i, action in enumerate(test_actions):
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        tip_pos = info.get('tip_position', [0, 0, 0])
        tendon_lengths = info.get('tendon_lengths', [0, 0, 0])
        
        print(f"   Action {i+1} {action}: tip=[{tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f}], "
              f"tendons=[{tendon_lengths[0]:.3f}, {tendon_lengths[1]:.3f}, {tendon_lengths[2]:.3f}]")
    
    env.close()
    print(f"\n🎉 RL environment test completed successfully!")

if __name__ == "__main__":
    test_rl_environment()
