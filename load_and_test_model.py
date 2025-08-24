#!/usr/bin/env python3
"""演示如何加载训练好的RL模型并在MuJoCo中测试"""

import sys
import os
import time
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from shoggoth_mini.training.rl.environment import TentacleTargetFollowingEnv
from shoggoth_mini.configs.rl_training import RLEnvironmentConfig

def find_latest_model():
    """查找最新训练的模型"""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    
    # 查找最新的训练运行
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("ppo_tentacle_")]
    if not run_dirs:
        return None
    
    latest_run = max(run_dirs, key=lambda d: d.name)
    
    # 优先选择最佳模型，其次是最终模型
    best_model_path = latest_run / "models" / "best_model.zip"
    final_model_path = latest_run / "models" / "final_model.zip"
    
    if best_model_path.exists():
        return best_model_path
    elif final_model_path.exists():
        return final_model_path
    else:
        return None

def test_trained_model(model_path: str = None, num_episodes: int = 5, step_delay: float = 0.05):
    """测试训练好的模型"""
    
    # 如果没有提供模型路径，自动查找
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("❌ 没有找到训练好的模型！")
            print("   请先运行训练: python -m shoggoth_mini.training.rl.training train")
            return
        print(f"🔍 自动找到模型: {model_path}")
    else:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return
    
    print(f"🚀 加载模型: {model_path}")
    
    try:
        # 加载训练好的PPO模型
        model = PPO.load(str(model_path))
        print("✅ 模型加载成功!")
        
        # 创建测试环境（带可视化）
        config = RLEnvironmentConfig()
        env = TentacleTargetFollowingEnv(config=config, render_mode="human")
        print("✅ MuJoCo环境已创建!")
        
        print(f"\n🎬 开始测试训练好的智能体...")
        print(f"   - 测试回合数: {num_episodes}")
        print(f"   - 步骤延迟: {step_delay:.3f}秒 (让动作更容易观察)")
        print(f"   - 观察智能体如何控制触手追踪目标")
        print(f"   - 按ESC可提前退出")
        
        # 性能统计
        episode_rewards = []
        episode_lengths = []
        success_episodes = 0
        success_threshold = 0.05  # 5cm内算成功
        
        for episode in range(num_episodes):
            print(f"\n📊 第 {episode + 1}/{num_episodes} 回合")
            
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # 智能体决策 (确定性动作)
                action, _states = model.predict(obs, deterministic=True)
                
                # 环境步进
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 添加延迟让步骤更容易观察
                time.sleep(step_delay)
                
                episode_reward += reward
                episode_length += 1
                
                # 显示当前状态
                if episode_length % 20 == 0:  # 每20步显示一次
                    tip_pos = info.get('tip_position', [0, 0, 0])
                    target_pos = info.get('target_position', [0, 0, 0])
                    distance = info.get('distance', 0)
                    print(f"     步骤 {episode_length}: 距离={distance:.3f}m, 奖励={reward:.3f}")
                
                # 检查是否结束
                if terminated or truncated:
                    break
            
            # 记录统计信息
            final_distance = info.get('distance', float('inf'))
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if final_distance < success_threshold:
                success_episodes += 1
                success_status = "✅ 成功"
            else:
                success_status = "❌ 失败"
            
            print(f"     回合结果: 奖励={episode_reward:.2f}, 步数={episode_length}, 最终距离={final_distance:.3f}m {success_status}")
        
        env.close()
        
        # 显示总体统计
        print(f"\n📈 测试完成! 整体表现:")
        print(f"   - 平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"   - 平均步数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"   - 成功率: {success_episodes}/{num_episodes} ({100*success_episodes/num_episodes:.1f}%)")
        print(f"   - 平均最终距离: {np.mean([info.get('distance', 0) for info in [{}]]):.3f}m")
        
        print(f"\n🎊 模型测试完成! 智能体表现:", end="")
        if success_episodes >= num_episodes * 0.8:
            print("优秀! 🏆")
        elif success_episodes >= num_episodes * 0.6:
            print("良好! 👍")
        elif success_episodes >= num_episodes * 0.4:
            print("还行! 🤔")
        else:
            print("需要更多训练! 💪")
            
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🧠 训练好的RL模型MuJoCo测试器")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"📁 指定模型路径: {model_path}")
    else:
        model_path = None
        print("🔍 自动查找最新训练的模型...")
    
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    step_delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    
    print(f"🎮 测试参数:")
    print(f"   - 回合数: {num_episodes}")
    print(f"   - 步骤延迟: {step_delay:.3f}秒")
    
    test_trained_model(model_path, num_episodes, step_delay)

if __name__ == "__main__":
    main()
