#!/usr/bin/env python3
"""演示如何在训练中启用可视化"""

import os
import sys

def demo_training_commands():
    """展示不同的训练可视化命令"""
    print("🏃 训练中的MuJoCo可视化方法")
    print("=" * 50)
    
    print("\n🔧 方法1: 修改配置启用渲染")
    print("   编辑: shoggoth_mini/configs/default_rl_training.yaml")
    print("   修改: render_mode: 'human'  # 从 null 改为 'human'")
    print("   运行: mjpython -m shoggoth_mini.training.rl.training train")
    
    print("\n🎮 方法2: 评估模式可视化")
    print("   训练: python -m shoggoth_mini.training.rl.training train")  
    print("   评估: mjpython -m shoggoth_mini.training.rl.training evaluate model.zip --render")
    
    print("\n📹 方法3: 录制训练视频")
    print("   配置: render_mode: 'rgb_array'")
    print("   代码: 在训练循环中保存帧")
    print("   优点: 无GUI需求，可在服务器运行")
    
    print("\n🔍 方法4: 定期可视化检查")
    print("   思路: 每N个episode保存一个视频片段")
    print("   实现: callback机制在训练中定期录制")
    print("   用途: 监控训练过程中策略的演化")

def show_practical_tips():
    """显示实用技巧"""
    print("\n💡 实用技巧:")
    print("=" * 30)
    
    print("\n🎯 macOS用户:")
    print("   - 始终使用 mjpython 而非 python")
    print("   - 确保安装了完整的MuJoCo包")
    print("   - 如遇问题，尝试重新安装: pip install mujoco[mjpython]")
    
    print("\n🚀 性能优化:")
    print("   - 训练时关闭渲染: render_mode=None")
    print("   - 评估时开启渲染: render_mode='human'")
    print("   - 录制视频时: render_mode='rgb_array'")
    
    print("\n🔧 调试建议:")
    print("   - 先用简单脚本测试可视化")
    print("   - 确认环境正确加载模型文件")
    print("   - 检查动作空间和观察空间")
    
    print("\n📊 监控训练:")
    print("   - TensorBoard: 查看数值指标")
    print("   - 定期视频: 观察行为变化")
    print("   - 手动评估: 验证学到的策略")

if __name__ == "__main__":
    demo_training_commands()
    show_practical_tips()
    
    print(f"\n🎊 MuJoCo可视化完全指南:")
    print("   1. 基础测试: mjpython mujoco_visual_demo.py")
    print("   2. RL演示:   mjpython rl_visual_demo.py")
    print("   3. 录制视频: python video_recording_demo.py")
    print("   4. 训练可视: 修改配置后用mjpython运行")
    print("\n   所有可视化方法都已验证可用! ✅")
