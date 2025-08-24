#!/usr/bin/env python3
"""MuJoCo可视化演示 - 展示不同的启动方式"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys


def demo_basic_visualization():
    """基础MuJoCo可视化演示"""
    print("🚀 MuJoCo可视化演示启动...")
    print("=" * 50)
    
    # 加载模型
    model_path = "assets/simulation/tentacle.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"✅ 模型加载成功!")
    print(f"   - 模型文件: {model_path}")
    print(f"   - 刚体数: {model.nbody}")
    print(f"   - 关节数: {model.njnt}")
    print(f"   - 执行器数: {model.nu}")
    
    # 设置初始控制
    initial_control = [0.25, 0.25, 0.25]
    data.ctrl[:] = initial_control
    
    print(f"\n🎮 启动3D可视化窗口...")
    print("操作说明:")
    print("  - 鼠标左键拖拽: 旋转视角")
    print("  - 鼠标右键拖拽: 平移视角")  
    print("  - 鼠标滚轮: 缩放")
    print("  - 空格键: 暂停/继续仿真")
    print("  - ESC或关闭窗口: 退出")
    print("  - 程序将运行15秒后自动退出")
    
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            step = 0
            
            print(f"\n🎬 开始仿真动画...")
            
            while viewer.is_running():
                current_time = time.time() - start_time
                
                # 创建有趣的运动模式：三个腱绳按不同频率振荡
                data.ctrl[0] = 0.25 + 0.08 * np.sin(current_time * 1.0)      # 慢频率
                data.ctrl[1] = 0.25 + 0.06 * np.sin(current_time * 1.5 + 2.0) # 中频率
                data.ctrl[2] = 0.25 + 0.04 * np.sin(current_time * 2.0 + 4.0) # 快频率
                
                # 运行仿真步骤
                mujoco.mj_step(model, data)
                
                # 更新显示
                viewer.sync()
                
                # 控制帧率
                time.sleep(model.opt.timestep)
                
                step += 1
                
                # 每100步显示一次状态
                if step % 100 == 0:
                    tip_pos = data.site_xpos[-1]  # 获取尖端位置
                    print(f"   步骤 {step}: 尖端位置 = [{tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f}]")
                
                # 15秒后退出
                if current_time > 15.0:
                    print(f"\n⏰ 演示完成 (15秒)，退出...")
                    break
            
            print(f"✅ 可视化演示成功完成!")
            
    except Exception as e:
        print(f"❌ 可视化错误: {e}")
        print(f"   提示: 在macOS上需要使用 mjpython 运行此脚本")
        print(f"   命令: mjpython mujoco_visual_demo.py")
        return False
    
    return True


def demo_rl_model_visualization(delay_seconds: float = 1.0):
    """演示如何可视化训练好的RL模型"""
    print(f"\n🧠 RL模型可视化演示...")
    
    try:
        from shoggoth_mini.training.rl.environment import TentacleTargetFollowingEnv
        from shoggoth_mini.configs.rl_training import RLEnvironmentConfig
        
        # 创建环境（带可视化）
        config = RLEnvironmentConfig()
        env = TentacleTargetFollowingEnv(config=config, render_mode="human")
        
        print("✅ RL环境已创建，开始演示随机策略...")
        print(f"   💡 演示步骤间延迟: {delay_seconds}秒")
        print("   👆 你将看到3D触手跟随目标移动")
        print("   📊 控制台显示每步的动作和奖励信息")
        
        # 重置环境
        obs, info = env.reset()
        
        # 运行几个随机动作
        for step in range(20):
            action = env.action_space.sample()  # 随机2D光标动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 获取更多信息用于显示
            tip_pos = info.get('tip_position', [0, 0, 0])
            target_pos = info.get('target_position', [0, 0, 0])
            distance = info.get('distance', 0)
            
            print(f"   步骤 {step:2d}: 动作=[{action[0]:6.3f}, {action[1]:6.3f}] | "
                  f"奖励={reward:7.3f} | 距离={distance:6.3f}m")
            print(f"           尖端位置=[{tip_pos[0]:6.3f}, {tip_pos[1]:6.3f}, {tip_pos[2]:6.3f}] | "
                  f"目标位置=[{target_pos[0]:6.3f}, {target_pos[1]:6.3f}, {target_pos[2]:6.3f}]")
            
            # 添加延迟让用户观察
            time.sleep(delay_seconds)
            
            if terminated or truncated:
                print("   🔄 重置环境...")
                obs, info = env.reset()
                time.sleep(0.5)  # 重置后短暂停顿
        
        env.close()
        print("✅ RL演示完成!")
        
    except Exception as e:
        print(f"❌ RL演示失败: {e}")
        return False
    
    return True


def show_visualization_methods():
    """展示不同的可视化启动方法"""
    print(f"\n📚 MuJoCo可视化方法总结:")
    print("=" * 50)
    
    print(f"\n🍎 方法1: macOS使用mjpython (推荐)")
    print("   命令: mjpython mujoco_visual_demo.py")
    print("   优点: 原生支持，最稳定")
    print("   适用: macOS系统")
    
    print(f"\n🐧 方法2: Linux/Windows直接运行")
    print("   命令: python mujoco_visual_demo.py")
    print("   优点: 直接支持")
    print("   适用: Linux/Windows系统")
    
    print(f"\n📹 方法3: 录制视频（所有系统）")
    print("   模式: render_mode='rgb_array'")
    print("   优点: 无需GUI，保存为MP4")
    print("   适用: 所有系统，服务器环境")
    
    print(f"\n🌐 方法4: Jupyter Notebook")
    print("   模式: 在线可视化")
    print("   优点: 浏览器中展示")
    print("   适用: Google Colab, Jupyter")


if __name__ == "__main__":
    print("🎯 MuJoCo可视化启动演示")
    print("   当前平台:", sys.platform)
    
    # 解析命令行参数
    delay_seconds = 1.0  # 默认延迟1秒
    
    # 检查是否指定了延迟时间
    for i, arg in enumerate(sys.argv):
        if arg == "--delay" and i + 1 < len(sys.argv):
            try:
                delay_seconds = float(sys.argv[i + 1])
                print(f"   ⏱️  设置延迟时间: {delay_seconds}秒")
            except ValueError:
                print(f"   ⚠️  无效的延迟时间: {sys.argv[i + 1]}，使用默认值1.0秒")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--methods":
        show_visualization_methods()
    elif len(sys.argv) > 1 and sys.argv[1] == "--rl":
        demo_rl_model_visualization(delay_seconds=delay_seconds)
    elif len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(f"\n📋 使用说明:")
        print("   python mujoco_visual_demo.py              # 基础MuJoCo演示")
        print("   python mujoco_visual_demo.py --rl         # RL环境演示（默认1秒延迟）")  
        print("   python mujoco_visual_demo.py --rl --delay 2.0  # RL演示（自定义延迟）")
        print("   python mujoco_visual_demo.py --methods    # 显示可视化方法")
        print("   python mujoco_visual_demo.py --help       # 显示此帮助信息")
        print(f"\n💡 延迟参数:")
        print("   --delay 0.5   # 快速演示（0.5秒间隔）")
        print("   --delay 1.0   # 标准演示（1秒间隔，默认）")  
        print("   --delay 2.0   # 慢速演示（2秒间隔）")
        print("   --delay 0.1   # 非常快速（0.1秒间隔）")
    else:
        # 主演示
        success = demo_basic_visualization()
        if success:
            print(f"\n🎊 所有可视化功能正常工作!")
        else:
            print(f"\n⚠️  可视化需要正确的运行方式")
            show_visualization_methods()
