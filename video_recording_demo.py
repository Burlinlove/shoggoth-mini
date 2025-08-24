#!/usr/bin/env python3
"""录制MuJoCo仿真视频演示"""

import numpy as np
from shoggoth_mini.training.rl.environment import TentacleTargetFollowingEnv
from shoggoth_mini.configs.rl_training import RLEnvironmentConfig
import cv2
import os

def record_mujoco_video():
    """录制MuJoCo仿真视频"""
    print("📹 MuJoCo视频录制演示")
    print("=" * 40)
    
    # 创建环境（rgb_array模式用于录制）
    config = RLEnvironmentConfig()
    env = TentacleTargetFollowingEnv(config=config, render_mode="rgb_array")
    
    print("✅ 环境创建成功 (rgb_array模式)")
    
    # 视频参数
    fps = 30
    video_filename = "tentacle_demo.mp4"
    frames = []
    
    try:
        # 重置环境
        obs, info = env.reset()
        
        print("🎬 开始录制...")
        
        # 录制50步
        for step in range(50):
            # 随机动作
            action = env.action_space.sample()
            
            # 执行步骤
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 渲染当前帧
            frame = env.render()  # 返回numpy数组 (height, width, 3)
            
            if frame is not None:
                frames.append(frame)
            
            if step % 10 == 0:
                print(f"   已录制 {step} 帧...")
            
            if terminated or truncated:
                obs, info = env.reset()
        
        print(f"✅ 录制完成! 总共 {len(frames)} 帧")
        
        # 保存视频
        if frames:
            print(f"💾 保存视频到 {video_filename}...")
            
            # 获取帧尺寸
            height, width, _ = frames[0].shape
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            
            for frame in frames:
                # OpenCV使用BGR，MuJoCo返回RGB，需要转换
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            # 检查文件
            if os.path.exists(video_filename):
                file_size = os.path.getsize(video_filename) / 1024 / 1024  # MB
                print(f"✅ 视频保存成功!")
                print(f"   - 文件名: {video_filename}")
                print(f"   - 文件大小: {file_size:.2f} MB")
                print(f"   - 分辨率: {width}x{height}")
                print(f"   - 帧率: {fps} FPS")
                print(f"   - 时长: {len(frames)/fps:.1f} 秒")
            else:
                print("❌ 视频保存失败")
        
    except Exception as e:
        print(f"❌ 录制错误: {e}")
    finally:
        env.close()

def show_video_advantages():
    """展示视频录制的优势"""
    print("\n📹 视频录制模式的优势:")
    print("=" * 40)
    print("✅ 跨平台兼容 - 所有系统都支持")
    print("✅ 服务器友好 - 无需GUI环境")
    print("✅ 便于分享 - 生成标准MP4文件")
    print("✅ 离线观看 - 随时播放查看")
    print("✅ 高质量录制 - 自定义分辨率和帧率")
    print("✅ 自动化友好 - 可集成到训练脚本")

if __name__ == "__main__":
    record_mujoco_video()
    show_video_advantages()
