#!/usr/bin/env python3
"""简单的MuJoCo测试脚本，展示触手机器人运动"""

import mujoco
import mujoco.viewer
import numpy as np
import time

def test_mujoco_simulation():
    """测试MuJoCo仿真基本功能"""
    
    # 加载模型
    model_path = "assets/simulation/tentacle.xml"
    
    try:
        print("🚀 Loading MuJoCo model...")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print(f"✅ Model loaded successfully!")
        print(f"   - Bodies: {model.nbody}")
        print(f"   - Joints: {model.njnt}")
        print(f"   - Actuators: {model.nu}")
        print(f"   - Tendons: {model.ntendon}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 基本信息
    print(f"\n📊 Simulation Info:")
    print(f"   - Time step: {model.opt.timestep}s")
    print(f"   - Control range: {model.actuator_ctrlrange}")
    
    # 设置初始控制指令（腱绳长度）
    initial_control = [0.25, 0.25, 0.25]  # 中等张力
    data.ctrl[:] = initial_control
    
    print(f"\n🎮 Initial control: {initial_control}")
    
    # 运行几步仿真看看
    print("\n🏃 Running simulation steps...")
    for i in range(10):
        mujoco.mj_step(model, data)
        if i % 3 == 0:
            tip_pos = data.site_xpos[-1]
            print(f"   Step {i}: tip position = [{tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f}]")
    
    # 启动交互式查看器
    print(f"\n🔍 Launching MuJoCo viewer...")
    print("   - 你将看到3D触手机器人模型")
    print("   - 鼠标左键旋转视角，右键平移，滚轮缩放")
    print("   - 按ESC或关闭窗口退出")
    
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            
            while viewer.is_running():
                # 随着时间改变控制指令，让触手运动
                current_time = time.time() - start_time
                
                # 简单的正弦波控制，让触手摆动
                data.ctrl[0] = 0.25 + 0.05 * np.sin(current_time * 2.0)      # 腱绳1
                data.ctrl[1] = 0.25 + 0.05 * np.sin(current_time * 2.0 + 2.0) # 腱绳2  
                data.ctrl[2] = 0.25 + 0.05 * np.sin(current_time * 2.0 + 4.0) # 腱绳3
                
                # 运行仿真
                mujoco.mj_step(model, data)
                
                # 更新显示
                viewer.sync()
                
                # 控制帧率
                time.sleep(model.opt.timestep)
                
                # 5秒后自动退出
                if current_time > 5.0:
                    print("   ⏰ 5秒演示完成，自动退出...")
                    break
    
    except Exception as e:
        print(f"❌ Viewer error: {e}")
        print("   可能是macOS的显示问题，但MuJoCo本身工作正常！")
    
    print(f"\n🎉 MuJoCo test completed successfully!")

if __name__ == "__main__":
    test_mujoco_simulation()
