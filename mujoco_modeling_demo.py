#!/usr/bin/env python3
"""详细解释MuJoCo中触手机器人的建模方式"""

import mujoco
import numpy as np

def analyze_tentacle_model():
    """分析触手模型的构成"""
    print("🎨 MuJoCo触手机器人建模解析")
    print("=" * 50)
    
    # 加载模型
    model_path = "assets/simulation/tentacle.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"📊 模型基本信息:")
    print(f"   - 模型文件: {model_path}")
    print(f"   - 刚体数量: {model.nbody}")
    print(f"   - 关节数量: {model.njnt}")
    print(f"   - 几何体数量: {model.ngeom}")
    print(f"   - 腱绳数量: {model.ntendon}")
    print(f"   - 执行器数量: {model.nu}")
    print(f"   - 锚点数量: {model.nsite}")
    
    return model, data

def explain_body_structure(model):
    """解释刚体结构"""
    print(f"\n🦴 刚体链结构分析:")
    print("=" * 30)
    
    print("触手由21个刚体组成，形成串联的运动链：")
    print()
    
    # 分析前几个刚体
    for i in range(min(5, model.nbody)):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            print(f"   Body {i}: {body_name}")
            # 查找与此刚体关联的几何体
            for j in range(model.ngeom):
                if model.geom_bodyid[j] == i:
                    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, j)
                    geom_type = model.geom_type[j]
                    print(f"     └─ 几何体: {geom_name} (类型: {geom_type})")
    
    print("   ...")
    print(f"   Body {model.nbody-1}: body{model.nbody-1}")
    print()
    print("💡 每个刚体都有对应的STL网格文件 (Part1.stl ~ Part21.stl)")
    print("   STL文件定义了触手节段的精确3D形状")

def explain_joint_system(model):
    """解释关节系统"""
    print(f"\n🔗 关节系统分析:")
    print("=" * 30)
    
    print("触手使用球形关节连接各个节段：")
    print()
    
    # 分析前几个关节
    for i in range(min(5, model.njnt)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = model.jnt_type[i]
        joint_pos = model.jnt_pos[i]
        stiffness = model.jnt_stiffness[i] if i < len(model.jnt_stiffness) else 0
        damping = model.dof_damping[i] if i < len(model.dof_damping) else 0
        
        print(f"   关节 {i+1}: {joint_name}")
        print(f"     └─ 类型: 球形关节 (3DOF)")
        print(f"     └─ 位置: [0, 0, {joint_pos[2]:.4f}]m")
        print(f"     └─ 刚度: {stiffness:.5f}")
        print(f"     └─ 阻尼: {damping:.7f}")
        print()
    
    print("   ...")
    print(f"   总计: {model.njnt} 个球形关节")
    print()
    print("💡 特点:")
    print("   - 每个关节有3个自由度 (DOF)")
    print("   - 刚度和阻尼沿触手逐渐减小 (模拟生物特性)")
    print("   - 关节位置精确计算，模拟真实几何")

def explain_tendon_system(model, data):
    """解释腱绳系统"""
    print(f"\n🕷️ 腱绳驱动系统:")
    print("=" * 30)
    
    print("触手使用3条空间腱绳进行驱动：")
    print()
    
    for i in range(model.ntendon):
        tendon_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
        tendon_length = data.ten_length[i]
        print(f"   腱绳 {i+1}: {tendon_name}")
        print(f"     └─ 当前长度: {tendon_length:.4f}m")
        
        print(f"     └─ 锚点数量: ~42 (每条腱绳)")
        print()
    
    print("💡 腱绳设计:")
    print("   - 每条腱绳通过21个锚点对 (42个锚点)")
    print("   - 3条腱绳按120度角度分布")
    print("   - 通过改变腱绳长度控制触手弯曲")

def explain_geometric_details(model):
    """解释几何细节"""
    print(f"\n📐 几何设计细节:")
    print("=" * 30)
    
    print("🎯 锚点设计 (前几个示例):")
    # 查找site信息
    for i in range(min(6, model.nsite)):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if site_name and "site_in_0_" in site_name:
            site_pos = model.site_pos[i]
            print(f"   {site_name}: [{site_pos[0]:.5f}, {site_pos[1]:.5f}, {site_pos[2]:.5f}]")
    
    print(f"\n📏 尺寸信息:")
    print(f"   - STL缩放比例: 0.001 (毫米 → 米)")
    print(f"   - 触手总长度: ~23cm")
    print(f"   - 基部直径: ~3.4mm")
    print(f"   - 尖端直径: ~1.5mm")
    
    print(f"\n🎨 视觉元素:")
    print(f"   - 腱绳颜色: 红色 (1 0 0 1)")
    print(f"   - 锚点颜色: 黄色 (1 1 0 1)")
    print(f"   - 目标球颜色: 红色半透明 (1 0 0 0.8)")
    print(f"   - 触手尖端标记: 绿色 (0 1 0 1)")

def explain_physics_properties(model):
    """解释物理属性"""
    print(f"\n⚖️ 物理属性设置:")
    print("=" * 30)
    
    print("🔧 仿真设置:")
    print(f"   - 时间步长: {model.opt.timestep}s")
    print(f"   - 求解器迭代: 50次")
    print(f"   - 角度单位: 弧度")
    
    print(f"\n🏗️ 材料特性:")
    print("   - 刚度: 沿触手递减 (基部→尖端)")
    print("   - 阻尼: 与刚度成正比")
    print("   - 摩擦: 碰撞检测关闭")
    print("   - 质量: 从STL几何自动计算")
    
    print(f"\n⚙️ 执行器参数:")
    print("   - 控制类型: 位置控制")
    print("   - 力范围: -200N ~ 0N (只能收缩)")
    print("   - 长度范围: 0.120m ~ 0.340m")
    print("   - 增益: 200.0")

def show_modeling_workflow():
    """展示建模工作流程"""
    print(f"\n🛠️ MuJoCo建模工作流程:")
    print("=" * 30)
    
    print("1️⃣ 3D几何模型设计:")
    print("   - 使用CAD软件设计21个触手节段")
    print("   - 导出为STL格式 (Part1.stl ~ Part21.stl)")
    print("   - 每个节段包含腱绳通道和连接点")
    
    print(f"\n2️⃣ XML模型定义:")
    print("   - 定义刚体链结构 (body1 → body2 → ... → body21)")
    print("   - 设置球形关节连接")
    print("   - 计算精确的关节位置和物理参数")
    
    print(f"\n3️⃣ 腱绳系统设计:")
    print("   - 计算42个锚点的3D坐标")
    print("   - 定义3条空间腱绳路径")
    print("   - 设置执行器控制参数")
    
    print(f"\n4️⃣ 物理参数调优:")
    print("   - 根据材料属性设定刚度/阻尼")
    print("   - 调整时间步长和求解器参数")
    print("   - 验证仿真稳定性")

def compare_modeling_approaches():
    """对比不同的建模方法"""
    print(f"\n🔄 建模方法对比:")
    print("=" * 30)
    
    print("🎯 MuJoCo触手建模 (当前方法):")
    print("   ✅ 优点:")
    print("     - 高精度3D几何 (STL网格)")
    print("     - 真实物理特性 (刚度梯度)")
    print("     - 精确腱绳路径 (42个锚点)")
    print("     - 稳定的仿真性能")
    print("   ⚠️ 复杂度:")
    print("     - 需要精确的几何计算")
    print("     - XML文件较大 (29KB)")
    print("     - 参数调优工作量大")
    
    print(f"\n🔧 简化建模方法 (替代方案):")
    print("   - 基本几何体组合 (圆柱+球体)")
    print("   - 简化的腱绳模型")
    print("   - 均匀的物理参数")
    print("   💭 适用: 概念验证、快速原型")

if __name__ == "__main__":
    model, data = analyze_tentacle_model()
    explain_body_structure(model)
    explain_joint_system(model)
    explain_tendon_system(model, data)
    explain_geometric_details(model)
    explain_physics_properties(model)
    show_modeling_workflow()
    compare_modeling_approaches()
    
    print(f"\n🎊 总结:")
    print("   MuJoCo触手模型是一个精密的工程作品，")
    print("   结合了精确的3D几何、真实的物理特性和复杂的控制系统！")
