#!/usr/bin/env python3
"""列出所有可用的训练好的RL模型"""

from pathlib import Path
import os
from datetime import datetime

def list_available_models():
    """列出所有可用的训练好的模型"""
    
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("❌ 结果目录不存在: results/")
        print("   请先运行训练: python -m shoggoth_mini.training.rl.training train")
        return []
    
    # 查找所有训练运行
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("ppo_tentacle_")]
    
    if not run_dirs:
        print("❌ 没有找到任何训练运行")
        print("   请先运行训练: python -m shoggoth_mini.training.rl.training train")
        return []
    
    print(f"🔍 找到 {len(run_dirs)} 个训练运行:")
    print("=" * 80)
    
    available_models = []
    
    for run_dir in sorted(run_dirs, key=lambda d: d.name, reverse=True):
        models_dir = run_dir / "models"
        
        if not models_dir.exists():
            continue
            
        print(f"\n📁 训练运行: {run_dir.name}")
        
        # 解析时间戳
        try:
            timestamp_str = run_dir.name.split("_")[-2] + "_" + run_dir.name.split("_")[-1]
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            print(f"   ⏰ 训练时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            print(f"   ⏰ 训练时间: 解析失败")
        
        # 查找不同类型的模型
        model_files = []
        
        # 最佳模型
        best_model_path = models_dir / "best_model.zip"
        if best_model_path.exists():
            size_mb = best_model_path.stat().st_size / (1024 * 1024)
            model_files.append({
                "type": "最佳模型",
                "path": best_model_path,
                "size": f"{size_mb:.1f}MB",
                "recommended": True
            })
        
        # 最终模型
        final_model_path = models_dir / "final_model.zip"
        if final_model_path.exists():
            size_mb = final_model_path.stat().st_size / (1024 * 1024)
            model_files.append({
                "type": "最终模型",
                "path": final_model_path,
                "size": f"{size_mb:.1f}MB",
                "recommended": False
            })
        
        # 检查点模型
        checkpoint_files = list(models_dir.glob("tentacle_model_*_steps.zip"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.stem.split("_")[-2]))
            size_mb = latest_checkpoint.stat().st_size / (1024 * 1024)
            steps = latest_checkpoint.stem.split("_")[-2]
            model_files.append({
                "type": f"检查点 ({steps} steps)",
                "path": latest_checkpoint,
                "size": f"{size_mb:.1f}MB",
                "recommended": False
            })
        
        if model_files:
            for model_info in model_files:
                status = " ⭐ (推荐)" if model_info["recommended"] else ""
                print(f"   📄 {model_info['type']}: {model_info['path']} ({model_info['size']}){status}")
                available_models.append(model_info)
        else:
            print(f"   ❌ 没有找到模型文件")
        
        # 显示日志信息
        logs_dir = run_dir / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.csv"))
            if log_files:
                print(f"   📊 训练日志: {len(log_files)} 个文件")
        
        # 显示配置文件
        config_file = run_dir / "config.yaml"
        if config_file.exists():
            print(f"   ⚙️ 配置文件: {config_file}")
    
    print("\n" + "=" * 80)
    print(f"📋 总结: 找到 {len(available_models)} 个可用模型")
    
    return available_models

def main():
    """主函数"""
    print("📚 训练好的RL模型列表")
    print("=" * 50)
    
    models = list_available_models()
    
    if models:
        print(f"\n🚀 使用方法:")
        print(f"   # 自动使用最新的最佳模型")
        print(f"   python load_and_test_model.py")
        print(f"   ")
        print(f"   # 指定特定模型")
        print(f"   python load_and_test_model.py results/ppo_tentacle_YYYYMMDD_HHMMSS/models/best_model.zip")
        print(f"   ")
        print(f"   # 使用官方评估脚本")
        print(f"   python -m shoggoth_mini.training.rl.evaluation results/.../models/best_model.zip --render")

if __name__ == "__main__":
    main()
