#!/usr/bin/env python3
"""
从CodeAgentRLDataset.parquet中抽取100条记录生成验证数据集
"""

import pandas as pd
import os
import sys
import argparse
from datetime import datetime

# 检查必要的依赖
try:
    import pandas as pd
except ImportError:
    print("[VAL_DATASET] 错误: 需要安装pandas")
    sys.exit(1)

try:
    # 尝试导入parquet支持
    pd.read_parquet
except AttributeError:
    print("[VAL_DATASET] 错误: pandas缺少parquet支持")
    print("[VAL_DATASET] 请安装: pip install pyarrow 或 pip install fastparquet")
    sys.exit(1)

def create_validation_dataset(input_file, output_file, num_samples=100, random_seed=42):
    """
    从原始数据集中抽取指定数量的样本创建验证数据集
    
    Args:
        input_file (str): 输入parquet文件路径
        output_file (str): 输出parquet文件路径
        num_samples (int): 抽取的样本数量
        random_seed (int): 随机种子
    """
    print(f"[VAL_DATASET] 开始创建验证数据集...")
    print(f"[VAL_DATASET] 输入文件: {input_file}")
    print(f"[VAL_DATASET] 输出文件: {output_file}")
    print(f"[VAL_DATASET] 抽取样本数: {num_samples}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"[VAL_DATASET] 错误: 输入文件不存在: {input_file}")
        return False
    
    try:
        # 读取原始数据集
        print(f"[VAL_DATASET] 读取原始数据集...")
        df = pd.read_parquet(input_file)
        total_samples = len(df)
        print(f"[VAL_DATASET] 原始数据集总样本数: {total_samples}")
        
        if total_samples < num_samples:
            print(f"[VAL_DATASET] 警告: 原始数据集样本数({total_samples})少于请求的样本数({num_samples})")
            print(f"[VAL_DATASET] 将使用全部{total_samples}个样本作为验证集")
            num_samples = total_samples
        
        # 随机抽取样本
        print(f"[VAL_DATASET] 使用随机种子{random_seed}抽取{num_samples}个样本...")
        val_df = df.sample(n=num_samples, random_state=random_seed)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"[VAL_DATASET] 创建输出目录: {output_dir}")
        
        # 保存验证数据集
        print(f"[VAL_DATASET] 保存验证数据集到: {output_file}")
        val_df.to_parquet(output_file, index=False)
        
        # 验证保存的文件
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"[VAL_DATASET] ✓ 验证数据集创建成功!")
            print(f"[VAL_DATASET] 文件大小: {file_size / (1024*1024):.2f} MB")
            print(f"[VAL_DATASET] 样本数量: {len(val_df)}")
            
            # 显示数据集基本信息
            print(f"[VAL_DATASET] 数据集列信息:")
            for col in val_df.columns:
                print(f"[VAL_DATASET]   - {col}: {val_df[col].dtype}")
            
            # 显示前几个样本的统计信息
            if 'messages' in val_df.columns:
                print(f"[VAL_DATASET] 消息列统计:")
                message_lengths = []
                for idx, row in val_df.iterrows():
                    if idx >= 10:  # 只统计前10个样本
                        break
                    messages = row['messages']
                    if isinstance(messages, list):
                        total_length = sum(len(str(msg).split()) for msg in messages)
                        message_lengths.append(total_length)
                
                if message_lengths:
                    print(f"[VAL_DATASET]   前10个样本的平均消息长度: {sum(message_lengths)/len(message_lengths):.1f} tokens")
            
            return True
        else:
            print(f"[VAL_DATASET] 错误: 文件保存失败")
            return False
            
    except Exception as e:
        print(f"[VAL_DATASET] 错误: 创建验证数据集时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='从CodeAgentRLDataset.parquet中抽取样本创建验证数据集')
    parser.add_argument('--input', '-i', required=True, help='输入parquet文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出parquet文件路径')
    parser.add_argument('--num_samples', '-n', type=int, default=100, help='抽取的样本数量 (默认: 100)')
    parser.add_argument('--random_seed', '-s', type=int, default=42, help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    print(f"[VAL_DATASET] ==========================================")
    print(f"[VAL_DATASET] 验证数据集创建工具")
    print(f"[VAL_DATASET] 创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[VAL_DATASET] ==========================================")
    
    success = create_validation_dataset(
        input_file=args.input,
        output_file=args.output,
        num_samples=args.num_samples,
        random_seed=args.random_seed
    )
    
    if success:
        print(f"[VAL_DATASET] ==========================================")
        print(f"[VAL_DATASET] ✓ 验证数据集创建完成!")
        print(f"[VAL_DATASET] ==========================================")
        sys.exit(0)
    else:
        print(f"[VAL_DATASET] ==========================================")
        print(f"[VAL_DATASET] ✗ 验证数据集创建失败!")
        print(f"[VAL_DATASET] ==========================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
