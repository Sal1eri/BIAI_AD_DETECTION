import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- 配置参数 ---
# 获取当前脚本文件所在的目录（即 utils 目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 向上移动一级到项目根目录，然后找到 dataset
# 假设数据集在项目根目录下的 /dataset 文件夹
PROJECT_ROOT = os.path.dirname(BASE_DIR) 

# --- 修改路径定义 ---
CSV_PATH = os.path.join(PROJECT_ROOT, "dataset", "5_class_10_12_2025.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "configs", "split_results")


TARGET_COLUMN = 'Group' 

# 比例定义：70% 训练, 10% 验证, 20% 测试
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

def perform_stratified_split_7_1_2(csv_path, target_column, output_dir):
    """
    加载数据，进行分层抽样分割 (7:1:2)，并保存结果。
    """
    print(f"--- 1. 正在加载数据: {csv_path} ---")
    try:
        # 假设数据格式与您提供的图片一致
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_path}。请检查路径。")
        return
    
    if target_column not in df.columns:
        print(f"错误: 找不到目标列 '{target_column}'。请确认列名是否正确。")
        return
        
    N = len(df)
    print(f"总样本数: {N}")

    # ----------------------------------------------------
    # 2. 第一次分割：分离测试集 (20%)
    # ----------------------------------------------------
    # stratify=df[target_column] 确保测试集中各类别比例均匀
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,  # 0.20 for testing
        random_state=42,       # 确保可复现性
        stratify=df[target_column]
    )
    
    # ----------------------------------------------------
    # 3. 第二次分割：分离验证集 (10%) 和训练集 (70%)
    # ----------------------------------------------------
    # 训练/验证池 (train_val_df) 占总体的 80% (1.0 - 0.20)
    # 我们需要验证集占总体的 10%。
    # 因此，验证集占 train_val_df 的比例是 0.10 / 0.80 = 0.125
    
    val_split_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)  # 0.10 / 0.80 = 0.125
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_split_ratio, # 0.125 for validation split
        random_state=42,
        stratify=train_val_df[target_column]
    )
    
    print("\n--- 2. 分割结果摘要 ---")
    print(f"训练集样本数: {len(train_df)} ({len(train_df)/N:.1%})")  # 应接近 70%
    print(f"验证集样本数: {len(val_df)} ({len(val_df)/N:.1%})")      # 应接近 10%
    print(f"测试集样本数: {len(test_df)} ({len(test_df)/N:.1%})")      # 应接近 20%
    print("-" * 30)

    # 4. 验证类别分布（确保各类别均匀分配）
    print("\n--- 3. 类别分布验证 (基于 'Group') ---")
    
    def get_distribution(data_df):
        # 计算各类别样本占总体的百分比，并保留两位小数
        return data_df[target_column].value_counts(normalize=True).sort_index().mul(100).round(2)

    distribution_df = pd.DataFrame({
        'Train %': get_distribution(train_df),
        'Validation %': get_distribution(val_df),
        'Test %': get_distribution(test_df)
    })
    
    print(distribution_df)
    
    
    # 5. 保存分割后的数据集
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, 'train_split_70.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_split_10.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split_20.csv'), index=False)
    
    print(f"\n--- 4. 分割结果已保存至目录: {output_dir} 中的三个 CSV 文件 ---")
    
# --- 运行脚本 ---
# 注意: 在运行前，请确保将 CSV_PATH 修改为您的实际文件路径，并确认列名 'Group' 正确。


if __name__ == "__main__":
    # ai生成的脚本分割 我感觉大部分没问题 先这么用吧
    perform_stratified_split_7_1_2(
    csv_path=CSV_PATH, 
    target_column=TARGET_COLUMN,
    output_dir=OUTPUT_DIR
)