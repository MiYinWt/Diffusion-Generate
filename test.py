# scripts/compute_vina_stats.py
import os
import numpy as np

root = 'outputs/sample_20251117_174936'  # 根据需要修改路径
vina_vals = []

for name in os.listdir(root):
    if not name.endswith('_SDF'):
        continue
    log_path = os.path.join(root, name, 'log.txt')
    if not os.path.isfile(log_path):
        continue
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        continue

    # 找到 header 行并确定 vina 列索引
    header_idx = 0
    header = lines[header_idx]
    cols = [c.strip().rstrip(':').lower() for c in header.split(',')]
    if 'vina' in cols:
        vina_idx = cols.index('vina')
        data_lines = lines[header_idx+1:]
    else:
        # 如果第一行不是 header，尝试第一行包含 'vina' 的行，或直接取每行最后一列
        vina_idx = None
        data_lines = lines

    for ln in data_lines:
        parts = [p.strip() for p in ln.split(',')]
        try:
            if vina_idx is not None and vina_idx < len(parts):
                val = float(parts[vina_idx])
            else:
                # 退化策略：取最后一列作为 vina 值
                val = float(parts[-1])
            vina_vals.append(val)
        except Exception:
            # 跳过无法解析的行
            continue

if len(vina_vals) == 0:
    print("未找到任何 vina 数据。")
else:
    arr = np.array(vina_vals)
    print(f"样本数: {len(arr)}")
    print(f"平均值 (mean): {arr.mean():.4f}")
    print(f"中位数 (median): {np.median(arr):.4f}")