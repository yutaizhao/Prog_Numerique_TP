#!/usr/bin/env python3
import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_results(file_path):
    """
    解析结果文件，返回字典：
      key: (N, gamma)
      value: [(x, u), (x, u), ...] 数据点列表
    """
    results = {}
    current_key = None
    current_data = []
    header_pattern = re.compile(r"Final solution for N=(\d+), gamma=([\d\.]+):")
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            header_match = header_pattern.match(line)
            if header_match:
                # 保存前一块数据
                if current_key is not None and current_data:
                    results[current_key] = current_data
                N = int(header_match.group(1))
                gamma = float(header_match.group(2))
                current_key = (N, gamma)
                current_data = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        u = float(parts[1])
                        current_data.append((x, u))
                    except ValueError:
                        continue
    if current_key is not None and current_data:
        results[current_key] = current_data
    return results

def plot_by_gamma(results, title_prefix, save_folder, file_tag):
    """
    按照 gamma 分组，每个 gamma 值生成一幅图：
      - results: key为 (N, gamma) 的字典
      - title_prefix: 用于图标题的前缀，例如 "Implicit, Set 1"
      - save_folder: 保存路径，例如 "results/implicit"
      - file_tag: 文件名标识，如 "implicit_set1"
    """
    # 以 gamma 为键进行分组
    gamma_groups = defaultdict(list)
    for (N, gamma), data in results.items():
        gamma_groups[gamma].append((N, data))
    
    # 对每个 gamma 绘图
    for gamma, group in gamma_groups.items():
        plt.figure(figsize=(8,6))
        # 按 N 值排序
        group.sort(key=lambda tup: tup[0])
        for N, data in group:
            xs, us = zip(*data)
            plt.plot(xs, us, marker='o', label=f"N={N}")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"{title_prefix}: Final Solutions (gamma={gamma})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(save_folder, f"solution_plot_{file_tag}_gamma{gamma}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")

# 处理各个结果文件
files = [
    ("results/implicit/implicit_set1.txt", "Implicit, Set 1", "results/implicit", "implicit_set1"),
    ("results/implicit/implicit_set2.txt", "Implicit, Set 2", "results/implicit", "implicit_set2"),
    ("results/newton/newton_set1.txt", "Newton, Set 1", "results/newton", "newton_set1"),
    ("results/newton/newton_set2.txt", "Newton, Set 2", "results/newton", "newton_set2")
]

for file_path, title_prefix, save_folder, file_tag in files:
    if os.path.exists(file_path):
        results = parse_results(file_path)
        plot_by_gamma(results, title_prefix, save_folder, file_tag)
    else:
        print(f"{file_path} not found!")
