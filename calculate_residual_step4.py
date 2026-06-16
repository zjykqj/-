# calculate_residual_step4.py
# Step 4：计算共性模型残差
#
# 目标：
# 1. 读取 Step 3 输出的共性预测结果
# 2. 计算 residual = target_queue - y_hat_shared
# 3. 输出带残差的数据表
# 4. 输出残差统计结果
# 5. 输出残差分布图，观察不同交叉口的个性偏差

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_required_columns(data, data_name):
    """
    检查计算残差所需字段是否存在。
    """
    required_columns = [
        "intersection_id",
        "time",
        "target_queue",
        "y_hat_shared"
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"{data_name} 缺少必要字段：{missing_columns}")


def add_residual_columns(data):
    """
    添加残差相关字段。

    residual = target_queue - y_hat_shared

    residual > 0：
        真实排队长度大于共性模型预测值，说明共性模型低估了拥堵。

    residual < 0：
        真实排队长度小于共性模型预测值，说明共性模型高估了拥堵。
    """
    data = data.copy()

    data["residual"] = data["target_queue"] - data["y_hat_shared"]
    data["residual"] = data["residual"].round(4)

    # 为了和 Step 3 的 shared_residual 保持一致，也保留这个字段
    data["shared_residual"] = data["residual"]

    # 残差方向
    data["residual_direction"] = np.where(
        data["residual"] > 0,
        "underestimate",
        np.where(data["residual"] < 0, "overestimate", "accurate")
    )

    return data


def summarize_residual(data, dataset_name):
    """
    生成残差统计结果。
    """
    residual = data["residual"]

    summary = {
        "dataset": dataset_name,
        "sample_size": len(data),
        "residual_mean": round(residual.mean(), 4),
        "residual_std": round(residual.std(), 4),
        "residual_min": round(residual.min(), 4),
        "residual_q25": round(residual.quantile(0.25), 4),
        "residual_median": round(residual.median(), 4),
        "residual_q75": round(residual.quantile(0.75), 4),
        "residual_max": round(residual.max(), 4),
        "residual_abs_mean": round(residual.abs().mean(), 4),
        "residual_rmse": round(np.sqrt(np.mean(residual ** 2)), 4)
    }

    return summary


def summarize_residual_by_intersection(data, dataset_name):
    """
    按交叉口统计残差。
    """
    summary = (
        data
        .groupby("intersection_id")["residual"]
        .agg(
            sample_size="count",
            residual_mean="mean",
            residual_std="std",
            residual_min="min",
            residual_median="median",
            residual_max="max"
        )
        .reset_index()
    )

    summary["dataset"] = dataset_name

    # 调整列顺序
    summary = summary[
        [
            "dataset",
            "intersection_id",
            "sample_size",
            "residual_mean",
            "residual_std",
            "residual_min",
            "residual_median",
            "residual_max"
        ]
    ]

    numeric_cols = [
        "residual_mean",
        "residual_std",
        "residual_min",
        "residual_median",
        "residual_max"
    ]

    summary[numeric_cols] = summary[numeric_cols].round(4)

    return summary


def plot_residual_boxplot(data, output_path, title):
    """
    绘制不同交叉口残差箱线图。
    """
    intersections = sorted(data["intersection_id"].unique())

    residual_list = [
        data[data["intersection_id"] == inter_id]["residual"].values
        for inter_id in intersections
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(residual_list, labels=intersections, showmeans=True)
    plt.axhline(y=0, linestyle="--", linewidth=1)
    plt.xlabel("Intersection ID")
    plt.ylabel("Residual = target_queue - y_hat_shared")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residual_histogram(data, output_path, title):
    """
    绘制残差直方图。
    """
    plt.figure(figsize=(8, 5))
    plt.hist(data["residual"], bins=30)
    plt.axvline(x=0, linestyle="--", linewidth=1)
    plt.xlabel("Residual = target_queue - y_hat_shared")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def calculate_residual_step4(
    source_pred_path="step3_outputs/source_with_shared_pred.csv",
    target_adapt_pred_path="step3_outputs/target_adapt_with_shared_pred.csv",
    target_test_pred_path="step3_outputs/target_test_with_shared_pred.csv",
    output_dir="step4_outputs"
):
    """
    Step 4 主函数：计算共性模型残差。

    参数
    ----
    source_pred_path:
        Step 3 输出的源域共性预测结果。

    target_adapt_pred_path:
        Step 3 输出的目标域小样本共性预测结果。

    target_test_pred_path:
        Step 3 输出的目标域测试集共性预测结果。
        注意：这里只计算并保存残差，后续 Step 5 不能用它训练模型。

    output_dir:
        Step 4 输出文件夹。
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取 Step 3 输出文件
    source_data = pd.read_csv(source_pred_path)
    target_adapt_data = pd.read_csv(target_adapt_pred_path)
    target_test_data = pd.read_csv(target_test_pred_path)

    # 2. 检查字段
    check_required_columns(source_data, "source_data")
    check_required_columns(target_adapt_data, "target_adapt_data")
    check_required_columns(target_test_data, "target_test_data")

    # 3. 计算残差
    source_with_residual = add_residual_columns(source_data)
    target_adapt_with_residual = add_residual_columns(target_adapt_data)
    target_test_with_residual = add_residual_columns(target_test_data)

    # 4. 输出带残差的数据表
    source_output_path = os.path.join(output_dir, "source_data_with_residual.csv")
    target_adapt_output_path = os.path.join(output_dir, "target_adapt_data_with_residual.csv")
    target_test_output_path = os.path.join(output_dir, "target_test_data_with_residual.csv")

    source_with_residual.to_csv(source_output_path, index=False, encoding="utf-8-sig")
    target_adapt_with_residual.to_csv(target_adapt_output_path, index=False, encoding="utf-8-sig")
    target_test_with_residual.to_csv(target_test_output_path, index=False, encoding="utf-8-sig")

    # 5. 生成整体残差统计
    residual_summary = pd.DataFrame([
        summarize_residual(source_with_residual, "source_A_B_C_D"),
        summarize_residual(target_adapt_with_residual, "target_T_adapt"),
        summarize_residual(target_test_with_residual, "target_T_test")
    ])

    residual_summary_path = os.path.join(output_dir, "residual_summary.csv")
    residual_summary.to_csv(residual_summary_path, index=False, encoding="utf-8-sig")

    # 6. 生成按交叉口残差统计
    residual_by_intersection = pd.concat(
        [
            summarize_residual_by_intersection(source_with_residual, "source_A_B_C_D"),
            summarize_residual_by_intersection(target_adapt_with_residual, "target_T_adapt"),
            summarize_residual_by_intersection(target_test_with_residual, "target_T_test")
        ],
        ignore_index=True
    )

    residual_by_intersection_path = os.path.join(
        output_dir,
        "residual_by_intersection.csv"
    )

    residual_by_intersection.to_csv(
        residual_by_intersection_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 7. 画残差分布图
    source_boxplot_path = os.path.join(
        output_dir,
        "source_residual_boxplot_by_intersection.png"
    )

    source_hist_path = os.path.join(
        output_dir,
        "source_residual_histogram.png"
    )

    target_adapt_hist_path = os.path.join(
        output_dir,
        "target_adapt_residual_histogram.png"
    )

    target_test_hist_path = os.path.join(
        output_dir,
        "target_test_residual_histogram.png"
    )

    plot_residual_boxplot(
        source_with_residual,
        source_boxplot_path,
        "Source Domain Residuals by Intersection"
    )

    plot_residual_histogram(
        source_with_residual,
        source_hist_path,
        "Source Domain Residual Distribution"
    )

    plot_residual_histogram(
        target_adapt_with_residual,
        target_adapt_hist_path,
        "Target Adaptation Residual Distribution"
    )

    plot_residual_histogram(
        target_test_with_residual,
        target_test_hist_path,
        "Target Test Residual Distribution"
    )

    # 8. 打印结果
    print("Step 4 残差计算完成。")
    print("-" * 70)

    print("输入文件：")
    print(f"  源域共性预测结果：{source_pred_path}")
    print(f"  目标适配集共性预测结果：{target_adapt_pred_path}")
    print(f"  目标测试集共性预测结果：{target_test_pred_path}")
    print()

    print("残差定义：")
    print("  residual = target_queue - y_hat_shared")
    print()

    print("残差解释：")
    print("  residual > 0：共性模型低估真实排队长度，该路口或该情景更拥堵")
    print("  residual < 0：共性模型高估真实排队长度，该路口或该情景更通畅")
    print()

    print("整体残差统计：")
    print(residual_summary)
    print()

    print("按交叉口残差统计：")
    print(residual_by_intersection)
    print()

    print("输出文件：")
    print(f"  源域残差数据：{source_output_path}")
    print(f"  目标适配集残差数据：{target_adapt_output_path}")
    print(f"  目标测试集残差数据：{target_test_output_path}")
    print(f"  整体残差统计：{residual_summary_path}")
    print(f"  分交叉口残差统计：{residual_by_intersection_path}")
    print(f"  源域箱线图：{source_boxplot_path}")
    print(f"  源域直方图：{source_hist_path}")
    print(f"  目标适配集直方图：{target_adapt_hist_path}")
    print(f"  目标测试集直方图：{target_test_hist_path}")

    return (
        source_with_residual,
        target_adapt_with_residual,
        target_test_with_residual,
        residual_summary,
        residual_by_intersection
    )


if __name__ == "__main__":
    calculate_residual_step4(
        source_pred_path="step3_outputs/source_with_shared_pred.csv",
        target_adapt_pred_path="step3_outputs/target_adapt_with_shared_pred.csv",
        target_test_pred_path="step3_outputs/target_test_with_shared_pred.csv",
        output_dir="step4_outputs"
    )