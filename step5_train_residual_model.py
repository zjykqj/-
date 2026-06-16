# train_residual_model_step5.py
# Step 5：训练目标路口残差修正模型
#
# 目标：
# 1. 读取 Step 4 输出的目标域适配集残差数据
# 2. 用 target_adapt_data_with_residual.csv 训练残差模型 g_T
# 3. 在目标测试集上预测 residual_hat
# 4. 最终预测：
#       y_hat_final = y_hat_shared + residual_hat
# 5. 比较直接迁移模型和残差修正模型的误差

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred, dataset_name, model_name):
    """
    计算回归预测误差。
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }


def add_final_prediction(data, residual_model, residual_features):
    """
    使用残差模型生成 residual_hat，并计算最终预测 y_hat_final。

    y_hat_final = y_hat_shared + residual_hat
    """
    data = data.copy()

    X_res = data[residual_features]

    residual_hat = residual_model.predict(X_res)

    data["residual_hat"] = np.round(residual_hat, 4)
    data["y_hat_final"] = data["y_hat_shared"] + data["residual_hat"]

    # 排队长度不能为负
    data["y_hat_final"] = data["y_hat_final"].clip(lower=0)
    data["y_hat_final"] = data["y_hat_final"].round(4)

    # 误差
    data["shared_error"] = data["target_queue"] - data["y_hat_shared"]
    data["final_error"] = data["target_queue"] - data["y_hat_final"]

    data["shared_abs_error"] = data["shared_error"].abs()
    data["final_abs_error"] = data["final_error"].abs()

    data["shared_error"] = data["shared_error"].round(4)
    data["final_error"] = data["final_error"].round(4)
    data["shared_abs_error"] = data["shared_abs_error"].round(4)
    data["final_abs_error"] = data["final_abs_error"].round(4)

    return data


def plot_prediction_comparison(data, output_path, title, max_points=200):
    """
    绘制真实值、共性预测值、残差修正后预测值对比图。
    为了图不太密，默认只画前 max_points 个点。
    """
    plot_data = data.head(max_points).copy()

    plt.figure(figsize=(12, 5))
    plt.plot(plot_data["target_queue"].values, label="True target_queue", linewidth=2)
    plt.plot(plot_data["y_hat_shared"].values, label="Shared prediction", linestyle="--")
    plt.plot(plot_data["y_hat_final"].values, label="Final prediction", linestyle="-.")

    plt.xlabel("Sample index")
    plt.ylabel("Queue length")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_error_comparison(metrics_df, output_path):
    """
    绘制 MAE 对比柱状图。
    """
    test_metrics = metrics_df[metrics_df["dataset"] == "target_test_T"].copy()

    plt.figure(figsize=(7, 5))
    plt.bar(test_metrics["model"], test_metrics["MAE"])
    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title("Target Test MAE Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_residual_model_step5(
    target_adapt_path="step4_outputs/target_adapt_data_with_residual.csv",
    target_test_path="step4_outputs/target_test_data_with_residual.csv",
    output_dir="step5_outputs"
):
    """
    Step 5 主函数：训练目标路口 T 的残差修正模型。

    参数
    ----
    target_adapt_path:
        Step 4 输出的目标域小样本残差数据。
        用于训练残差模型 g_T。

    target_test_path:
        Step 4 输出的目标域测试集残差数据。
        只用于最终测试，不能参与训练。

    output_dir:
        Step 5 输出文件夹。
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    target_adapt_data = pd.read_csv(target_adapt_path)
    target_test_data = pd.read_csv(target_test_path)

    # 2. 残差模型输入特征
    # 小样本情况下不要放太多变量，避免过拟合
    residual_features = [
        "flow",
        "speed",
        "occupancy",
        "queue_length",
        "green_time",
        "lane_num",
        "is_peak"
    ]

    residual_target = "residual"

    required_columns = residual_features + [
        "target_queue",
        "y_hat_shared",
        residual_target
    ]

    # 3. 检查字段
    for name, df in [
        ("target_adapt_data", target_adapt_data),
        ("target_test_data", target_test_data)
    ]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{name} 缺少必要字段：{missing_columns}")

    # 4. 构造残差模型训练数据
    X_res_train = target_adapt_data[residual_features]
    y_res_train = target_adapt_data[residual_target]

    # 5. 定义残差修正模型 g_T
    # Ridge 比普通线性回归更稳，适合 20/50/100 这种小样本
    g_T = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0))
        ]
    )

    # 6. 训练残差模型
    g_T.fit(X_res_train, y_res_train)

    # 7. 在目标适配集和目标测试集上生成最终预测
    target_adapt_final = add_final_prediction(
        target_adapt_data,
        g_T,
        residual_features
    )

    target_test_final = add_final_prediction(
        target_test_data,
        g_T,
        residual_features
    )

    # 8. 计算误差：直接迁移 vs 残差修正
    metrics = []

    # 目标适配集上的误差
    metrics.append(
        evaluate_regression(
            target_adapt_final["target_queue"],
            target_adapt_final["y_hat_shared"],
            "target_adapt_T",
            "shared_direct_transfer"
        )
    )

    metrics.append(
        evaluate_regression(
            target_adapt_final["target_queue"],
            target_adapt_final["y_hat_final"],
            "target_adapt_T",
            "shared_plus_residual"
        )
    )

    # 目标测试集上的误差
    metrics.append(
        evaluate_regression(
            target_test_final["target_queue"],
            target_test_final["y_hat_shared"],
            "target_test_T",
            "shared_direct_transfer"
        )
    )

    metrics.append(
        evaluate_regression(
            target_test_final["target_queue"],
            target_test_final["y_hat_final"],
            "target_test_T",
            "shared_plus_residual"
        )
    )

    metrics_df = pd.DataFrame(metrics)

    # 9. 保存模型、配置和结果
    residual_model_path = os.path.join(output_dir, "target_residual_model.pkl")
    feature_config_path = os.path.join(output_dir, "residual_model_features.json")
    metrics_path = os.path.join(output_dir, "final_prediction_metrics.csv")

    adapt_final_path = os.path.join(output_dir, "target_adapt_final_prediction.csv")
    test_final_path = os.path.join(output_dir, "target_test_final_prediction.csv")

    prediction_plot_path = os.path.join(output_dir, "target_test_prediction_comparison.png")
    error_plot_path = os.path.join(output_dir, "target_test_mae_comparison.png")

    joblib.dump(g_T, residual_model_path)

    with open(feature_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "residual_features": residual_features,
                "residual_target": residual_target,
                "final_prediction_formula": "y_hat_final = y_hat_shared + residual_hat",
                "model": "StandardScaler + Ridge(alpha=10.0)"
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    target_adapt_final.to_csv(adapt_final_path, index=False, encoding="utf-8-sig")
    target_test_final.to_csv(test_final_path, index=False, encoding="utf-8-sig")

    # 10. 画图
    plot_prediction_comparison(
        target_test_final,
        prediction_plot_path,
        "Target Test Prediction: Shared vs Shared + Residual",
        max_points=200
    )

    plot_error_comparison(
        metrics_df,
        error_plot_path
    )

    # 11. 打印结果
    print("Step 5 目标路口残差修正模型训练完成。")
    print("-" * 70)

    print("输入文件：")
    print(f"  目标适配集残差数据：{target_adapt_path}")
    print(f"  目标测试集残差数据：{target_test_path}")
    print()

    print("残差模型输入特征：")
    for col in residual_features:
        print(f"  - {col}")
    print()

    print("残差模型输出：")
    print("  residual_hat")
    print()

    print("最终预测公式：")
    print("  y_hat_final = y_hat_shared + residual_hat")
    print()

    print("误差对比：")
    print(metrics_df)
    print()

    print("输出文件：")
    print(f"  残差模型：{residual_model_path}")
    print(f"  特征配置：{feature_config_path}")
    print(f"  误差对比表：{metrics_path}")
    print(f"  目标适配集最终预测：{adapt_final_path}")
    print(f"  目标测试集最终预测：{test_final_path}")
    print(f"  预测曲线图：{prediction_plot_path}")
    print(f"  MAE 对比图：{error_plot_path}")

    return g_T, metrics_df, target_adapt_final, target_test_final


if __name__ == "__main__":
    train_residual_model_step5(
        target_adapt_path="step4_outputs/target_adapt_data_with_residual.csv",
        target_test_path="step4_outputs/target_test_data_with_residual.csv",
        output_dir="step5_outputs"
    )
