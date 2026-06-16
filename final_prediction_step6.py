# final_prediction_step6.py
# Step 6：目标交叉口小样本适配与最终预测
#
# 目标：
# 1. 构造三种模型对比：
#    M1：目标小样本直接训练
#    M2：源域共性模型直接迁移
#    M3：共性模型 + 目标残差修正
# 2. 在目标测试集 target_test 上比较 MAE、RMSE、R2
# 3. 输出 prediction_compare.csv、error_compare.csv、误差对比图和预测曲线图

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


def evaluate_regression(y_true, y_pred, model_name):
    """
    计算回归误差指标。
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }


def train_target_only_model(target_adapt_data, target_test_data, features):
    """
    M1：只用目标交叉口少量样本直接训练预测模型。

    这个模型不使用源交叉口 A/B/C/D 的数据，
    只用 T 路口的 target_adapt_data 训练，
    然后在 target_test_data 上测试。
    """

    X_train = target_adapt_data[features]
    y_train = target_adapt_data["target_queue"]

    X_test = target_test_data[features]

    # 小样本下使用 Ridge，比普通线性回归更稳
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0))
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 排队长度不能为负
    y_pred = np.maximum(0, y_pred)

    return model, y_pred


def plot_error_comparison(error_df, output_path):
    """
    绘制 M1、M2、M3 的 MAE 和 RMSE 对比图。
    """
    model_names = error_df["model"].tolist()
    mae_values = error_df["MAE"].tolist()
    rmse_values = error_df["RMSE"].tolist()

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, mae_values, width, label="MAE")
    plt.bar(x + width / 2, rmse_values, width, label="RMSE")

    plt.xticks(x, model_names, rotation=15)
    plt.ylabel("Error")
    plt.title("Target Test Error Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_prediction_curve(prediction_df, output_path, max_points=200):
    """
    绘制目标测试集预测曲线。
    默认只画前 max_points 个点，避免曲线太密。
    """
    plot_data = prediction_df.head(max_points).copy()

    plt.figure(figsize=(12, 5))

    plt.plot(
        plot_data["target_queue"].values,
        label="True target_queue",
        linewidth=2
    )

    plt.plot(
        plot_data["y_hat_M1_target_only"].values,
        label="M1 target only",
        linestyle="--"
    )

    plt.plot(
        plot_data["y_hat_M2_shared"].values,
        label="M2 shared direct transfer",
        linestyle="-."
    )

    plt.plot(
        plot_data["y_hat_M3_shared_plus_residual"].values,
        label="M3 shared + residual",
        linestyle=":"
    )

    plt.xlabel("Sample index")
    plt.ylabel("Queue length")
    plt.title("Target Test Prediction Curve: M1 vs M2 vs M3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def final_prediction_step6(
    target_adapt_path="step5_outputs/target_adapt_final_prediction.csv",
    target_test_path="step5_outputs/target_test_final_prediction.csv",
    output_dir="step6_outputs"
):
    """
    Step 6 主函数：目标交叉口最终预测与模型对比。

    输入
    ----
    target_adapt_path:
        Step 5 输出的目标适配集最终预测文件。
        用于训练 M1 目标小样本直接模型。

    target_test_path:
        Step 5 输出的目标测试集最终预测文件。
        其中已经包含：
        - y_hat_shared：M2 共性模型直接迁移预测
        - y_hat_final：M3 共性模型 + 残差修正预测

    输出
    ----
    step6_outputs/
        target_only_model.pkl
        prediction_compare.csv
        error_compare.csv
        step6_config.json
        error_compare.png
        target_prediction_curve.png
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取 Step 5 输出
    target_adapt_data = pd.read_csv(target_adapt_path)
    target_test_data = pd.read_csv(target_test_path)

    # 2. M1 使用的特征
    # 小样本直接训练模型不放太多变量，避免过拟合
    target_only_features = [
        "flow",
        "speed",
        "occupancy",
        "queue_length",
        "green_time",
        "cycle_length",
        "lane_num",
        "is_peak"
    ]

    required_columns = target_only_features + [
        "target_queue",
        "y_hat_shared",
        "y_hat_final"
    ]

    for name, df in [
        ("target_adapt_data", target_adapt_data),
        ("target_test_data", target_test_data)
    ]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{name} 缺少必要字段：{missing_columns}")

    # 3. 训练 M1：目标小样本直接训练模型
    target_only_model, y_hat_M1 = train_target_only_model(
        target_adapt_data=target_adapt_data,
        target_test_data=target_test_data,
        features=target_only_features
    )

    # 4. 读取 M2、M3 的预测结果
    y_true = target_test_data["target_queue"].values

    # M2：共性模型直接迁移
    y_hat_M2 = target_test_data["y_hat_shared"].values

    # M3：共性模型 + 残差修正
    y_hat_M3 = target_test_data["y_hat_final"].values

    # 5. 构造预测对比表
    prediction_compare = target_test_data.copy()

    prediction_compare["y_hat_M1_target_only"] = np.round(y_hat_M1, 4)
    prediction_compare["y_hat_M2_shared"] = np.round(y_hat_M2, 4)
    prediction_compare["y_hat_M3_shared_plus_residual"] = np.round(y_hat_M3, 4)

    prediction_compare["error_M1"] = np.round(
        prediction_compare["target_queue"] - prediction_compare["y_hat_M1_target_only"],
        4
    )

    prediction_compare["error_M2"] = np.round(
        prediction_compare["target_queue"] - prediction_compare["y_hat_M2_shared"],
        4
    )

    prediction_compare["error_M3"] = np.round(
        prediction_compare["target_queue"] - prediction_compare["y_hat_M3_shared_plus_residual"],
        4
    )

    prediction_compare["abs_error_M1"] = prediction_compare["error_M1"].abs()
    prediction_compare["abs_error_M2"] = prediction_compare["error_M2"].abs()
    prediction_compare["abs_error_M3"] = prediction_compare["error_M3"].abs()

    # 6. 计算三种模型误差
    error_rows = [
        evaluate_regression(
            y_true,
            y_hat_M1,
            "M1_target_only"
        ),
        evaluate_regression(
            y_true,
            y_hat_M2,
            "M2_shared_direct_transfer"
        ),
        evaluate_regression(
            y_true,
            y_hat_M3,
            "M3_shared_plus_residual"
        )
    ]

    error_compare = pd.DataFrame(error_rows)

    # 7. 保存结果
    model_path = os.path.join(output_dir, "target_only_model.pkl")
    prediction_path = os.path.join(output_dir, "prediction_compare.csv")
    error_path = os.path.join(output_dir, "error_compare.csv")
    config_path = os.path.join(output_dir, "step6_config.json")

    error_plot_path = os.path.join(output_dir, "error_compare.png")
    prediction_curve_path = os.path.join(output_dir, "target_prediction_curve.png")

    joblib.dump(target_only_model, model_path)

    prediction_compare.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig"
    )

    error_compare.to_csv(
        error_path,
        index=False,
        encoding="utf-8-sig"
    )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "Step 6 target intersection final prediction",
                "target_only_features": target_only_features,
                "models": {
                    "M1": "Target-only model trained only on target_adapt_data",
                    "M2": "Shared model directly transferred to target_test_data",
                    "M3": "Shared model plus target residual correction"
                },
                "final_prediction_formula": "M3: y_hat_final = y_hat_shared + residual_hat"
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    # 8. 绘图
    plot_error_comparison(
        error_df=error_compare,
        output_path=error_plot_path
    )

    plot_prediction_curve(
        prediction_df=prediction_compare,
        output_path=prediction_curve_path,
        max_points=200
    )

    # 9. 打印结果
    print("Step 6 目标交叉口最终预测完成。")
    print("-" * 70)

    print("输入文件：")
    print(f"  目标适配集：{target_adapt_path}")
    print(f"  目标测试集：{target_test_path}")
    print()

    print("三种模型含义：")
    print("  M1：只用目标交叉口少量样本直接训练")
    print("  M2：源交叉口训练的共性模型直接迁移")
    print("  M3：共性模型 + 目标路口残差修正")
    print()

    print("目标测试集误差对比：")
    print(error_compare)
    print()

    print("输出文件：")
    print(f"  M1 目标小样本模型：{model_path}")
    print(f"  预测对比表：{prediction_path}")
    print(f"  误差对比表：{error_path}")
    print(f"  配置文件：{config_path}")
    print(f"  误差对比图：{error_plot_path}")
    print(f"  预测曲线图：{prediction_curve_path}")

    return error_compare, prediction_compare


if __name__ == "__main__":
    final_prediction_step6(
        target_adapt_path="step5_outputs/target_adapt_final_prediction.csv",
        target_test_path="step5_outputs/target_test_final_prediction.csv",
        output_dir="step6_outputs"
    )