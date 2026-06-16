# train_shared_model_step3.py
# Step 3：训练共性拥堵预测模型 f_shared
#
# 目标：
# 1. 使用源交叉口 A/B/C/D 的数据训练共享拥堵预测模型
# 2. 输入 x: flow, speed, occupancy, queue_length, green_time,
#          cycle_length, lane_num, is_peak, weather
# 3. 输出 y: target_queue
# 4. 保存模型 shared_model.pkl
# 5. 为后续残差建模输出 y_hat_shared 和 shared_residual

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_onehot_encoder():
    """
    兼容不同 sklearn 版本的 OneHotEncoder。
    新版本使用 sparse_output=False，旧版本使用 sparse=False。
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def evaluate_regression(y_true, y_pred, dataset_name):
    """
    计算回归预测误差。
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "dataset": dataset_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }


def add_shared_prediction(data, model, features):
    """
    给数据集添加共性模型预测值 y_hat_shared 和残差 shared_residual。

    shared_residual = target_queue - y_hat_shared

    其中 shared_residual 表示：
    当前样本中，共性模型解释不了的部分。
    后续 Step 4 就要对这个残差建模。
    """
    data = data.copy()

    X = data[features]
    y_true = data["target_queue"]

    y_hat_shared = model.predict(X)

    data["y_hat_shared"] = np.round(y_hat_shared, 4)
    data["shared_residual"] = np.round(y_true - y_hat_shared, 4)

    return data


def train_shared_model(
    source_path="step2_outputs/source_data.csv",
    target_adapt_path="step2_outputs/target_adapt_data_50.csv",
    target_test_path="step2_outputs/target_test_data_50.csv",
    output_dir="step3_outputs"
):
    """
    训练共性拥堵预测模型 f_shared。

    参数
    ----
    source_path:
        Step 2 输出的源域数据路径，包含 A/B/C/D。

    target_adapt_path:
        Step 2 输出的目标域小样本适配集路径，包含 T 的前 50 条样本。

    target_test_path:
        Step 2 输出的目标域测试集路径，包含 T 的剩余样本。

    output_dir:
        Step 3 输出文件夹。
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    source_data = pd.read_csv(source_path)
    target_adapt_data = pd.read_csv(target_adapt_path)
    target_test_data = pd.read_csv(target_test_path)

    # 2. 设定输入特征和输出标签
    numeric_features = [
        "flow",
        "speed",
        "occupancy",
        "queue_length",
        "green_time",
        "cycle_length",
        "lane_num",
        "is_peak"
    ]

    categorical_features = [
        "weather"
    ]

    features = numeric_features + categorical_features
    target = "target_queue"

    # 3. 检查必要字段是否存在
    required_columns = features + [target]

    for name, df in [
        ("source_data", source_data),
        ("target_adapt_data", target_adapt_data),
        ("target_test_data", target_test_data)
    ]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{name} 缺少必要字段：{missing_columns}")

    # 4. 构造训练数据
    X_source = source_data[features]
    y_source = source_data[target]

    # 5. 构造预处理器
    # 数值变量直接进入模型，weather 做 One-Hot 编码
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", make_onehot_encoder(), categorical_features)
        ]
    )

    # 6. 定义共性模型
    # RandomForestRegressor 对非线性关系和变量交互比较稳健，适合课程项目
    shared_regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    # 7. 建立完整管道：预处理 + 模型
    f_shared = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", shared_regressor)
        ]
    )

    # 8. 训练共性模型
    f_shared.fit(X_source, y_source)

    # 9. 在源域、目标适配集、目标测试集上生成共性预测
    source_with_pred = add_shared_prediction(source_data, f_shared, features)
    target_adapt_with_pred = add_shared_prediction(target_adapt_data, f_shared, features)
    target_test_with_pred = add_shared_prediction(target_test_data, f_shared, features)

    # 10. 计算预测误差
    metrics = []

    metrics.append(
        evaluate_regression(
            source_with_pred[target],
            source_with_pred["y_hat_shared"],
            "source_train_A_B_C_D"
        )
    )

    metrics.append(
        evaluate_regression(
            target_adapt_with_pred[target],
            target_adapt_with_pred["y_hat_shared"],
            "target_adapt_T_direct_transfer"
        )
    )

    metrics.append(
        evaluate_regression(
            target_test_with_pred[target],
            target_test_with_pred["y_hat_shared"],
            "target_test_T_direct_transfer"
        )
    )

    metrics_df = pd.DataFrame(metrics)

    # 11. 保存模型和结果
    model_path = os.path.join(output_dir, "shared_model.pkl")
    feature_path = os.path.join(output_dir, "shared_model_features.json")
    metrics_path = os.path.join(output_dir, "shared_model_metrics.csv")

    source_pred_path = os.path.join(output_dir, "source_with_shared_pred.csv")
    target_adapt_pred_path = os.path.join(output_dir, "target_adapt_with_shared_pred.csv")
    target_test_pred_path = os.path.join(output_dir, "target_test_with_shared_pred.csv")

    joblib.dump(f_shared, model_path)

    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "features": features,
                "target": target
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    source_with_pred.to_csv(source_pred_path, index=False, encoding="utf-8-sig")
    target_adapt_with_pred.to_csv(target_adapt_pred_path, index=False, encoding="utf-8-sig")
    target_test_with_pred.to_csv(target_test_pred_path, index=False, encoding="utf-8-sig")

    # 12. 打印结果
    print("Step 3 共性模型训练完成。")
    print("-" * 70)
    print(f"源域训练数据：{source_path}")
    print(f"目标适配数据：{target_adapt_path}")
    print(f"目标测试数据：{target_test_path}")
    print()
    print("输入特征：")
    for col in features:
        print(f"  - {col}")
    print()
    print("输出标签：target_queue")
    print()
    print("预测误差：")
    print(metrics_df)
    print()
    print("输出文件：")
    print(f"  共性模型：{model_path}")
    print(f"  特征配置：{feature_path}")
    print(f"  误差指标：{metrics_path}")
    print(f"  源域预测结果：{source_pred_path}")
    print(f"  目标适配集预测结果：{target_adapt_pred_path}")
    print(f"  目标测试集预测结果：{target_test_pred_path}")

    return f_shared, metrics_df, source_with_pred, target_adapt_with_pred, target_test_with_pred


if __name__ == "__main__":
    train_shared_model(
        source_path="step2_outputs/source_data.csv",
        target_adapt_path="step2_outputs/target_adapt_data_50.csv",
        target_test_path="step2_outputs/target_test_data_50.csv",
        output_dir="step3_outputs"
    )