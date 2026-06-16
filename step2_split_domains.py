# split_domains_step2.py
# Step 2：划分源交叉口和目标交叉口
# 目标：
# 1. A/B/C/D 作为源域 source_data
# 2. T 作为目标域 target_data
# 3. 从 T 中取少量样本作为 target_adapt_data，用于小样本适配
# 4. T 中剩余样本作为 target_test_data，用于最终测试
# 5. 确保 target_test_data 不参与模型训练，避免测试集泄漏

import os
import pandas as pd


def split_source_target_data(
    input_path="simulated_intersection_data.csv",
    output_dir="step2_outputs",
    source_ids=None,
    target_id="T",
    adapt_size=50,
    split_method="time"
):
    """
    划分源交叉口数据和目标交叉口数据。

    参数
    ----
    input_path:
        Step 1 生成的数据文件路径。

    output_dir:
        Step 2 输出文件夹。

    source_ids:
        源交叉口编号列表，默认 ["A", "B", "C", "D"]。

    target_id:
        目标交叉口编号，默认 "T"。

    adapt_size:
        目标域小样本适配数量，可设为 20、50、100。

    split_method:
        目标域划分方式。
        "time"：按时间顺序划分，前 adapt_size 条用于适配，剩余用于测试。
        "random"：随机抽取 adapt_size 条用于适配，剩余用于测试。
    """

    if source_ids is None:
        source_ids = ["A", "B", "C", "D"]

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取 Step 1 生成的数据
    data = pd.read_csv(input_path)

    # 2. 基本字段检查
    required_columns = [
        "intersection_id",
        "time",
        "flow",
        "speed",
        "occupancy",
        "queue_length",
        "green_time",
        "lane_num",
        "is_peak",
        "target_queue"
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要字段：{missing_columns}")

    # 3. 确保 time 是时间格式，方便按时间排序
    data["time"] = pd.to_datetime(data["time"])

    # 4. 划分源域数据：A/B/C/D
    source_data = data[data["intersection_id"].isin(source_ids)].copy()

    # 5. 划分目标域数据：T
    target_data = data[data["intersection_id"] == target_id].copy()

    if source_data.empty:
        raise ValueError("source_data 为空，请检查 source_ids 是否正确。")

    if target_data.empty:
        raise ValueError("target_data 为空，请检查 target_id 是否正确。")

    # 6. 目标域按时间排序
    target_data = target_data.sort_values("time").reset_index(drop=True)

    # 7. 检查小样本数量是否合理
    if adapt_size <= 0:
        raise ValueError("adapt_size 必须大于 0。")

    if adapt_size >= len(target_data):
        raise ValueError(
            f"adapt_size={adapt_size} 过大。目标域总样本数只有 {len(target_data)}，"
            "必须给 target_test_data 留出测试样本。"
        )

    # 8. 从目标域中划分小样本适配集和测试集
    if split_method == "time":
        # 更贴近真实新路口上线场景：
        # 前期少量数据用于适配，后期数据用于测试
        target_adapt_data = target_data.iloc[:adapt_size].copy()
        target_test_data = target_data.iloc[adapt_size:].copy()

    elif split_method == "random":
        # 课程展示也可以使用随机划分
        # 但报告里需要说明这是随机划分
        target_adapt_data = target_data.sample(
            n=adapt_size,
            random_state=42
        ).copy()

        target_test_data = target_data.drop(
            index=target_adapt_data.index
        ).copy()

        target_adapt_data = target_adapt_data.sort_values("time").reset_index(drop=True)
        target_test_data = target_test_data.sort_values("time").reset_index(drop=True)

    else:
        raise ValueError("split_method 只能是 'time' 或 'random'。")

    # 9. 再次重置索引
    source_data = source_data.sort_values(["intersection_id", "time"]).reset_index(drop=True)
    target_data = target_data.reset_index(drop=True)
    target_adapt_data = target_adapt_data.reset_index(drop=True)
    target_test_data = target_test_data.reset_index(drop=True)

    # 10. 检查是否发生测试集泄漏
    # 适配集和测试集不能有相同的 intersection_id + time
    adapt_keys = set(
        zip(target_adapt_data["intersection_id"], target_adapt_data["time"])
    )
    test_keys = set(
        zip(target_test_data["intersection_id"], target_test_data["time"])
    )

    leakage = adapt_keys.intersection(test_keys)

    if len(leakage) > 0:
        raise ValueError("发现 target_adapt_data 和 target_test_data 存在重叠，发生测试集泄漏。")

    # 11. 输出 CSV 文件
    source_path = os.path.join(output_dir, "source_data.csv")
    target_path = os.path.join(output_dir, "target_data.csv")
    adapt_path = os.path.join(output_dir, f"target_adapt_data_{adapt_size}.csv")
    test_path = os.path.join(output_dir, f"target_test_data_{adapt_size}.csv")

    source_data.to_csv(source_path, index=False, encoding="utf-8-sig")
    target_data.to_csv(target_path, index=False, encoding="utf-8-sig")
    target_adapt_data.to_csv(adapt_path, index=False, encoding="utf-8-sig")
    target_test_data.to_csv(test_path, index=False, encoding="utf-8-sig")

    # 12. 打印划分结果
    print("Step 2 数据划分完成。")
    print("-" * 60)
    print(f"输入文件：{input_path}")
    print(f"输出文件夹：{output_dir}")
    print()
    print(f"源域交叉口：{source_ids}")
    print(f"目标域交叉口：{target_id}")
    print()
    print(f"source_data 样本数：{len(source_data)}")
    print(f"target_data 样本数：{len(target_data)}")
    print(f"target_adapt_data 样本数：{len(target_adapt_data)}")
    print(f"target_test_data 样本数：{len(target_test_data)}")
    print()
    print("各数据集交叉口分布：")
    print("source_data:")
    print(source_data["intersection_id"].value_counts())
    print()
    print("target_adapt_data:")
    print(target_adapt_data["intersection_id"].value_counts())
    print()
    print("target_test_data:")
    print(target_test_data["intersection_id"].value_counts())
    print()
    print("输出文件：")
    print(source_path)
    print(target_path)
    print(adapt_path)
    print(test_path)

    return source_data, target_data, target_adapt_data, target_test_data


if __name__ == "__main__":
    # 可以把 adapt_size 改成 20、50、100，比较不同小样本规模下的效果
    source_data, target_data, target_adapt_data, target_test_data = split_source_target_data(
        input_path="simulated_intersection_data.csv",
        output_dir="step2_outputs",
        source_ids=["A", "B", "C", "D"],
        target_id="T",
        adapt_size=50,
        split_method="time"
    )
