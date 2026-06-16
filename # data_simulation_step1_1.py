# data_simulation_step1_1.py
# Step 1.1：建立一个可以支撑迁移学习实验的数据表
# 每一行表示某个交叉口在某个时间片的交通状态
# 标签 target_queue 表示下一时段排队长度

import numpy as np
import pandas as pd


def is_peak_hour(hour: int) -> int:
    """
    判断是否为早晚高峰
    """
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return 1
    return 0


def generate_weather(rng: np.random.Generator) -> str:
    """
    随机生成天气
    """
    weather_list = ["sunny", "rainy", "cloudy"]
    weather_prob = [0.65, 0.15, 0.20]
    return rng.choice(weather_list, p=weather_prob)


def weather_factor(weather: str) -> float:
    """
    天气对交通拥堵的影响系数
    雨天更容易拥堵
    """
    if weather == "rainy":
        return 1.25
    elif weather == "cloudy":
        return 1.08
    else:
        return 1.00


def simulate_intersection_data(
    intersection_ids=None,
    start_time="2026-06-01 06:00:00",
    periods=1000,
    freq="5min",
    random_seed=42
):
    """
    生成多交叉口交通状态数据表。

    参数
    ----
    intersection_ids:
        交叉口编号列表，例如 ["A", "B", "C", "D", "T"]
        A-D 可作为源交叉口，T 可作为目标交叉口。

    start_time:
        起始时间。

    periods:
        时间片数量。1000 个 5 分钟时间片约为 3.47 天。
    如果 periods=288，则刚好表示一天。

    freq:
        时间间隔，默认 5 分钟。

    random_seed:
        随机种子，保证结果可复现。

    返回
    ----
    data:
        pandas.DataFrame
    """

    if intersection_ids is None:
        intersection_ids = ["A", "B", "C", "D", "T"]

    rng = np.random.default_rng(random_seed)
    time_index = pd.date_range(start=start_time, periods=periods, freq=freq)

    rows = []

    # 每个交叉口的静态特征
    # 这些特征可以理解为 Z_d，即路口的可观测静态属性
    static_config = {
    "A": {
        "lane_num": 4,
        "cycle_length": 90,
        "base_demand": 260,
        "intersection_bias": 1.00,
        "peak_sensitivity": 1.00,
        "has_bus_stop": 0,
        "near_school_hospital": 0
    },
    "B": {
        "lane_num": 3,
        "cycle_length": 100,
        "base_demand": 230,
        "intersection_bias": 1.10,
        "peak_sensitivity": 1.10,
        "has_bus_stop": 1,
        "near_school_hospital": 0
    },
    "C": {
        "lane_num": 5,
        "cycle_length": 110,
        "base_demand": 310,
        "intersection_bias": 0.95,
        "peak_sensitivity": 0.95,
        "has_bus_stop": 0,
        "near_school_hospital": 1
    },
    "D": {
        "lane_num": 4,
        "cycle_length": 95,
        "base_demand": 280,
        "intersection_bias": 1.15,
        "peak_sensitivity": 1.20,
        "has_bus_stop": 1,
        "near_school_hospital": 1
    },
    "T": {
        "lane_num": 3,
        "cycle_length": 100,
        "base_demand": 250,
        "intersection_bias": 1.25,
        "peak_sensitivity": 1.30,
        "has_bus_stop": 1,
        "near_school_hospital": 1
    },
}

    for inter_id in intersection_ids:
        config = static_config[inter_id]

        lane_num = config["lane_num"]
        cycle_length = config["cycle_length"]
        base_demand = config["base_demand"]
        intersection_bias = config["intersection_bias"]
        has_bus_stop = config["has_bus_stop"]
        near_school_hospital = config["near_school_hospital"]
        peak_sensitivity = config["peak_sensitivity"]

        # 初始排队长度
        queue_length = rng.uniform(5, 15)

        for t in time_index:
            hour = t.hour
            minute = t.minute
            peak = is_peak_hour(hour)
            weather = generate_weather(rng)

            # 时间周期性：模拟一天内交通需求变化
            # 早晚高峰更高，夜间更低
            if peak:
                time_factor = 1.55* peak_sensitivity
            elif 10 <= hour <= 16:
                time_factor = 1.10
            elif 20 <= hour <= 22:
                time_factor = 0.85
            else:
                time_factor = 0.55

            # 公交站、学校/医院等会增加局部扰动
            facility_factor = 1.0
            if has_bus_stop:
                facility_factor += 0.08
            if near_school_hospital:
                facility_factor += 0.10

            # 当前时间片到达交通量
            flow = (
                base_demand
                * time_factor
                * weather_factor(weather)
                * facility_factor
                * intersection_bias
                + rng.normal(0, 20)
            )
            flow = max(flow, 20)

            # 当前信号相位，简单设定为 1-4
            signal_phase = rng.integers(1, 5)

            # 绿灯时间：高峰期略长
            if peak:
                green_time = rng.normal(42, 5)
            else:
                green_time = rng.normal(35, 5)

            green_time = np.clip(green_time, 20, 60)

            # 道路通行能力
            # 简化理解：车道越多、绿信比越高，排队释放能力越强
            saturation_flow_per_lane = 28  # 每 5 分钟每车道可释放车辆数，简化设定
            discharge_capacity = (
                lane_num
                * saturation_flow_per_lane
                * green_time
                / cycle_length
            )

            # 占有率：与流量和排队长度正相关
            occupancy = (
                0.15
                + 0.0012 * flow
                + 0.008 * queue_length
                + rng.normal(0, 0.03)
            )
            occupancy = np.clip(occupancy, 0.05, 0.98)

            # 平均速度：与占有率和排队长度负相关
            speed = (
                55
                - 28 * occupancy
                - 0.35 * queue_length
                + rng.normal(0, 3)
            )
            speed = np.clip(speed, 5, 60)

            # 保存当前时刻状态
            rows.append({
                "intersection_id": inter_id,
                "time": t,
                "hour": hour,
                "minute": minute,
                "flow": round(flow, 2),
                "speed": round(speed, 2),
                "occupancy": round(occupancy, 3),
                "queue_length": round(queue_length, 2),
                "signal_phase": int(signal_phase),
                "green_time": round(green_time, 2),
                "cycle_length": cycle_length,
                "lane_num": lane_num,
                "is_peak": peak,
                "peak_sensitivity": peak_sensitivity,
                "weather": weather,
                "has_bus_stop": has_bus_stop,
                "near_school_hospital": near_school_hospital,
            })

            # 更新下一时段排队长度
            # 下一时段排队 = 当前剩余排队 + 新到达车辆 - 被绿灯释放车辆 + 随机扰动
            arrival = flow / 12  # 5分钟流量，简化处理
            random_disturbance = rng.normal(0, 3)

            next_queue_length = (
                0.75 * queue_length
                + arrival
                - discharge_capacity
                + random_disturbance
            )

            queue_length = max(0, next_queue_length)

    data = pd.DataFrame(rows)

    # 关键步骤：构造标签 target_queue
    # 对每一个交叉口，将下一时段 queue_length 作为当前行的标签
    data["target_queue"] = (
        data
        .groupby("intersection_id")["queue_length"]
        .shift(-1)
    )

    # 最后一个时间片没有下一时段标签，删除
    data = data.dropna(subset=["target_queue"]).reset_index(drop=True)

    # 保留两位小数
    data["target_queue"] = data["target_queue"].round(2)

    return data


if __name__ == "__main__":
    data = simulate_intersection_data(
        intersection_ids=["A", "B", "C", "D", "T"],
        start_time="2026-06-01 06:00:00",
        periods=1000,
        freq="5min",
        random_seed=42
    )

    # 输出 CSV 文件
    output_path = "simulated_intersection_data.csv"
    data.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("数据表已生成：", output_path)
    print("数据规模：", data.shape)
    print("\n前 10 行数据：")
    print(data.head(10))

    print("\n字段列表：")
    print(data.columns.tolist())

    print("\n各交叉口样本数量：")
    print(data["intersection_id"].value_counts())