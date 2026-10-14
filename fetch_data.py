"""
历史数据管道: Open-Meteo 多城市空气质量 + 气象数据拉取
------------------------------------------------------
数据源（均免费、无需 API Key）:
  - 空气质量: https://air-quality-api.open-meteo.com/v1/air-quality
    CAMS 全球再分析数据, 小时级, 含 PM2.5/PM10/O3/NO2/SO2/CO
  - 气象: https://archive-api.open-meteo.com/v1/archive
    ERA5 再分析, 小时级, 含温度/湿度/风速, 与空气质量同一经纬度对齐

输出: data/air_quality_cities.csv
  city, timestamp, PM25, PM10, SO2, NO2, CO(mg/m³), O3,
  temperature, humidity, wind_speed(m/s), AQI

AQI 按 HJ 633-2012 各污染物 IAQI 断点分段线性计算后取最大值。
"""
import os
import time

import numpy as np
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 引擎（backend/backend/engine.py）实际读取 backend/data/，输出目录必须与其一致
DATA_DIR = os.path.join(BASE_DIR, "backend", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 支持的城市: key -> (中文名, 纬度, 经度)
CITIES = {
    "beijing":   ("北京", 39.9042, 116.4074),
    "shanghai":  ("上海", 31.2304, 121.4737),
    "guangzhou": ("广州", 23.1291, 113.2644),
    "shenzhen":  ("深圳", 22.5431, 114.0579),
    "chengdu":   ("成都", 30.5728, 104.0668),
    "xian":      ("西安", 34.3416, 108.9398),
}

# 拉取最近一整年（留 2 天余量避开再分析数据延迟）
END_DATE = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
START_DATE = (pd.Timestamp.now() - pd.Timedelta(days=367)).strftime("%Y-%m-%d")

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"

# HJ 633-2012 IAQI 断点: 污染物 -> (浓度断点, 指数断点)
IAQI_BP = {
    "PM25": ([0, 35, 75, 115, 150, 250, 350], [0, 50, 100, 150, 200, 300, 500]),
    "PM10": ([0, 50, 150, 250, 350, 420, 500], [0, 50, 100, 150, 200, 300, 500]),
    "SO2":  ([0, 50, 150, 475, 800, 1600, 2100], [0, 50, 100, 150, 200, 300, 500]),
    "NO2":  ([0, 40, 80, 180, 280, 565, 750], [0, 50, 100, 150, 200, 300, 500]),
    "CO":   ([0, 2, 4, 14, 24, 36, 48], [0, 50, 100, 150, 200, 300, 500]),  # mg/m³
    "O3":   ([0, 160, 200, 300, 400, 800], [0, 50, 100, 150, 200, 300]),
}


def conc_to_iaqi(pollutant, conc):
    c, i = IAQI_BP[pollutant]
    return float(np.interp(conc, c, i))


def compute_aqi(row):
    """取各污染物 IAQI 最大值作为 AQI"""
    return max(
        conc_to_iaqi("PM25", row["PM25"]),
        conc_to_iaqi("PM10", row["PM10"]),
        conc_to_iaqi("SO2", row["SO2"]),
        conc_to_iaqi("NO2", row["NO2"]),
        conc_to_iaqi("CO", row["CO"]),
        conc_to_iaqi("O3", row["O3"]),
    )


def fetch_city(city_key, lat, lon):
    """拉取单城市一年小时级空气质量 + 气象数据并合并"""
    aq = requests.get(AQ_URL, params={
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
    }, timeout=120).json()["hourly"]

    wx = requests.get(WX_URL, params={
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    }, timeout=120).json()["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(aq["time"]),
        "PM25": aq["pm2_5"],
        "PM10": aq["pm10"],
        "O3": aq["ozone"],
        "NO2": aq["nitrogen_dioxide"],
        "SO2": aq["sulphur_dioxide"],
        "CO": np.array(aq["carbon_monoxide"], dtype=float) / 1000.0,  # µg/m³ -> mg/m³
    })
    wx_df = pd.DataFrame({
        "timestamp": pd.to_datetime(wx["time"]),
        "temperature": wx["temperature_2m"],
        "humidity": wx["relative_humidity_2m"],
        "wind_speed": np.array(wx["wind_speed_10m"], dtype=float) / 3.6,  # km/h -> m/s
    })
    df = df.merge(wx_df, on="timestamp", how="inner")
    df["city"] = city_key
    return df


def main():
    frames = []
    for key, (name, lat, lon) in CITIES.items():
        print(f"拉取 {name} ({key}) {START_DATE} ~ {END_DATE} ...")
        df = fetch_city(key, lat, lon)
        # 缺测值按时间线性插值
        num_cols = ["PM25", "PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity", "wind_speed"]
        df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
        df["AQI"] = df.apply(compute_aqi, axis=1).round().astype(int)
        frames.append(df)
        print(f"  -> {len(df)} 条, PM2.5 均值 {df['PM25'].mean():.1f}, AQI 均值 {df['AQI'].mean():.0f}")
        time.sleep(1)

    out = pd.concat(frames, ignore_index=True)
    out = out[["city", "timestamp", "PM25", "PM10", "SO2", "NO2", "CO", "O3",
               "temperature", "humidity", "wind_speed", "AQI"]]
    path = os.path.join(DATA_DIR, "air_quality_cities.csv")
    out.to_csv(path, index=False)
    print(f"\n已保存 {path}: {len(out):,} 条, {out['city'].nunique()} 城市")


if __name__ == "__main__":
    main()
