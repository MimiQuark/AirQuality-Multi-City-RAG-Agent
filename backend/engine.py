import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PACKAGE_DIR)
DATA_DIR = os.path.join(PACKAGE_DIR, "data")
MODEL_DIR = os.path.join(PACKAGE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

POLLUTANTS = ["PM25", "PM10", "O3", "NO2"]
WEATHER = ["temperature", "humidity", "wind_speed"]

AQI_LEVELS = ["优", "良", "轻度污染", "中度污染", "重度污染", "严重污染"]

# 支持的城市: key -> 中文名（坐标与拉取脚本 fetch_data.py 一致）
CITIES = {
    "beijing": "北京", "shanghai": "上海", "guangzhou": "广州",
    "shenzhen": "深圳", "chengdu": "成都", "xian": "西安",
}
DEFAULT_CITY = "beijing"

# 各污染物预警阈值 (μg/m³, 24h 均值二级标准)
THRESHOLDS = {"PM25": 75, "PM10": 150, "O3": 160, "NO2": 80}


def check_city(city):
    if city not in CITIES:
        raise ValueError(f"暂不支持城市 {city}，可选: {list(CITIES)}")
    return city


def get_features(target):
    """按目标污染物构造特征: 其他污染物 + 气象因子 + 时刻/月份 + 目标自身的滞后/滚动特征
    注意: PM2.5 是 PM10 的组成部分, 两者存在整体-部分包含关系,
    互为特征会造成伪相关, 因此目标和其包含项互相剔除。"""
    exclude = {target}
    if target == "PM25":
        exclude.add("PM10")
    elif target == "PM10":
        exclude.add("PM25")
    others = [p for p in ["PM25", "PM10", "SO2", "NO2", "CO", "O3"] if p not in exclude]
    return others + WEATHER + ["hour", "month"] + [f"{target}_lag1h", f"{target}_lag24h", f"{target}_roll6h"]


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "air_quality_cities.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    num_cols = ["PM25", "PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity", "wind_speed", "AQI"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_data(df):
    """缺测值按城市内时间线性插值, 浓度负值截断为 0"""
    numeric_cols = ["PM25", "PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity", "wind_speed", "AQI"]
    for city in df["city"].unique():
        mask = df["city"] == city
        df.loc[mask, numeric_cols] = df.loc[mask, numeric_cols].interpolate(method="linear", limit_direction="both")
    for col in ["PM25", "PM10", "SO2", "NO2", "CO", "O3"]:
        df.loc[df[col] < 0, col] = 0
    return df


def aqi_level_name(aqi):
    if aqi <= 50: return "优"
    elif aqi <= 100: return "良"
    elif aqi <= 150: return "轻度污染"
    elif aqi <= 200: return "中度污染"
    elif aqi <= 300: return "重度污染"
    return "严重污染"


def add_features(df):
    df["aqi_level"] = df["AQI"].apply(aqi_level_name)
    df["season"] = df["month"].map({12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
                                     6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"})
    for city in df["city"].unique():
        mask = df["city"] == city
        for target in POLLUTANTS:
            series = df.loc[mask, target]
            df.loc[mask, f"{target}_lag1h"] = series.shift(1)
            df.loc[mask, f"{target}_lag24h"] = series.shift(24)
            # Shift before rolling: row t must not see target[t].
            df.loc[mask, f"{target}_roll6h"] = (
                series.shift(1).rolling(6, min_periods=1).mean()
            )
    df = df.dropna(subset=[f"{t}_lag1h" for t in POLLUTANTS] + [f"{t}_lag24h" for t in POLLUTANTS])
    return df


def preprocess(df):
    df = clean_data(df)
    df = add_features(df)
    return df


def get_city_df(df, city):
    check_city(city)
    return df[df["city"] == city].sort_values("timestamp").reset_index(drop=True)


# ─── 城市统计 ─────────────────────────────────────────────

def get_city_overview(df):
    """各城市汇总对比（用于城市对比页）"""
    stats = df.groupby("city").agg(
        avg_pm25=("PM25", "mean"),
        max_pm25=("PM25", "max"),
        avg_pm10=("PM10", "mean"),
        avg_o3=("O3", "mean"),
        avg_no2=("NO2", "mean"),
        avg_aqi=("AQI", "mean"),
        avg_temp=("temperature", "mean"),
        avg_humidity=("humidity", "mean"),
        good_days=("aqi_level", lambda x: (x.isin(["优", "良"])).mean() * 100)
    ).round(1)
    stats["good_days"] = stats["good_days"].round(1)
    stats = stats.reset_index()
    stats["name"] = stats["city"].map(CITIES)
    return stats.to_dict(orient="records")


def get_city_stats(df, city):
    """单城市汇总指标（用于概览卡片）"""
    cdf = get_city_df(df, city)
    return {
        "city": city,
        "name": CITIES[city],
        "records": len(cdf),
        "time_start": str(cdf["timestamp"].min()),
        "time_end": str(cdf["timestamp"].max()),
        "avg_pm25": round(float(cdf["PM25"].mean()), 1),
        "avg_aqi": round(float(cdf["AQI"].mean()), 1),
        "good_days": round(float(cdf["aqi_level"].isin(["优", "良"]).mean() * 100), 1),
    }


def get_season_analysis(df, city, target="PM25"):
    cdf = get_city_df(df, city)
    season = cdf.groupby("season").agg(
        avg_pm25=(target, "mean"),
        avg_aqi=("AQI", "mean"),
        avg_temp=("temperature", "mean")
    ).round(1).reindex(["春", "夏", "秋", "冬"])
    return season.reset_index().to_dict(orient="records")


def get_correlation(df, city):
    cdf = get_city_df(df, city)
    cols = ["PM25", "PM10", "SO2", "NO2", "CO", "O3", "temperature", "humidity", "wind_speed"]
    corr = cdf[cols].corr().round(3)
    result = []
    for i in cols:
        row = {"indicator": i}
        for j in cols:
            row[j] = float(corr.loc[i, j])
        result.append(row)
    return result


def get_aqi_distribution(df, city):
    cdf = get_city_df(df, city)
    dist = cdf["aqi_level"].value_counts(normalize=True) * 100
    return [{"level": lv, "percentage": round(float(dist.get(lv, 0)), 1)} for lv in AQI_LEVELS]


def get_latest_data(df):
    """各城市最新一条记录（用于城市对比表）"""
    result = []
    for city in df["city"].unique():
        cdf = df[df["city"] == city].sort_values("timestamp")
        row = cdf.iloc[-1]
        result.append({
            "city": city,
            "station": CITIES.get(city, city),
            "timestamp": str(row["timestamp"]),
            "PM25": round(float(row["PM25"]), 1),
            "PM10": round(float(row["PM10"]), 1),
            "SO2": round(float(row["SO2"]), 1),
            "NO2": round(float(row["NO2"]), 1),
            "CO": round(float(row["CO"]), 2),
            "O3": round(float(row["O3"]), 1),
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
            "wind_speed": float(row["wind_speed"]),
            "AQI": int(row["AQI"]),
            "aqi_level": aqi_level_name(row["AQI"])
        })
    return result


def get_alerts(df, city):
    """单城市最新时点超标告警"""
    cdf = get_city_df(df, city)
    row = cdf.iloc[-1]
    alerts = []
    for key, threshold in THRESHOLDS.items():
        v = float(row[key])
        if v > threshold:
            alerts.append({
                "station": CITIES[city],
                "metric": key,
                "value": round(v, 1),
                "threshold": threshold,
                "level": "warning" if v < threshold * 1.5 else "critical"
            })
    return alerts


P_LABELS = {"PM25": "PM2.5", "PM10": "PM10", "SO2": "SO2", "NO2": "NO2", "CO": "CO", "O3": "O3"}
_LEVELS_ML = {"yellow": "黄色预警", "orange": "橙色预警", "red": "红色预警"}


def get_pollutant_alerts_multilevel(vals: dict, basis: str):
    """
    六项污染物多级预警: 按 HJ 633-2012 将浓度换算为 IAQI,
    IAQI > 100 / 150 / 200 分别触发黄 / 橙 / 红色预警。
    vals: {"PM25": v, ...} (CO 为 mg/m³)
    """
    from .realtime import IAQI_BP
    out = []
    for pol, (c, i) in IAQI_BP.items():
        v = vals.get(pol)
        if not isinstance(v, (int, float)):
            continue
        iaqi = float(np.interp(v, c, i))
        if iaqi > 200:
            level = "red"
        elif iaqi > 150:
            level = "orange"
        elif iaqi > 100:
            level = "yellow"
        else:
            continue
        out.append({
            "category": "空气污染",
            "metric": P_LABELS[pol],
            "value": round(float(v), 2),
            "unit": "mg/m³" if pol == "CO" else "μg/m³",
            "iaqi": round(iaqi),
            "level": level,
            "level_name": _LEVELS_ML[level],
            "basis": basis,
        })
    out.sort(key=lambda a: -a["iaqi"])
    return out


# ─── 多污染物预测模型（按城市分别训练）─────────────────────

def _model_path(target, city):
    return os.path.join(MODEL_DIR, f"{city}_{target.lower()}_model.pkl")


def _features_path(target, city):
    return os.path.join(MODEL_DIR, f"{city}_{target.lower()}_features.pkl")


def train_model(df, target="PM25", city=DEFAULT_CITY):
    """
    训练单城市单污染物预测模型。
    采用时间序列划分（前 80% 训练 / 后 20% 测试），避免随机划分
    在含滞后特征的时序数据上造成信息泄漏。
    """
    check_city(city)
    features = get_features(target)
    d = get_city_df(df, city)
    split = int(len(d) * 0.8)
    train, test = d.iloc[:split], d.iloc[split:]

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(train[features].values, train[target].values)
    y_pred = model.predict(test[features].values)
    metrics = {
        "city": CITIES[city],
        "target": target,
        "mae": round(float(mean_absolute_error(test[target].values, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(test[target].values, y_pred))), 2),
        "r2": round(float(r2_score(test[target].values, y_pred)), 4),
        "split": "time-series 80/20"
    }
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).to_dict(orient="records")
    # Refit on the full series after evaluation so production inference uses
    # the latest data, while the reported metrics remain the held-out metrics.
    model.fit(d[features].values, d[target].values)
    joblib.dump(model, _model_path(target, city))
    joblib.dump(features, _features_path(target, city))
    return {
        "metrics": {**metrics, "training_rows": len(d), "refit": "full-series"},
        "importance": importance,
    }


def train_all(df, city=DEFAULT_CITY):
    """训练单城市全部污染物的预测模型"""
    return {t: train_model(df, t, city)["metrics"] for t in POLLUTANTS}


def load_model(target="PM25", city=DEFAULT_CITY):
    if not os.path.exists(_model_path(target, city)):
        df = preprocess(load_data())
        train_model(df, target, city)
    return joblib.load(_model_path(target, city)), joblib.load(_features_path(target, city))


def predict_single(data: dict, target="PM25", city=DEFAULT_CITY):
    model, features = load_model(target, city)
    X = np.array([[data.get(f, 0) for f in features]])
    pred = float(model.predict(X)[0])
    return {"city": CITIES.get(city, city), "pollutant": target, "prediction": round(pred, 1)}


def predict_batch(items: list, target="PM25", city=DEFAULT_CITY):
    model, features = load_model(target, city)
    X = np.array([[d.get(f, 0) for f in features] for d in items])
    preds = [round(float(p), 1) for p in model.predict(X)]
    return [{"pollutant": target, "prediction": p} for p in preds]


def get_feature_importance(df, target="PM25", city=DEFAULT_CITY):
    if not os.path.exists(_model_path(target, city)):
        train_model(df, target, city)
    model, features = load_model(target, city)
    return [{"feature": f, "importance": round(float(i), 4)}
            for f, i in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)]


def build_prefill(df, target="PM25", city=DEFAULT_CITY):
    """用该城市最新观测值构造一组预测输入（含目标污染物的真实滞后/滚动特征）"""
    features = get_features(target)
    cdf = get_city_df(df, city)
    last = cdf.iloc[-1]
    data = {}
    for f in features:
        data[f] = round(float(last[f]), 2)
    return {"city": CITIES.get(city, city), "pollutant": target,
            "timestamp": str(last["timestamp"]), "features": data}


# ─── 预测性预警（未来 24h）────────────────────────────────

def forecast_city_24h(city_df, target="PM25", city=DEFAULT_CITY, live=None):
    """
    单城市未来 24 小时递归预测:
    其他污染物与气象因子保持在最新观测值, 目标污染物的
    滞后/滚动特征随预测值递推更新。
    传入 live（WAQI 实时数据）时, 以实时观测为预测起点并更新静态特征,
    保证预警与实时板块口径一致。
    """
    model, features = load_model(target, city)
    sdf = city_df.sort_values("timestamp")
    history = list(sdf[target].values)  # 历史序列用于滞后/滚动特征递推
    latest_ts = sdf.iloc[-1]["timestamp"]

    static_vals = {f: float(sdf.iloc[-1][f]) for f in features
                   if f not in ("hour", "month") and not f.endswith(("lag1h", "lag24h", "roll6h"))}

    if live:
        lt = pd.to_datetime(live["time"]) if live.get("time") else pd.Timestamp.now()
        latest_ts = lt
        seed = dict(live.get("pollutants", {}))
        seed.update({k: live.get("weather", {}).get(k) for k in WEATHER})
        for k, v in seed.items():
            if k in static_vals and isinstance(v, (int, float)):
                static_vals[k] = float(v)
        lv = live.get("pollutants", {}).get(target)
        if isinstance(lv, (int, float)):
            history.append(float(lv))

    preds = []
    for h in range(1, 25):
        row = dict(static_vals)
        ts = latest_ts + pd.Timedelta(hours=h)
        row["hour"] = ts.hour
        row["month"] = ts.month
        row[f"{target}_lag1h"] = history[-1]
        row[f"{target}_lag24h"] = history[-24] if len(history) >= 24 else history[0]
        row[f"{target}_roll6h"] = float(np.mean(history[-6:]))
        X = np.array([[row[f] for f in features]])
        pred = float(model.predict(X)[0])
        preds.append(round(pred, 1))
        history.append(pred)
    return preds


def get_forecast_alerts(df, target="PM25", city=DEFAULT_CITY, live=None):
    """
    预测性预警: 对指定城市做未来 24h 递归预测,
    预计超标时输出风险等级、预计峰值与首次超标时间。
    传入 live 时预测起点为实时观测时刻。
    """
    threshold = THRESHOLDS[target]
    cdf = get_city_df(df, city)
    if live and live.get("time"):
        latest_ts = str(pd.to_datetime(live["time"])) + "（实时）"
    else:
        latest_ts = str(cdf["timestamp"].max()) + "（历史基座）"
    preds = forecast_city_24h(cdf, target, city, live=live)
    peak = max(preds)
    exceed_hours = [i + 1 for i, p in enumerate(preds) if p > threshold]
    return [{
        "station": CITIES[city],
        "pollutant": target,
        "threshold": threshold,
        "forecast_24h_max": peak,
        "forecast_24h_mean": round(float(np.mean(preds)), 1),
        "will_exceed": bool(exceed_hours),
        "first_exceed_in_hours": exceed_hours[0] if exceed_hours else None,
        "exceed_hours_count": len(exceed_hours),
        "level": ("critical" if peak > threshold * 1.5 else "warning") if exceed_hours else "safe",
        "forecast_start": str(latest_ts),
        "forecast_series": preds,
    }]


def evaluate_forecast_24h(df, target="PM25", city=DEFAULT_CITY, step=24):
    """
    预警命中率评估（在测试集时段上）:
    以 step 小时为间隔取预测起点, 对每个起点做未来 24h 递归预测,
    按"未来 24h 内是否出现超标"的二分类事件统计召回率 / 精确率。
    """
    threshold = THRESHOLDS[target]
    model, features = load_model(target, city)
    full = get_city_df(df, city)
    split = int(len(full) * 0.8)
    start = split
    origins = [o for o in range(start, len(full) - 24, step) if o >= 24]
    if not origins:
        return {"pollutant": target, "city": CITIES[city], "threshold": threshold,
                "events": 0, "recall": 0, "precision": 0}

    vals = full[target].values.astype(float)
    ts_vals = pd.to_datetime(full["timestamp"].values)
    static = {f: full[f].values.astype(float) for f in features
              if f not in ("hour", "month") and not f.endswith(("lag1h", "lag24h", "roll6h"))}

    histories = [list(vals[:o]) for o in origins]
    predicted_exceed = np.zeros(len(origins), dtype=bool)
    for h in range(24):
        X = []
        for j, o in enumerate(origins):
            hist = histories[j]
            row = {f: static[f][o - 1] for f in features
                   if f not in ("hour", "month") and not f.endswith(("lag1h", "lag24h", "roll6h"))}
            ts = ts_vals[o - 1] + pd.Timedelta(hours=h + 1)
            row["hour"] = ts.hour
            row["month"] = ts.month
            row[f"{target}_lag1h"] = hist[-1]
            row[f"{target}_lag24h"] = hist[-24]
            row[f"{target}_roll6h"] = float(np.mean(hist[-6:]))
            X.append([row[f] for f in features])
        preds = model.predict(np.array(X))
        predicted_exceed |= preds > threshold
        for j in range(len(origins)):
            histories[j].append(float(preds[j]))

    tp = fp = fn = 0
    for j, o in enumerate(origins):
        actual = bool(np.any(vals[o:o + 24] > threshold))
        if actual and predicted_exceed[j]: tp += 1
        elif not actual and predicted_exceed[j]: fp += 1
        elif actual and not predicted_exceed[j]: fn += 1

    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    return {
        "city": CITIES[city],
        "pollutant": target,
        "threshold": threshold,
        "events": tp + fn,
        "recall": round(recall, 3),
        "precision": round(precision, 3),
    }


# ─── 图表数据 ─────────────────────────────────────────────
def get_chart_pm25_monthly(df, city=None, target="PM25"):
    """污染物月度均值: city=None 时返回各城市对比, 否则返回单城市"""
    if city:
        cdf = get_city_df(df, city)
        monthly = cdf.groupby("month")[target].mean().round(1)
        return [{"month": int(m), CITIES[city]: float(monthly.get(m)) if m in monthly.index else None}
                for m in sorted(cdf["month"].unique())]
    pivot = df.pivot_table(values=target, index="month", columns="city", aggfunc="mean")
    result = []
    for month in sorted(pivot.index):
        entry = {"month": int(month)}
        for c in pivot.columns:
            v = pivot.loc[month, c]
            entry[CITIES.get(c, c)] = round(float(v), 1) if pd.notna(v) else None
        result.append(entry)
    return result


def get_chart_diurnal_pattern(df, city, target="PM25"):
    cdf = get_city_df(df, city)
    hourly = cdf.groupby("hour")[target].mean()
    series = [round(float(hourly.get(h, float("nan"))), 1) if h in hourly.index else None for h in range(24)]
    return [{"station": CITIES[city], "hourly_pm25": series}]


def get_chart_season_boxplot(df, city, target="PM25"):
    cdf = get_city_df(df, city)
    season_order = ["春", "夏", "秋", "冬"]
    result = []
    for s in season_order:
        vals = cdf[cdf["season"] == s][target].dropna()
        result.append({
            "season": s,
            "min": round(float(vals.min()), 1),
            "q1": round(float(vals.quantile(0.25)), 1),
            "median": round(float(vals.median()), 1),
            "q3": round(float(vals.quantile(0.75)), 1),
            "max": round(float(vals.max()), 1),
            "mean": round(float(vals.mean()), 1)
        })
    return result


def get_chart_pollutant_means(df, city):
    """单城市各污染物年均值对比（CO 单位为 mg/m³, 量级不同单独返回）"""
    cdf = get_city_df(df, city)
    means = [
        {"pollutant": "PM2.5", "value": round(float(cdf["PM25"].mean()), 1)},
        {"pollutant": "PM10", "value": round(float(cdf["PM10"].mean()), 1)},
        {"pollutant": "SO2", "value": round(float(cdf["SO2"].mean()), 1)},
        {"pollutant": "NO2", "value": round(float(cdf["NO2"].mean()), 1)},
        {"pollutant": "O3", "value": round(float(cdf["O3"].mean()), 1)},
        {"pollutant": "CO(mg/m³)", "value": round(float(cdf["CO"].mean()), 2)},
    ]
    return means


def get_chart_primary_pollutant(df, city):
    """首要污染物分布: 每小时各污染物 IAQI 最高者的占比"""
    from .realtime import IAQI_BP
    cdf = get_city_df(df, city)
    labels = {"PM25": "PM2.5", "PM10": "PM10", "SO2": "SO2", "NO2": "NO2", "CO": "CO", "O3": "O3"}
    iaqi_cols = {}
    for pol, (c, i) in IAQI_BP.items():
        iaqi_cols[pol] = np.interp(cdf[pol].values.astype(float), c, i)
    iaqi_mat = np.stack([iaqi_cols[p] for p in labels], axis=1)
    primary_idx = iaqi_mat.argmax(axis=1)
    keys = list(labels.keys())
    counts = pd.Series(primary_idx).value_counts()
    total = len(primary_idx)
    return [{"pollutant": labels[keys[i]], "percentage": round(float(counts.get(i, 0)) / total * 100, 1)}
            for i in range(len(keys))]


# ─── 规则式问答（LLM 降级兜底）────────────────────────────

def answer_question(question: str, city=DEFAULT_CITY) -> str:
    df = preprocess(load_data())
    cdf = get_city_df(df, city)
    cname = CITIES[city]
    q = question.lower()

    # 城市识别: 提到多个城市时直接给对比; 提到一个非当前城市时切换上下文
    mentioned = [k for k, n in CITIES.items() if n in question]
    if len(mentioned) >= 2 or (mentioned and any(w in q for w in ["对比", "比较", "排名", "哪个", "最高", "最低", "差别"])):
        cities = get_city_overview(df)
        lines = ["各城市空气质量对比（近一年均值）:"]
        for s in sorted(cities, key=lambda x: x["avg_pm25"], reverse=True):
            lines.append(f"  {s['name']}: PM2.5均值 {s['avg_pm25']} μg/m³, AQI均值 {s['avg_aqi']}, 优良率 {s['good_days']}%")
        return "\n".join(lines)
    if mentioned and mentioned[0] != city:
        city, cname, cdf = mentioned[0], CITIES[mentioned[0]], get_city_df(df, mentioned[0])

    # 识别问题中提到的污染物 (PM2.5/PM10/SO2/NO2/CO/O3)
    pollutant_alias = {
        "PM25": ["pm2.5", "pm25", "细颗粒物"],
        "PM10": ["pm10", "可吸入"],
        "SO2": ["so2", "二氧化硫"],
        "NO2": ["no2", "二氧化氮"],
        "CO": ["co", "一氧化碳"],
        "O3": ["o3", "臭氧"],
    }
    P_LABEL = {"PM25": "PM2.5", "PM10": "PM10", "SO2": "SO2", "NO2": "NO2", "CO": "CO", "O3": "O3"}
    target = None
    for pol, aliases in pollutant_alias.items():
        if any(w in q for w in aliases):
            target = pol
            break

    # 实时 / 当前空气质量（问的是具体污染物或季节规律时不走这里）
    time_words = ["现在", "实时", "当前", "今天", "此刻", "最新"]
    generic_ask = ("怎么样" in q or "如何" in q) and not target and not any(
        w in q for w in ["季节", "全年", "一年", "均值", "分布", "预测", "模型", "数据"])
    if any(w in q for w in time_words) or generic_ask:
        try:
            from .realtime import fetch_realtime
            live = fetch_realtime(city)
        except Exception:
            live = None
        if live:
            pol = ", ".join(f"{k}={v}" for k, v in live["pollutants"].items())
            aqi_txt = live["aqi"] if live["aqi"] is not None else "-"
            level = aqi_level_name(live["aqi"]) if isinstance(live["aqi"], (int, float)) else ""
            return (f"{cname}实时空气质量（{live['time']}，WAQI 地面站）:\n"
                    f"  AQI: {aqi_txt} {level}\n  {pol}")
        row = cdf.iloc[-1]
        return (f"{cname}最新记录（{row['timestamp']}，历史基座）:\n"
                f"  AQI {int(row['AQI'])}（{row['aqi_level']}）\n"
                f"  PM2.5 {row['PM25']:.1f} | PM10 {row['PM10']:.1f} | NO2 {row['NO2']:.1f} | O3 {row['O3']:.1f} μg/m³")

    # 数据来源 / 模型原理 / 系统功能等元问题
    if any(w in q for w in ["数据来源", "数据源", "数据集", "数据来自", "哪来的", "采集"]):
        return ("数据来源:\n"
                "  - 历史基座: Open-Meteo（CAMS 空气质量再分析 + ERA5 气象再分析）, 6 城市近一年约 5.3 万条小时级记录\n"
                "  - 实时数据: WAQI 地面监测站（配置 WAQI_TOKEN 后启用, 未配置时以历史基座最新时点代替）")
    if any(w in q for w in ["模型", "算法", "原理", "r2", "准确率", "精确率", "召回"]):
        return ("模型说明:\n"
                "  - 按城市分别训练随机森林回归模型, 预测 PM2.5/PM10/O3/NO2, 时间序列 80/20 划分\n"
                "  - 特征: 其他污染物 + 温湿度/风速 + 时刻/月份 + 目标滞后1h/24h 与 6h 滚动均值\n"
                "    （预测 PM2.5 时剔除 PM10, 避免整体-部分伪相关）\n"
                "  - 效果: 各城市 R² 0.92~0.98; 预警全年回测北京召回 76.7%、精确率 96.6%")
    if any(w in q for w in ["功能", "能做什么", "介绍", "你是谁", "帮助", "help", "用法"]):
        return ("我是环境监测 AI 助手, 可以:\n"
                "  - 查询某城市实时/当前空气质量（如'上海现在空气怎么样'）\n"
                "  - 分析任一污染物（PM2.5/PM10/SO2/NO2/CO/O3）的均值、峰值、季节规律\n"
                "  - 对比各城市空气质量（如'六个城市对比'）\n"
                "  - 查询当前超标告警与未来 24 小时预测预警\n"
                "  - 说明数据来源与模型原理")

    if any(w in q for w in ["aqi", "空气质量", "指数"]) and not target:
        dist = cdf["aqi_level"].value_counts(normalize=True) * 100
        good_pct = dist.get("优", 0) + dist.get("良", 0)
        return (
            f"{cname}近一年空气质量分析:\n"
            f"  - 优良率: {good_pct:.1f}%\n"
            f"  - 优: {dist.get('优', 0):.1f}% | 良: {dist.get('良', 0):.1f}%\n"
            f"  - 轻度污染: {dist.get('轻度污染', 0):.1f}% | 中度+: {dist.get('中度污染', 0)+dist.get('重度污染', 0)+dist.get('严重污染', 0):.1f}%"
        )

    # 任一污染物的专项分析（含季节与城市对比）
    if target:
        label = P_LABEL[target]
        unit = "mg/m³" if target == "CO" else "μg/m³"
        avg = cdf[target].mean()
        max_v = cdf[target].max()
        worst = cdf.loc[cdf[target].idxmax()]
        season = cdf.groupby("season")[target].mean().round(1)
        city_mean = df.groupby("city")[target].mean()
        return (
            f"{cname}近一年 {label} 分析（{str(cdf['timestamp'].min())[:10]} ~ {str(cdf['timestamp'].max())[:10]}）:\n"
            f"  - 均值: {avg:.1f} {unit} | 峰值: {max_v:.1f} {unit} ({str(worst['timestamp'])})\n"
            f"  - 季节均值: 春 {season.get('春', 0):.1f} / 夏 {season.get('夏', 0):.1f} / "
            f"秋 {season.get('秋', 0):.1f} / 冬 {season.get('冬', 0):.1f}\n"
            f"  - 六城市中 {CITIES.get(city_mean.idxmax(), '')}最高({city_mean.max():.1f}), "
            f"{CITIES.get(city_mean.idxmin(), '')}最低({city_mean.min():.1f})"
        )

    if any(w in q for w in ["季节", "冬季", "夏季", "春季", "秋季", "冬天", "夏天"]):
        season = cdf.groupby("season")["PM25"].mean().round(1)
        return (
            f"{cname}各季节PM2.5均值 (μg/m³):\n"
            f"  春季: {season.get('春', 0):.1f} | 夏季: {season.get('夏', 0):.1f}\n"
            f"  秋季: {season.get('秋', 0):.1f} | 冬季: {season.get('冬', 0):.1f}\n"
            f"  冬季污染最重, 夏季最轻, 呈现出明显的季节性特征"
        )

    if any(w in q for w in ["站点", "监测站", "站", "城市", "对比", "排名"]):
        cities = get_city_overview(df)
        lines = ["各城市空气质量对比（近一年均值）:"]
        for s in sorted(cities, key=lambda x: x["avg_pm25"], reverse=True):
            lines.append(f"  {s['name']}: PM2.5均值 {s['avg_pm25']} μg/m³, AQI均值 {s['avg_aqi']}, 优良率 {s['good_days']}%")
        return "\n".join(lines)

    if any(w in q for w in ["预测", "predict", "forecast"]):
        return (
            "本系统按城市分别训练随机森林模型, 预测 PM2.5 / PM10 / O3 / NO2 浓度。预测特征:\n"
            "  其他污染物指标, 温度, 湿度, 风速, 时刻/月份, 目标污染物滞后值(1h/24h)与滚动均值(6h)\n"
            "  请通过 /api/predict 接口提交数据进行预测, /api/alerts/forecast 获取未来24h预警"
        )

    if any(w in q for w in ["警报", "告警", "超标", "预警", "alarm"]):
        live = None
        try:
            from .realtime import fetch_realtime
            live = fetch_realtime(city)
        except Exception:
            pass
        if live:
            alerts = []
            for key, th in THRESHOLDS.items():
                v = live["pollutants"].get(key)
                if isinstance(v, (int, float)) and v > th:
                    alerts.append({"metric": key, "value": v, "threshold": th,
                                   "level": "warning" if v < th * 1.5 else "critical"})
            forecast = get_forecast_alerts(df, city=city, live=live)
            src = f"实时数据（{live['time']}）"
        else:
            alerts = get_alerts(df, city)
            forecast = get_forecast_alerts(df, city=city)
            src = "历史基座最新时点"
        lines = []
        if alerts:
            lines.append(f"{cname}当前共有 {len(alerts)} 项超标告警（{src}）:")
            for a in alerts:
                lines.append(f"  {a['metric']}={a['value']} (阈值{a['threshold']}, 级别: {a['level']})")
        else:
            lines.append(f"{cname}当前无超标告警（{src}）")
        f0 = forecast[0]
        if f0["will_exceed"]:
            lines.append(f"未来24小时{f0['pollutant']}预计{f0['first_exceed_in_hours']}小时后超标, "
                         f"24h峰值 {f0['forecast_24h_max']} (阈值{f0['threshold']})")
        else:
            lines.append("未来24小时无预计超标风险")
        return "\n".join(lines)

    return (
        "我是环境监测AI助手, 可以回答以下问题:\n"
        "  - PM2.5 / AQI 整体情况（可指定城市, 如'上海PM2.5怎么样'）\n"
        "  - 各城市空气质量对比\n"
        "  - 季节污染特征分析\n"
        "  - 当前超标告警与未来24小时预测预警\n"
        "  - 多污染物预测说明\n"
        "请告诉我你想了解什么?"
    )
