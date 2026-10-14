"""
实时空气质量接入 (WAQI / World Air Quality Index)
------------------------------------------------
注册免费 Token: https://aqicn.org/data-platform/token/

环境变量:
    WAQI_TOKEN   WAQI API Token (未配置时实时接口自动降级为历史最新数据)

注意: WAQI 返回的污染物为 IAQI 指数值, 本模块按 HJ 633-2012 断点
分段线性反解回浓度 (μg/m³, CO 为 mg/m³), 供预测模型与阈值告警使用。
"""
import os
import time
import numpy as np
import requests

WAQI_TOKEN = os.getenv("WAQI_TOKEN", "")
TIMEOUT = 10
CACHE_TTL = 600  # 10 分钟缓存

# WAQI 城市 feed 名（与 engine.CITIES 的 key 一致）
WAQI_FEED = {
    "beijing": "beijing", "shanghai": "shanghai", "guangzhou": "guangzhou",
    "shenzhen": "shenzhen", "chengdu": "chengdu", "xian": "xian",
}

# 城市中心坐标: 城市 feed 上报指标不全时, 按经纬度就近取站兜底
CITY_GEO = {
    "beijing": (39.9042, 116.4074), "shanghai": (31.2304, 121.4737),
    "guangzhou": (23.1291, 113.2644), "shenzhen": (22.5431, 114.0579),
    "chengdu": (30.5728, 104.0668), "xian": (34.3416, 108.9398),
}

# 与 prepare_data.py 一致的 IAQI 断点 (浓度, 指数)
IAQI_BP = {
    "PM25": ([0, 35, 75, 115, 150, 250, 350], [0, 50, 100, 150, 200, 300, 500]),
    "PM10": ([0, 50, 150, 250, 350, 420, 500], [0, 50, 100, 150, 200, 300, 500]),
    "SO2":  ([0, 50, 150, 475, 800, 1600, 2100], [0, 50, 100, 150, 200, 300, 500]),
    "NO2":  ([0, 40, 80, 180, 280, 565, 750], [0, 50, 100, 150, 200, 300, 500]),
    "CO":   ([0, 2, 4, 14, 24, 36, 48], [0, 50, 100, 150, 200, 300, 500]),
    "O3":   ([0, 160, 200, 300, 400, 800], [0, 50, 100, 150, 200, 300]),
}

_KEY_MAP = {"pm25": "PM25", "pm10": "PM10", "so2": "SO2", "no2": "NO2", "co": "CO", "o3": "O3"}

_cache = {}  # city -> {"ts": float, "data": dict}


def realtime_configured() -> bool:
    return bool(WAQI_TOKEN)


def _iaqi_to_conc(pollutant, iaqi_val):
    c, i = IAQI_BP[pollutant]
    return float(np.interp(iaqi_val, i, c))


def _fetch_feed(feed):
    """请求单个 WAQI feed, 成功返回解析后的 dict, 失败返回 None"""
    try:
        resp = requests.get(f"https://api.waqi.info/feed/{feed}/?token={WAQI_TOKEN}", timeout=TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "ok":
            return None
        d = payload["data"]
        iaqi = d.get("iaqi", {})

        pollutants = {}
        for key, name in _KEY_MAP.items():
            if key in iaqi and isinstance(iaqi[key].get("v"), (int, float)):
                pollutants[name] = round(_iaqi_to_conc(name, iaqi[key]["v"]), 2)
        if not pollutants:
            return None

        # 部分站点 aqi 字段返回 "-"（未发布综合指数）, 此时取各污染物 IAQI 最大值
        aqi = d.get("aqi")
        if not isinstance(aqi, (int, float)):
            vals = [iaqi[k]["v"] for k in _KEY_MAP
                    if k in iaqi and isinstance(iaqi[k].get("v"), (int, float))]
            aqi = max(vals) if vals else None

        return {
            "source_name": d.get("city", {}).get("name", feed),
            "aqi": aqi,
            "time": d.get("time", {}).get("s"),
            "pollutants": pollutants,
            "weather": {
                "temperature": iaqi.get("t", {}).get("v"),
                "humidity": iaqi.get("h", {}).get("v"),
                "wind_speed": iaqi.get("w", {}).get("v"),
            },
        }
    except Exception:
        return None


def fetch_realtime(city="beijing", force=False):
    """
    获取指定城市实时空气质量。成功返回 dict, 未配置或失败返回 None。
    城市 feed 上报指标不足 3 项时, 自动按城市中心坐标就近取站兜底。
    """
    if not WAQI_TOKEN:
        return None
    cached = _cache.get(city)
    if not force and cached and time.time() - cached["ts"] < CACHE_TTL:
        return cached["data"]

    result = _fetch_feed(WAQI_FEED.get(city, city))
    if (not result or len(result["pollutants"]) < 3) and city in CITY_GEO:
        lat, lon = CITY_GEO[city]
        geo = _fetch_feed(f"geo:{lat};{lon}")
        # 取污染物更全的一版
        if geo and (not result or len(geo["pollutants"]) > len(result["pollutants"])):
            result = geo

    if not result:
        return None
    result.update({
        "live": True,
        "city": city,
        "source": f"WAQI 实时（{result.pop('source_name')}）",
    })
    _cache[city] = {"ts": time.time(), "data": result}
    return result
