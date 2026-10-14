"""
气象预警模块
------------
两类来源:

1. Open-Meteo 预报 API（免费, 无需 Key）:
   未来 24 小时逐小时的温度 / 阵风 / 降水预报,
   按中国气象局预警分级标准本地计算多级预警:
     高温: 黄 ≥35°C / 橙 ≥37°C / 红 ≥40°C        (24h 最高气温)
     大风: 蓝 ≥10.8 / 黄 ≥17.2 / 橙 ≥24.5 / 红 ≥32.7 m/s  (阵风, 对应 6/8/10/12 级)
     暴雨: 蓝 ≥50 / 黄 ≥100 / 橙 ≥150 / 红 ≥250 mm  (24h 累计降水)

2. 和风天气官方预警 API（可选, 需免费 Key）:
   气象部门正式发布的预警（台风、暴雨、高温、大风等全部类型）。
   注册: https://console.qweather.com  -> 创建项目获取 Key
   环境变量:
     QWEATHER_KEY   和风天气项目 Key
     QWEATHER_HOST  API 域名, 默认 https://devapi.qweather.com
                    （新项目需在控制台查看专属域名, 如 https://xxx.re.qweatherapi.com）
"""
import os
import time

import requests

CITY_GEO = {
    "beijing": (39.9042, 116.4074), "shanghai": (31.2304, 121.4737),
    "guangzhou": (23.1291, 113.2644), "shenzhen": (22.5431, 114.0579),
    "chengdu": (30.5728, 104.0668), "xian": (34.3416, 108.9398),
}

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
QWEATHER_KEY = os.getenv("QWEATHER_KEY", "")
QWEATHER_HOST = os.getenv("QWEATHER_HOST", "https://devapi.qweather.com")

TIMEOUT = 10
CACHE_TTL = 600  # 10 分钟缓存

_fc_cache = {}  # city -> {"ts": float, "data": dict}

LEVEL_NAMES = {"blue": "蓝色预警", "yellow": "黄色预警", "orange": "橙色预警", "red": "红色预警"}


def fetch_weather_forecast(city, force=False):
    """Open-Meteo 未来 48h 逐小时预报（温度/降水/风速/阵风）"""
    if city not in CITY_GEO:
        return None
    cached = _fc_cache.get(city)
    if not force and cached and time.time() - cached["ts"] < CACHE_TTL:
        return cached["data"]
    lat, lon = CITY_GEO[city]
    try:
        resp = requests.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m",
            "forecast_days": 2, "timezone": "auto",
        }, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        _fc_cache[city] = {"ts": time.time(), "data": data}
        return data
    except Exception:
        return None


def _mk(category, metric, value, unit, threshold, level, basis):
    return {
        "category": category, "metric": metric,
        "value": round(float(value), 1), "unit": unit,
        "threshold": threshold, "level": level,
        "level_name": LEVEL_NAMES[level], "basis": basis,
    }


def compute_weather_alerts(city):
    """基于未来 24h 预报计算高温 / 大风 / 暴雨多级预警"""
    fc = fetch_weather_forecast(city)
    if not fc:
        return []
    temps = [t for t in fc.get("temperature_2m", [])[:24] if t is not None]
    gusts = [g for g in fc.get("wind_gusts_10m", [])[:24] if g is not None]
    rains = [p for p in fc.get("precipitation", [])[:24] if p is not None]
    basis = "Open-Meteo 未来24h预报"
    alerts = []

    if temps:
        tmax = max(temps)
        if tmax >= 40:   alerts.append(_mk("高温", "最高气温", tmax, "°C", 40, "red", basis))
        elif tmax >= 37: alerts.append(_mk("高温", "最高气温", tmax, "°C", 37, "orange", basis))
        elif tmax >= 35: alerts.append(_mk("高温", "最高气温", tmax, "°C", 35, "yellow", basis))

    if gusts:
        gmax = max(gusts) / 3.6  # km/h -> m/s
        if gmax >= 32.7:   alerts.append(_mk("大风", "最大阵风", gmax, "m/s", 32.7, "red", basis))
        elif gmax >= 24.5: alerts.append(_mk("大风", "最大阵风", gmax, "m/s", 24.5, "orange", basis))
        elif gmax >= 17.2: alerts.append(_mk("大风", "最大阵风", gmax, "m/s", 17.2, "yellow", basis))
        elif gmax >= 10.8: alerts.append(_mk("大风", "最大阵风", gmax, "m/s", 10.8, "blue", basis))

    if rains:
        total = sum(rains)
        if total >= 250:   alerts.append(_mk("暴雨", "24h累计降水", total, "mm", 250, "red", basis))
        elif total >= 150: alerts.append(_mk("暴雨", "24h累计降水", total, "mm", 150, "orange", basis))
        elif total >= 100: alerts.append(_mk("暴雨", "24h累计降水", total, "mm", 100, "yellow", basis))
        elif total >= 50:  alerts.append(_mk("暴雨", "24h累计降水", total, "mm", 50, "blue", basis))

    return alerts


def qweather_configured() -> bool:
    return bool(QWEATHER_KEY)


def fetch_official_warnings(city):
    """
    和风天气官方预警（台风/暴雨/高温/大风等, 以气象部门发布为准）。
    未配置 Key 或调用失败返回 None。
    """
    if not QWEATHER_KEY or city not in CITY_GEO:
        return None
    lat, lon = CITY_GEO[city]
    try:
        resp = requests.get(
            f"{QWEATHER_HOST}/v7/warning/now",
            params={"location": f"{lon:.2f},{lat:.2f}", "key": QWEATHER_KEY},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") != "200":
            return None
        out = []
        for w in payload.get("warning", []):
            out.append({
                "category": w.get("typeName", "预警"),
                "level": w.get("level", ""),
                "title": w.get("title", ""),
                "text": (w.get("text") or "")[:200],
                "time": w.get("startTime", ""),
            })
        return out
    except Exception:
        return None
