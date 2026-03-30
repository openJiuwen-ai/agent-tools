# OpenWeatherMap Plugin（openJiuwen）

基于 OpenWeather **Current weather data API (2.5)** 与多日预报接口，支持当前天气、5 天每日预报、16 天每日预报与 30 天气候预报；当前天气与 5 天预报免费 Key 可用，16天每日预报与30天每日预报需 Pro 订阅。

参考：[Current weather data](https://openweathermap.org/current)、[5 Day / 3 Hour Forecast](https://openweathermap.org/forecast5)、[Daily Forecast 16 Days](https://openweathermap.org/forecast16)、[Climatic forecast 30 days](https://openweathermap.org/api/forecast30)

## 功能

| 工具 | 说明 |
|------|------|
| **openweathermap_weather** | 按经纬度或城市名查询当前天气，返回温度、体感、湿度、风速及简要描述 |
| **openweathermap_forecast_5d** | 未来 5 天、每 3 小时一档预报（温度、天气描述、降水概率等），支持经纬度或城市名 |
| **openweathermap_forecast_16d** | 未来 1~16 天每日预报（日温、天气描述、降水概率等），支持经纬度或城市名。需 Pro 订阅|
| **openweathermap_forecast_30d** | 未来 1~30 天气候预报（日温、天气描述、降水概率等），支持经纬度或城市名。需 Pro 订阅|

## 安装

```bash
cd plugins/openweathermap
pip install -e .
```

## 配置
在环境中配置 OpenWeatherMap API Key：

```bash
export openweathermap_api_key="your_api_key"
```

申请地址：[OpenWeatherMap API Keys](https://home.openweathermap.org/api_keys)


## 参数说明

- **位置二选一**：
  - **lat** + **lon**：经纬度，十进制度数。
  - **city**：城市英文名（如 Beijing、London），使用接口内置 `q` 参数。
- **units**（可选）：`standard` / `metric`（摄氏度）/ `imperial`（华氏度），默认 `metric`。
- **lang**（可选）：描述语言，默认 `zh_cn`。
- **openweathermap_forecast_5d** 额外参数：**cnt**（可选，1~40）返回的时间点数量，不传则返回完整约 40 档。
- **openweathermap_forecast_16d** 额外参数：**cnt**（1~16，默认 7）返回天数；该接口需 Pro 订阅。
- **openweathermap_forecast_30d** 额外参数：**cnt**（1~30，默认 7）返回天数；该接口需 [Pro 订阅](https://openweathermap.org/api/forecast30)。

## 返回示例

```json
{
  "report": "位置: Beijing, CN\n天气: 晴\n温度: 15℃ (体感 14℃)\n湿度: 45%\n风速: 3.5 m/s",
  "raw": { ... }
}
```

错误时返回 `{"error": "错误信息"}`。
