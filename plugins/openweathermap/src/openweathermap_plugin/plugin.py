from openweathermap_plugin.weather_current import openweathermap_weather
from openweathermap_plugin.weather_forecast_5d import openweathermap_forecast_5d
from openweathermap_plugin.weather_forecast_16d import openweathermap_forecast_16d
from openweathermap_plugin.weather_forecast_30d import openweathermap_forecast_30d


def register(context=None):
    return [
        openweathermap_weather,
        openweathermap_forecast_5d,
        openweathermap_forecast_16d,
        openweathermap_forecast_30d,
    ]
