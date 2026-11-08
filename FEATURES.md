# Feature dictionary

Target: `y` = log1p(river_discharge_m3s)

Feature sets: `with_discharge` (all groups) and `rain_only` (no discharge_history).

## weather_same_day

- `precipitation_mm`: Total precipitation on day t (mm).
- `soil_moisture_0_100cm_m3m3`: Soil moisture, 0-100 cm layer, day t.
- `temperature_mean_c`: Mean air temperature, day t (deg C).
- `dew_point_mean_c`: Mean dew point, day t (deg C).
- `relative_humidity_mean_pct`: Mean relative humidity, day t (%).
- `wind_speed_max_kmh`: Maximum wind speed, day t (km/h).
- `wind_gusts_max_kmh`: Maximum wind gust, day t (km/h).

## rain_history

- `precip_sum_3d`: Sum of precipitation over days t-2..t (mm), per station.
- `precip_sum_7d`: Sum of precipitation over days t-6..t (mm), per station.

## seasonal

- `doy_sin`: sin(2*pi*day_of_year/365.25).
- `doy_cos`: cos(2*pi*day_of_year/365.25).
- `is_monsoon`: 1 if month is June to September, else 0.

## station

- `station_Bahrabise`: 1 if station is Bahrabise, else 0.
- `station_Belsot`: 1 if station is Belsot, else 0.
- `station_Bhada_Bridge`: 1 if station is Bhada Bridge, else 0.
- `station_Chameliya_Nayalbadi`: 1 if station is Chameliya/Nayalbadi, else 0.
- `station_Chatara`: 1 if station is Chatara, else 0.
- `station_Chisapani`: 1 if station is Chisapani, else 0.
- `station_Devghat`: 1 if station is Devghat, else 0.
- `station_Khokana`: 1 if station is Khokana, else 0.
- `station_Kusum`: 1 if station is Kusum, else 0.
- `station_Rasuwagadhi`: 1 if station is Rasuwagadhi, else 0.

## discharge_history

- `log_q_lag1`: log1p(discharge) on day t-1.
- `log_q_lag2`: log1p(discharge) on day t-2.
- `log_q_lag3`: log1p(discharge) on day t-3.
- `log_q_lag7`: log1p(discharge) on day t-7.
- `log_q_mean7`: Mean of log1p(discharge) over days t-7..t-1.
