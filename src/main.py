import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model import DigitalTwin
from api import API
from ui import UI




if __name__ == '__main__':

   

    api = API()

    # sensor_data = api.sensor.fetch()
    # start_date = sensor_data.index.min().strftime('%Y-%m-%d')
    # end_date = (sensor_data.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    # wind_data = api.frost.fetch(start_date, end_date)

    # weather_data = api.weather.fetch()

    start_date = '2026-09-01'
    end_date   = '2026-09-10'

    inverter_data = api.solaredge.fetch(start_date, end_date)
    solar_data = api.openmeteo.fetch(start_date, end_date)

    # Digital twin
    model = DigitalTwin()
    model.build_model()
    # simulated_data, _ = model.calculate_power(sensor_data)
    simulated_data = model.calculate_power_pvgis(solar_data)
    simulated_data = simulated_data.resample('15min').interpolate(method='linear')

    # ui = UI(weather_data, sensor_data, inverter_data, wind_data, simulated_data)
    # ui.render()

    print(simulated_data)


    inverter_data['dt_hours'] = inverter_data.index.to_series().diff().dt.total_seconds() / 3600
    inverter_data['energy_kWh'] = inverter_data['power_kw'] * inverter_data['dt_hours']

    simulated_data['dt_hours'] = simulated_data.index.to_series().diff().dt.total_seconds() / 3600
    simulated_data['energy_kWh'] = simulated_data['power_kw'] * simulated_data['dt_hours']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(
        simulated_data.index,
        simulated_data['power_kw'],
        label='Simulated',
        linewidth=2,
    )

    ax1.plot(
        inverter_data.index,
        inverter_data['power_kw'],
        label='Inverter',
        linewidth=2,
    )

    ax1.set_ylabel('Effekt (kW)')
    ax1.set_title('Effekt og energiproduksjon')
    ax1.legend()
    ax1.grid(True)

    ax2.bar(
        simulated_data.index,
        simulated_data['energy_kWh'],
        width=0.01,
        label='Simulated',
        alpha=0.6,
    )

    ax2.bar(
        inverter_data.index,
        inverter_data['energy_kWh'],
        width=0.01,
        label='Inverter',
        alpha=0.6,
    )

    ax2.set_xlabel('Tid')
    ax2.set_ylabel('Energi (kWh)')
    ax2.legend()
    ax2.grid(True)


    plt.tight_layout()
    plt.show()
    
