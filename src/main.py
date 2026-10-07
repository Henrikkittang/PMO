import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model import DigitalTwin
from api import API
from ui import UI




if __name__ == '__main__':

    ui = UI()
    ui.render()
        
    quit()

    api = API()

    # sensor_data = api.sensor.fetch()
    # start_date = sensor_data.index.min().strftime('%Y-%m-%d')
    # end_date = (sensor_data.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    # wind_data = api.frost.fetch(start_date, end_date)

    # weather_data = api.weather.fetch()

    start_date = '2026-09-10'
    end_date   = '2026-09-23'

    inverter_data = api.solaredge.fetch(start_date, end_date)
    solar_data    = api.openmeteo.fetch(start_date, end_date)
    nordpool_data = api.nordpool.fetch(start_date, end_date, 'NO3')

    # Digital twin
    model = DigitalTwin()
    model.build_model()
    # simulated_data, _ = model.calculate_power(sensor_data)
    simulated_data = model.calculate_power(solar_data)
    # simulated_data = simulated_data.resample('15min').interpolate(method='linear')

    # ui = UI(weather_data, sensor_data, inverter_data, wind_data, simulated_data)
    # ui.render()

   
    
    dt_hours = 15 / 60 # Resolution of 15 min -> kwh
    inverter_data['energy_kwh'] = ((inverter_data['power_kw'] + inverter_data['power_kw'].shift(1)) / 2 * dt_hours )
    simulated_data['energy_kwh'] = ((simulated_data['power_kw'] + simulated_data['power_kw'].shift(1)) / 2 * dt_hours )

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


   
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Power and energy production')
    ax1.legend()
    ax1.grid(True)


    bar_width = 15 / (24 * 60)  # 15 minutes in days
    ax2.bar(
        simulated_data.index,
        simulated_data['energy_kwh'],
        width=bar_width,
        label='Simulated',
        alpha=0.6,
    )

    ax2.bar(
        inverter_data.index,
        inverter_data['energy_kwh'],
        width=bar_width,
        label='Inverter',
        alpha=0.6,
    )

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy (kWh)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(inverter_data.index.min(), inverter_data.index.max())

    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
    
