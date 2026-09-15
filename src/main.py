import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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

    # Digital twin|
    model = DigitalTwin()
    model.build_model()
    # simulated_data, _ = model.calculate_power(sensor_data)
    simulated_data = model.calculate_power_pvgis(solar_data)
   
    # ui = UI(weather_data, sensor_data, inverter_data, wind_data, simulated_data)
    # ui.render()


    plt.plot(
        simulated_data.index,
        simulated_data.values,
        label='Simulated',
        linewidth=2,
    )

    plt.plot(
        inverter_data.index,
        inverter_data['effekt_kw'],
        label='Inverter',
        linewidth=2,
    )

    plt.xlabel('Tid')
    plt.ylabel('Effekt (kW)')
    plt.legend()
    plt.grid(True)
    plt.show()

 
