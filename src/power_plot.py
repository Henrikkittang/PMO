import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
print("Project root:", ROOT_DIR)
print("Root exists:", ROOT_DIR.is_dir())
print("Folders:", [p.name for p in ROOT_DIR.iterdir()])
sys.path.insert(0, str(ROOT_DIR))


from digital_twin.model import DigitalTwin
from digital_twin.api import API

SITE_TZ = 'Europe/Oslo'

if __name__ == '__main__':
    api = API()

    start_date = '2026-09-14'
    end_date   = '2026-09-14'

    # Fetch data
    inverter_data = api.solaredge.fetch(start_date, end_date)
    solar_data = api.openmeteo.fetch(start_date, end_date)
    nordpool_data = api.nordpool.fetch(start_date, end_date, 'NO3')

    # Build and run digital twin
    model = DigitalTwin()
    model.build_model()
    simulated_data = model.calculate_power(solar_data)

    # Ensure indices are datetime objects
    inverter_data.index = pd.to_datetime(inverter_data.index)
    simulated_data.index = pd.to_datetime(simulated_data.index)

    # Calculate energy from power (15-minute resolution)
    dt_hours = 15 / 60

    inverter_data['energy_kwh'] = (
        (inverter_data['power_kw'] + inverter_data['power_kw'].shift(1))
        / 2
        * dt_hours
    )

    simulated_data['energy_kwh'] = (
        (simulated_data['power_kw'] + simulated_data['power_kw'].shift(1))
        / 2
        * dt_hours
    )

    # Create plots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True
    )

    # Power plot
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
    ax1.set_title('Power and Energy Production')
    ax1.legend()
    ax1.grid(True)

    # Energy plot
    bar_width = 15 / (24 * 60)

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

    # Format x-axis
    ax2.set_xlim(
        inverter_data.index.min(),
        inverter_data.index.max(),
    )

    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=SITE_TZ))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=SITE_TZ))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
