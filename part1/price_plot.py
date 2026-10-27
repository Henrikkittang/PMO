import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


from digital_twin.api import API

SITE_TZ = 'Europe/Oslo'

if __name__ == '__main__':
    api = API()

    start_date = '2026-09-14'
    end_date   = '2026-09-14'

    df = api.nordpool.fetch(start_date, end_date, 'NO3')

    print(df)
    print(df.index.min())
    print(df.index.max())
    print(df.index.tz)

    plt.step(df.index, df['price_nok_kwh'], where='post')

    ax = plt.gca()
    ax.set_xlim(df.index.min(), df.index.max())
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=SITE_TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=SITE_TZ))
    
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.title('Nord Pool NO3 Electricty Price')
    plt.xlabel('Time [h]')
    plt.ylabel('Price [NOK/kWh]')

    plt.grid()
    plt.show()