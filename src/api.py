import os
import requests
import pandas as pd
import numpy as np
import collections
import io
from concurrent.futures import ThreadPoolExecutor, as_completed


import env

SITE_TZ = 'Europe/Oslo'
SITE_LAT = 63.4183333
SITE_LON = 10.4008333

class MET:
    # Returns weather data

    def __init__(self):
        self.url     = f'https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={round(SITE_LAT, 4)}&lon={round(SITE_LON, 4)}'
        self.headers = {'User-Agent': ''}

    def fetch(self):
        return self._fetch_remote()

    def _normalize(self, data):
         #finne været 'akkurat nå'
        timeseries = data['properties']['timeseries'][0]
        instant_data = timeseries['data']['instant']['details']
        forecast_time = pd.to_datetime(timeseries['time'], utc=True).tz_convert(SITE_TZ)
        
        #Hent symbol-koden for den neste timen (f.eks 'clearsky_day')
        symbol_code = timeseries['data']['next_1_hours']['summary']['symbol_code']
        return {
            'air_temperature': instant_data['air_temperature'],
            'wind_speed':      instant_data['wind_speed'],
            'weather_code':     symbol_code,
            'forecast_time':   forecast_time,
        }
    
    def _fetch_remote(self) -> dict | None:
        
        try:
            response = requests.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f'Klarte ikke hente vær fra MET: {e}')
            return None

        data = response.json()
        return data


class Frost:
    # Returns wind data

    def __init__(self):
        self.url       = 'https://frost.met.no/observations/v0.jsonld'
        self.client_id = env.FROST_ID

    def fetch(self, start_dato: str, slutt_dato: str) -> pd.DataFrame:
        data = self._fetch_remote(start_dato, slutt_dato)
        data =  self._normalize(data)
        return data

    def _normalize(self, data):
        data = data['data']
        rader = []
        for item in data:
            tidspunkt = item['referenceTime']
            if 'observations' in item and len(item['observations']) > 0:
                verdi = item['observations'][0]['value']
                rader.append({'TIMESTAMP': tidspunkt, 'wind_speed': verdi})
                
        df_vind = pd.DataFrame(rader)
        if not df_vind.empty:
            #Konverter til datetime og sett til riktig tidssone (Europe/Oslo)
            df_vind['TIMESTAMP'] = pd.to_datetime(df_vind['TIMESTAMP'])
            df_vind['TIMESTAMP'] = df_vind['TIMESTAMP'].dt.tz_convert(SITE_TZ)
            df_vind.set_index('TIMESTAMP', inplace=True)
            #Resample for å fjerne eventuelle duplikater på samme minutt
            df_vind = df_vind[~df_vind.index.duplicated(keep='first')]
            return df_vind
            
        return pd.DataFrame(columns=['wind_speed']) # Returner tom hvis feil

    def _fetch_remote(self, start_dato: str, slutt_dato: str) -> pd.DataFrame:
        #her legger man inn sin egen client_id
        #SN68860 er Voll stasjons id
        parameters = {
            'sources': 'SN68860', 
            'elements': 'wind_speed',
            'referencetime': f'{start_dato}/{slutt_dato}',
        }

        try:
            response = requests.get(self.url, parameters, auth=(self.client_id, ''), timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f'Frost API error: {e}')
            return pd.DataFrame(columns=['wind_speed']) # Returner tom hvis feil

        data = response.json()
        return data


class PVGIS:
    def __init__(self):
        pass

    def fetch(self):
        data = self._fetch_remote()
        data = self._normalize(data)
        return data

    def _normalize(self, data):

        tmy_df = pd.DataFrame(data['outputs']['tmy_hourly'])
        tmy_df = tmy_df.rename(columns={
            'time(UTC)': 'timestamp',
            'G(h)':  'ghi',
            'Gb(n)': 'dni',
            'Gd(h)': 'dhi',
            'T2m':   'temp_air',
            'WS10m': 'wind_speed'
        })

        tmy_df['timestamp'] = pd.to_datetime(tmy_df['timestamp'], format='%Y%m%d:%H%M', utc=True)
        tmy_df.set_index('timestamp', inplace=True)
    
        tmy_df = tmy_df.drop(columns=['RH', 'SP', 'IR(h)', 'WD10m'])

        numeric_columns = [
            'ghi',
            'dni',
            'dhi',
            'temp_air',
            'wind_speed'
        ]

        tmy_df[numeric_columns] = tmy_df[numeric_columns].astype(float)

        # tmy_df.index = tmy_df.index.tz_convert(SITE_TZ)

        return tmy_df

    def _fetch_remote(self) -> dict:
       
        #Prøv SARAH3 (nyeste, 2005–2023), så SARAH2, så ERA5 som siste fallback.
        #SARAH3 er den mest oppdaterte, med best oppløsning og presisjon for Europa
        databaser = ['PVGIS-SARAH3', 'PVGIS-SARAH2', 'PVGIS-ERA5']
        for db in databaser:
            url = (
                f'https://re.jrc.ec.europa.eu/api/v5_3/tmy'
                f'?lat={SITE_LAT}&lon={SITE_LON}&outputformat=json'
                f'&raddatabase={db}'
            )
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f'PVGIS {db} feilet: {e} — prøver neste database')

        return data


class OpenMeteo:
    def __init__(self) -> None:
        # self.url = 'https://archive-api.open-meteo.com/v1/archive'
        # self.url = 'https://historical-forecast-api.open-meteo.com/v1/forecast'
        self.url = 'https://api.open-meteo.com/v1/forecast'


    def fetch(self, start_date: str, end_date: str):
        data =  self._fetch_remote(start_date, end_date)
        data = self._normalize(data)
        return data

    def _normalize(self, data):
        df = pd.DataFrame(data['minutely_15'])

        df = df.rename(columns={
            'shortwave_radiation': 'ghi',
            'direct_normal_irradiance': 'dni',
            'diffuse_radiation': 'dhi',
            'temperature_2m': 'temp_air',
            'wind_speed_10m': 'wind_speed',
        })

        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

        return df

    def _fetch_remote(self, start_date: str, end_date: str):
        
        params = {
            'latitude':  SITE_LAT,
            'longitude': SITE_LON,
            'start_date': start_date,
            'end_date':   end_date,
            'minutely_15': [
                'shortwave_radiation',
                'direct_normal_irradiance',
                'diffuse_radiation',
                'temperature_2m',
                'wind_speed_10m',
                'cloud_cover',
            ],
            'timezone': SITE_TZ,
        }


        try:
            response = requests.get(self.url, params=params)
            response.raise_for_status()
        except Exception as e:
            print(f'OpenMeteo feilet: {e}')

        data = response.json()

        return data


class Sensor:
    # Sensor data from elektrobygget

    def __init__(self):
        pass

    def fetch(self):
        return self._fetch_remote('./data/PVCR1000_FiveSec.dat', 30)

    def _fetch_remote(self, filename: str, days: int, row_pr_day: int = 1440) -> pd.DataFrame | None:
        antall_rader = int(days * row_pr_day)

        if not os.path.exists(filename):
            print('Cant fine sensor data file', filename)
            return None

        # Et subset av dataen skrives til minne, nyere data prioriteres
        with open(filename, 'r', encoding='utf-8') as f:
            #leser headeren som er 4 linjer
            header_linjer = [next(f) for _ in range(4)]
            #deque fylles opp, når den er full, dyttes den ut eldste
            siste_rader = collections.deque(f, maxlen=antall_rader)
    
        #CR1000 format: Rad 2 inneholder kolonnenavnene
        #Vi slår sammen kolonnenavnene og dataradene til én lang tekststreng.
        csv_tekst = header_linjer[1] + ''.join(siste_rader)
        
        #pandas leser strengen istedenfor hele filen
        df = pd.read_csv(io.StringIO(csv_tekst), na_values='NAN', on_bad_lines='skip')

        df = df.rename(columns={
            'SP215_wm2': 'GHI_avg',
            'ST100_T': 'AirTemp_avg',
            'CS241_T': 'CellTemp_avg'
        })
        
        #formatering
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df.set_index('TIMESTAMP', inplace=True)
        
        df.index = df.index.tz_localize(SITE_TZ, ambiguous='NaT', nonexistent='NaT')
        df['GHI_avg'] = np.maximum(df['GHI_avg'], 0)
        
        #Ignorerer innstråling under 10 W/m2 for å unngå støy og evt. 
        #modellkrasj.
        df.loc[df['GHI_avg'] < 10, 'GHI_avg'] = 0

            
        return df


class SolarEdge:
    # Inverter data from elektrobygget

    def __init__(self):
        self.api_key = env.SOLAREDGE_KEY
        self.site_id = env.SOLAREDGE_SITE
        self.url     = f'https://monitoringapi.solaredge.com/site/{self.site_id}/power'
        self.filepath = './data/solaredge_master.csv'


    def fetch(self, start_date: str, end_date: str):
        ''' *start_date* and end_date can maximum be 14 days apart'''

        data = self._fetch_remote(start_date, end_date)
        df   = self._normalize(data)
        return df
        # df = pd.DataFrame()


        if df is None or df.empty:
            return None

        if os.path.exists(self.filepath):
            #last inn eksisterende historikk
            eksisterende_df = self._read_local(self.filepath)
                
            #fletter sammen
            oppdatert_df = pd.concat([eksisterende_df, df])
            
            #fjerner duplikater/overlapp i nye/gamle verdier
            oppdatert_df = oppdatert_df[~oppdatert_df.index.duplicated(keep='last')]
            oppdatert_df.sort_index(inplace=True)
        else:
            oppdatert_df = df
        
        #lagrer filen
        oppdatert_df.to_csv(self.filepath)
        return oppdatert_df


    def _read_local(self, filepath: str) -> pd.DataFrame:
        historikk_df = pd.read_csv(filepath, index_col=0, on_bad_lines='skip')
        if historikk_df.empty:
            return historikk_df

        tidsindeks = pd.to_datetime(historikk_df.index, utc=True, errors='coerce')
        gyldige_rader = tidsindeks.notna()
        historikk_df = historikk_df.loc[gyldige_rader]
        historikk_df.index = pd.DatetimeIndex(tidsindeks[gyldige_rader]).tz_convert(SITE_TZ)
        historikk_df.sort_index(inplace=True)
        return historikk_df

    def _normalize(self, data):
        df = pd.DataFrame(data['power']['values'])
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.index = df.index.tz_localize(SITE_TZ, ambiguous='NaT', nonexistent='NaT')
        
        df = df.rename(columns={'value': 'power_kw'})
        df['power_kw'] = df['power_kw'] / 1000

        df = df.fillna(0)

        return df

    def _fetch_remote(self, start_date: str, end_date: str) -> pd.DataFrame | None:
        
        start_time = start_date + ' 00:00:00'
        end_time   = end_date + ' 23:59:00'
        print(start_time, end_time)


        params  = {'api_key': self.api_key, 'startTime': start_time, 'endTime': end_time}

        try:
            response = requests.get(self.url, params=params, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f'SolarEdge feil: {e}')
            return None

        data = response.json()
  
        # Sjekk om kvotestopp er gjemt inne i et 200-svar
        feilmelding = (data.get('String') or data.get('message') or '').lower()
        if 'usage limit' in feilmelding or 'rate limit' in feilmelding:
            print(('⚠️ API-dagskvote nådd! SolarEdge tillater maks 300 kall/dag. Viser lagret historikk.'))
            return None
        
        return data
        

class NordPool:
    def __init__(self):
        self.url = 'https://dataportal-api.nordpoolgroup.com/api/DayAheadPriceIndices'


    def fetch(self, start_date, end_date, price_zone):


        dates = pd.date_range(start_date, end_date, freq='D')

        with ThreadPoolExecutor(max_workers=16) as executor:

            results = executor.map(
                lambda date: self._fetch_remote(
                    date.strftime('%Y-%m-%d'),
                    price_zone
                ),
                dates
            )

        all_data = []

        for data in results:
            if data is not None:
                all_data.extend(data['multiIndexEntries'])

        data = {'multiIndexEntries': all_data}

        return self._normalize(data, price_zone)

    def _normalize(self, data, price_zone):
      

        df = pd.DataFrame([
            {
                'timestamp': item['deliveryStart'],
                'price_nok_mwh': item['entryPerArea'][price_zone]
            }
            for item in data['multiIndexEntries']
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert(SITE_TZ)
        df['price_nok_kwh'] = df['price_nok_mwh'] / 1000

        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')


        return df

    def _fetch_remote(self, date, price_zone):
            
        
        params = {
            'currency': 'NOK', 
            'market': 'DayAhead', 
            'date': date, 
            'resolutionInMinutes': 15, 
            'indexNames': price_zone
        }

        try:
            response = requests.get(self.url, params=params, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f'NordPool error: {e}')
            return None

        data = response.json()
        
        return data

class API:
    def __init__(self):
        self.weather   = MET()
        self.frost     = Frost()
        self.sensor    = Sensor()
        self.pvgis     = PVGIS()
        self.solaredge = SolarEdge()
        self.openmeteo = OpenMeteo()
        self.nordpool  = NordPool()

# Testing APIs

if __name__ == '__main__':


    start_date = '2026-09-10'
    end_date   = '2026-09-23'
    

    nordpool = NordPool()
    df       = nordpool.fetch(start_date, end_date, 'NO3')



    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.plot(
        df.index,
        df['price_nok_kwh'],
        linewidth=1.5,
        drawstyle='steps-post'
    )

    plt.xlabel('Time')
    plt.ylabel('Price [NOK/kWh]')
    plt.title('Nord Pool NO3 Electricity Price')
    plt.grid(True, alpha=0.3)

    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
 
    plt.tight_layout()
    plt.show()
