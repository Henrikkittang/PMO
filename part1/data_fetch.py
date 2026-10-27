import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


from digital_twin.model import DigitalTwin
from digital_twin.api import API


start_date = '2026-09-12'
end_date   = '2026-09-12'

api = API()

openmeteo  = api.openmeteo
df_weather = openmeteo.fetch(start_date, end_date) 

df_price = api.nordpool.fetch(start_date, end_date, 'NO3')
df_power = api.solaredge.fetch(start_date, end_date)

model = DigitalTwin()
model.build_model()
df_model = model.calculate_power(df_weather)

# df_price.to_csv('./data/price.csv')
df_power.to_csv('./data/power_x.csv')
# df_model.to_csv('./data/model.csv')