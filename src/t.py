



import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


from digital_twin.model import DigitalTwin
from digital_twin.api import API

#######################
# Python Variables
#######################

dt = 0.25

start_date = '2026-09-14'
end_date   = '2026-09-14'

api = API()


df_price = api.nordpool.fetch(start_date, end_date, 'NO3')
df_power = api.solaredge.fetch(start_date, end_date)


df_price.to_csv('./data/price.csv')
df_power.to_csv('./data/power.csv')