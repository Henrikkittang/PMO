import hashlib
import json
import numpy as np
import collections
import io
import os
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
import pvlib
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.ivtools.sdm import fit_cec_sam

#================================
#KONFIGURASJON, KONSTANTER OG API
#================================
st.set_page_config(page_title="NTNU Gløshaugen Digital Tvilling", layout="wide")

API_KEY = 'DE4WO3GRIPV9FDOBALZ5GQCUB97TF27F'
SITE_ID = '1459628' 
SITE_LAT = 63.4183333
SITE_LON = 10.4008333
SITE_TZ = 'Europe/Oslo'
APP_DIR = Path(__file__).resolve().parent
LEGACY_FAKTISK_FIL = APP_DIR / "pv-dashboard" / "data" / "faktisk.csv"
LEGACY_TEORETISK_FIL    = APP_DIR / "pv-dashboard" / "data" / "teoretisk.csv"
FILNAVN_DATALOGGER      = APP_DIR / "data" / "PVCR1000_FiveSec.dat" 
DATALOGGER_MASTER_FIL   = APP_DIR / "data" / "datalogger_master.csv" 
SOLAREDGE_HISTORIKK_FIL = APP_DIR / "data" / "solaredge_master.csv"
SIMULERING_MASTER_FIL   = APP_DIR / "data" / "simulering_master.csv"
SIMULERING_META_FIL     = APP_DIR / "data" / "simulering_meta.json"
SYSTEM_STC_KW = 20.15 # 62 paneler * 325W

#Kalibrert 2026-04-16 mot målte dagsenergier på klare dager (11., 13. og 20. april)
#OBS: inkluderer også smuss/skitt på panelene siden kalibreringen ikke skiller disse.
#Funksjon som burde implementeres her: utvid til sesongavhengig og solhøydeavhengig faktor.
SE_SHADING_FACTOR = 0.554

#Knappene som vises på skjermen og oppdateringshastighet
PERIODE_OPTIONS = ["24t", "7d", "30d", "År"]
LIVE_REFRESH_SECONDS = 5
FULL_REFRESH_SECONDS = 300
YTELSESRATIO_PERIODE_HELP = "Målt energi delt på simulert energi for valgt periode, ganget med 100."
YTELSESRATIO_LIVE_HELP = "Målt effekt delt på simulert effekt i sanntid, ganget med 100."
TYPISK_DAG_HELP = (
    "Beregnet fra PVGIS TMY (typisk meteorologisk år) for denne datoen. "
    "Viser hva anlegget forventes å produsere på en gjennomsnittlig dag i denne måneden."
)
MODULTEMPERATUR_HELP = (
    "Modultemperaturen måles med CS241 overflatetemperatursensor på baksiden av ett av modulene. "
    "Verdien konverteres til celletemperatur med SAPM-konverteringsfunksjon."
)
MÅNED_NAVN = [
    "Januar", "Februar", "Mars", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Desember",
]
EKSPORT_KOLONNENAVN = {
    "timestamp": "Tidspunkt",
    "irradiance_wm2": "Solinnstråling (W/m²)",
    "ambient_temp_c": "Omgivelsestemperatur (°C)",
    "maalt_panel_temp_c": "Modultemperatur (°C)",
    "effekt_kw_teor": "Simulert effekt (kW)",
    "effekt_kw": "Målt effekt (kW)",
    "residual_kw": "Effektavvik (kW)",
    "energi_kwh_teor": "Simulert energi (kWh)",
    "energi_kwh": "Målt energi (kWh)",
    "energi_avvik_kwh": "Energiavvik (kWh)",
}
EKSPORT_AVRUNDING = {
    "Solinnstråling (W/m²)": 1,
    "Omgivelsestemperatur (°C)": 2,
    "Modultemperatur (°C)": 2,
    "Simulert effekt (kW)": 3,
    "Målt effekt (kW)": 3,
    "Effektavvik (kW)": 3,
    "Simulert energi (kWh)": 3,
    "Målt energi (kWh)": 3,
    "Energiavvik (kWh)": 3,
}
EKSPORT_STANDARD_BREDDE = {
    "Tidspunkt": 130,
    "Solinnstråling (W/m²)": 130,
    "Omgivelsestemperatur (°C)": 155,
    "Modultemperatur (°C)": 145,
    "Simulert effekt (kW)": 135,
    "Målt effekt (kW)": 125,
    "Effektavvik (kW)": 130,
    "Simulert energi (kWh)": 165,
    "Målt energi (kWh)": 150,
    "Energiavvik (kWh)": 160,
}

#BACKUP AV SOLAREDGE DATA, FLETTER NYESTE VERDIER INN I EN MASTER-FIL SÅ VI IKKE MISTER VERDIER ELDRE ENN 30d
#Gjør også at hvis SolarEdge API kall ikke går gjennom, så vises fremdeles grafene.
def les_historikk_med_tidsindeks(filnavn: str) -> pd.DataFrame:
    historikk_df = pd.read_csv(filnavn, index_col=0, on_bad_lines='skip')
    if historikk_df.empty:
        return historikk_df
                                                                                                                                                                
    tidsindeks = pd.to_datetime(historikk_df.index, utc=True, errors='coerce')
    gyldige_rader = tidsindeks.isna()
    historikk_df = historikk_df.loc[gyldige_rader].copy()
    historikk_df.index = pd.DatetimeIndex(tidsindeks[gyldige_rader]).tz_convert(SITE_TZ)
    historikk_df.sort_index(inplace=True)
    return historikk_df

def synkroniser_solaredge_data(nye_api_data):

    if nye_api_data is None or nye_api_data.empty:
        return
    
    if os.path.exists(SOLAREDGE_HISTORIKK_FIL):
        #last inn eksisterende historikk
        eksisterende_df = les_historikk_med_tidsindeks(SOLAREDGE_HISTORIKK_FIL)
            
        #fletter sammen
        oppdatert_df = pd.concat([eksisterende_df, nye_api_data])
        
        #fjerner duplikater/overlapp i nye/gamle verdier
        oppdatert_df = oppdatert_df[~oppdatert_df.index.duplicated(keep='last')]
        oppdatert_df.sort_index(inplace=True)
    else:
        oppdatert_df = nye_api_data
    
    #lagrer filen
    oppdatert_df.to_csv(SOLAREDGE_HISTORIKK_FIL)
    return oppdatert_df

#BACKUP DATALOGGER DATA
#Denne funksjonen har vært litt komplikasjoner med. En dag var header til filen 
#korrupt, og de 5 høyre kolonnene var flyttet flere kolonner mot høyre.
def synkroniser_datalogger_data(renset_df):
    if renset_df is None or renset_df.empty:
        return renset_df

    #Kjente kolonner — beskytter mot phantom-kolonner om master-filen
    #har blitt skrevet med feil struktur (f.eks. etter pd.concat-mismatch).
    kjente_kolonner = ['RECORD', 'GHI_avg', 'CellTemp_avg', 'AirTemp_avg', 'wind_speed']

    if os.path.exists(DATALOGGER_MASTER_FIL):
        eksisterende_df = les_historikk_med_tidsindeks(DATALOGGER_MASTER_FIL)

        #Filtrer bort eventuelle phantom-kolonner fra korrupt master-fil
        gyldige_eks = [k for k in kjente_kolonner if k in eksisterende_df.columns]
        eksisterende_df = eksisterende_df[gyldige_eks]

        #Hopp over skriving om ingen nye tidspunkter finnes, unngår konstant
        #disk-skriving hvert 5. sekund når live-cachen utløper.
        siste_lagret = eksisterende_df.index.max() if not eksisterende_df.empty else pd.Timestamp.min.tz_localize('UTC')
        siste_ny = renset_df.index.max()
        if siste_ny <= siste_lagret:
            return eksisterende_df

        oppdatert_df = pd.concat([eksisterende_df, renset_df])
        oppdatert_df = oppdatert_df[~oppdatert_df.index.duplicated(keep='last')]
        oppdatert_df.sort_index(inplace=True)
    else:
        oppdatert_df = renset_df

    oppdatert_df.to_csv(DATALOGGER_MASTER_FIL)
    return oppdatert_df

#==========================
#BYGG SYSTEMET OG CACHE DET
#==========================
@st.cache_resource
def bygg_digital_tvilling():
    site = Location(SITE_LAT, SITE_LON, tz = SITE_TZ, altitude=60, name ='NTNU Elektro')

    temp_open = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']

    module_specs = {
        'celltype': 'monoSi', 'STC': 325, 'V_mp_ref': 57.6, 'I_mp_ref': 5.65,
        'V_oc_ref': 69.6, 'I_sc_ref': 6.03, 'alpha_sc': 0.00332, 'beta_voc': -0.164,
        'gamma_r': -0.258, 'N_s': 96,
    }

    calc_IL, calc_Io, calc_Rs, calc_Rsh, calc_a, calc_Adjust = fit_cec_sam(
        celltype=module_specs['celltype'], v_mp=module_specs['V_mp_ref'],
        i_mp=module_specs['I_mp_ref'], v_oc=module_specs['V_oc_ref'],
        i_sc=module_specs['I_sc_ref'], alpha_sc=module_specs['alpha_sc'],
        beta_voc=module_specs['beta_voc'], gamma_pmp=module_specs['gamma_r'],
        cells_in_series=module_specs['N_s']
    )
    module_parameters = module_specs.copy()
    module_parameters.update({'Name': 'Panasonic', 'I_L_ref': calc_IL,
                              'I_o_ref': calc_Io, 'R_sh_ref': calc_Rsh,
                              'R_s': calc_Rs, 'a_ref': calc_a,
                              'Adjust': calc_Adjust, 'A_c': 1.67})

    #skalerer SE20k inverter ned til SE17k med datablad spesifikasjoner
    #SE17K datablad: Paco=17000W, Vdco=750V (nominell), Pnt<2.5W.
    #Sandia C0-C3 fra SE20K siden de er samme "produkt-familie"
    #C0 (units: 1/W) invers-skalert
    k = 17000 / 20000
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    se20k = cec_inverters['SolarEdge_Technologies_Ltd___SE20K__480V_'].copy()
    inv_17k = se20k.copy()
    inv_17k['Paco'] = 17000                  #AC rating
    inv_17k['Pdco'] = se20k['Pdco'] * k      #Skalert DC input ved rated AC
    inv_17k['Vdco'] = 750                    #Nominell DC spenning
    inv_17k['Pso']  = se20k['Pso']  * k      #skalert oppstart-threshhold
    inv_17k['Pnt']  = 2.5                    #max nattestid-tap
    inv_17k['C0']   = se20k['C0']   / k      #invers-skalert

    arrays = [
        Array(FixedMount(15, 67), module_parameters=module_parameters, 
              temperature_model_parameters=temp_open, modules_per_string=12),
        Array(FixedMount(14, 247), module_parameters=module_parameters, 
              temperature_model_parameters=temp_open, modules_per_string=12),
        Array(FixedMount(55, 157), module_parameters=module_parameters, 
              temperature_model_parameters=temp_open, modules_per_string=7),
        Array(FixedMount(55, 157), module_parameters=module_parameters, 
              temperature_model_parameters=temp_open, modules_per_string=7),
        Array(FixedMount(48, 246), module_parameters=module_parameters, 
              temperature_model_parameters=temp_open, modules_per_string=12),
        Array(FixedMount(45, 180), module_parameters=module_parameters, 
              temperature_model_parameters=temp_open, modules_per_string=12)
    ]
    system = PVSystem(arrays=arrays, inverter_parameters=inv_17k)
    return site, inv_17k, ModelChain(system, site, aoi_model='physical', spectral_model='no_loss')

#====================================================
#PVGIS TMY — FORVENTET DAGSPRODUKSJON PER KALENDERDAG
#====================================================
@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def hent_tmy_dagsproduksjon() -> dict:
    """
    Henter PVGIS TMY-data (typisk meteorologisk år) og kjører pvlib-simulering
    for å beregne forventet dagsproduksjon (kWh) per kalenderdag.
    Returnerer dict: {(month, day): forventet_kwh}
    Cachet i 7 dager, TMY-data er statisk for et gitt sted.
    """
    #Prøv SARAH3 (nyeste, 2005–2023), så SARAH2, så ERA5 som siste fallback.
    #SARAH3 er den mest oppdaterte, med best oppløsning og presisjon for Europa
    databaser = ['PVGIS-SARAH3', 'PVGIS-SARAH2', 'PVGIS-ERA5']
    data = None
    for db in databaser:
        url = (
            f"https://re.jrc.ec.europa.eu/api/v5_3/tmy"
            f"?lat={SITE_LAT}&lon={SITE_LON}&outputformat=json"
            f"&raddatabase={db}"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            print(f"[TMY] Hentet data fra {db}")
            break
        except Exception as e:
            print(f"[TMY] {db} feilet: {e} — prøver neste database")

    if data is None:
        return {}

    hourly = data.get('outputs', {}).get('tmy_hourly', [])
    if not hourly:
        print(f"[TMY] Ingen tmy_hourly i svar: {list(data.keys())}")
        return {}

    records = []
    for entry in hourly:
        t_str = entry.get('time(UTC)', '')
        try:
            dt = pd.to_datetime(t_str, format='%Y%m%d:%H%M', utc=True)
        except Exception:
            continue
        records.append({
            'timestamp':  dt,
            'ghi':        float(entry.get('G(h)',   0) or 0),
            'dni':        float(entry.get('Gb(n)',  0) or 0),
            'dhi':        float(entry.get('Gd(h)',  0) or 0),
            'temp_air':   float(entry.get('T2m',   10) or 10),
            'wind_speed': float(entry.get('WS10m', 1.0) or 1.0),
        })

    if not records:
        return {}
    tmy_df = pd.DataFrame(records).set_index('timestamp')
    tmy_df.index = tmy_df.index.tz_convert(SITE_TZ)

    #ModelChain for TMY simulering
    site, inv_17k, mc_main = bygg_digital_tvilling()
    mc_tmy = ModelChain(
        mc_main.system, site, aoi_model='physical', spectral_model='no_loss'
    )
    weather = pd.DataFrame({
        'ghi':        tmy_df['ghi'].clip(lower=0),
        'dni':        tmy_df['dni'].clip(lower=0),
        'dhi':        tmy_df['dhi'].clip(lower=0),
        'temp_air':   tmy_df['temp_air'],
        'wind_speed': tmy_df['wind_speed'].clip(lower=0),
    }, index=tmy_df.index)
    mc_tmy.run_model(weather)
    dc = mc_tmy.results.dc
    p_dc = [
        dc[0]['p_mp'],
        dc[1]['p_mp'],
        dc[2]['p_mp'] * SE_SHADING_FACTOR,
        dc[3]['p_mp'] * SE_SHADING_FACTOR,
        dc[4]['p_mp'],
        dc[5]['p_mp'],
    ]
    total_dc = sum(p_dc)
    v_num    = sum(dc[i]['v_mp'] * p_dc[i] for i in range(6))
    safe_tot = total_dc.clip(lower=1e-9)
    v_dc     = (v_num / safe_tot).fillna(inv_17k['Vdco'])
    ac_kw = (
        pvlib.inverter.sandia(v_dc=v_dc, p_dc=total_dc, inverter=inv_17k)
        .fillna(0).clip(lower=0) / 1000.0
    )
    gruppert = ac_kw.groupby([ac_kw.index.month, ac_kw.index.day]).sum()
    return gruppert.to_dict()

#==============
#VÆRMELDING API
#==============
@st.cache_data(ttl=900)
def hent_live_vaer():
    #nærmeste er voll værstasjon som er rundt 2.77 km unna site location
    url = "https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=63.4183&lon=10.4008"
    #her legges user agent inn.
    headers = {"User-Agent": "BacheloroppgaveDigitalTvilling/1.0 (user@domain.com)"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            
            #finne været "akkurat nå"
            timeseries = data["properties"]["timeseries"][0]
            instant_data = timeseries["data"]["instant"]["details"]
            forecast_time = pd.to_datetime(timeseries["time"], utc=True).tz_convert(SITE_TZ)
            
            #Hent symbol-koden for den neste timen (f.eks "clearsky_day")
            symbol_code = timeseries["data"]["next_1_hours"]["summary"]["symbol_code"]
            
            return {
                "temperature": instant_data["air_temperature"],
                "windspeed": instant_data["wind_speed"],
                "weathercode": symbol_code,
                "forecast_time": forecast_time,
            }
        else:
            print(f"MET API feil: {res.status_code}")
    except Exception as e:
        print(f"Klarte ikke hente vær fra MET: {e}")
    return None


def oversett_vaerkode(symbol_code):
    if not symbol_code: 
        return "🌤️ Mangler data"
     
    #splitter strengen ved "_" så det blir enklere å matche med symbolet
    hovedkode = symbol_code.split('_')[0]
    
    ikoner = {
        "clearsky": "☀️ Klart",
        "fair": "🌤️ Lettskyet",
        "partlycloudy": "⛅ Delvis skyet",
        "cloudy": "☁️ Overskyet",
        "lightrain": "🌦️ Lett regn",
        "rain": "🌧️ Regn",
        "heavyrain": "🌧️ Kraftig regn",
        "rainshowers": "🌦️ Regnbyger",
        "snow": "❄️ Snø",
        "lightsnow": "🌨️ Lett snø",
        "heavysnow": "❄️ Kraftig snø",
        "sleet": "🌧️❄️ Sludd",
        "fog": "🌫️ Tåke",
        "thunder": "⛈️ Tordenvær"
    }
    
    #Returnerer ikon som hører til symbolkoden
    return ikoner.get(hovedkode, "🌤️ Varierende")

#=======================
#Funksjon hente vinddata
#=======================
#Vinddata benyttes i SAPM temperaturmodell for "Forventet Ytelse" panel,
#samt fallback hvis CS241 temperatursensor skulle feile.
@st.cache_data(ttl=3600, show_spinner=False)
def hent_vind_fra_frost(start_dato: str, slutt_dato: str) -> pd.DataFrame:
    #her legger man inn sin egen client_id
    client_id = '------'
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    #SN68860 er Voll stasjons id
    parameters = {
        'sources': 'SN68860', 
        'elements': 'wind_speed',
        'referencetime': f'{start_dato}/{slutt_dato}',
    }
    try:
        r = requests.get(endpoint, parameters, auth=(client_id, ''), timeout=15)
        if r.status_code == 200:
            data = r.json()['data']
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
                df_vind['TIMESTAMP'] = df_vind['TIMESTAMP'].dt.tz_convert('Europe/Oslo')
                df_vind.set_index('TIMESTAMP', inplace=True)
                #Resample for å fjerne eventuelle duplikater på samme minutt
                df_vind = df_vind[~df_vind.index.duplicated(keep='first')]
                return df_vind
    except Exception as e:
        print(f"Frost API feil: {e}")
        
    return pd.DataFrame(columns=['wind_speed']) # Returner tom hvis feil

#=========================
#HENT DATA API OG KJØR SIM
#=========================
@st.cache_data(ttl=900)
def hent_solaredge_data() -> pd.Series:
    naa = pd.Timestamp.now(tz='Europe/Oslo')
    start_dato = naa - pd.Timedelta(days=7)
    start_tid = start_dato.replace(hour=0, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
    slutt_tid = naa.strftime('%Y-%m-%d %H:%M:%S')

    url = f"https://monitoringapi.solaredge.com/site/{SITE_ID}/power"
    params = {'api_key': API_KEY, 'startTime': start_tid, 'endTime': slutt_tid}

    maalt_kw = pd.Series(dtype=float)
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code in (429, 403):
            st.sidebar.error("⚠️ API-dagskvote nådd! SolarEdge tillater maks 300 kall/dag. Viser lagret historikk.")
        elif response.status_code == 200:
            data = response.json()
            # Sjekk om kvotestopp er gjemt inne i et 200-svar
            feilmelding = (data.get('String') or data.get('message') or '').lower()
            if 'usage limit' in feilmelding or 'rate limit' in feilmelding:
                st.sidebar.error("⚠️ API-dagskvote nådd! SolarEdge tillater maks 300 kall/dag. Viser lagret historikk.")
            else:
                df_se = pd.DataFrame(data['power']['values'])
                if not df_se.empty:
                    df_se['date'] = pd.to_datetime(df_se['date'])
                    df_se.set_index('date', inplace=True)
                    df_se.index = df_se.index.tz_localize('Europe/Oslo', ambiguous='NaT', nonexistent='NaT')

                    #fletter inn nye verdier i backup filen
                    api_df = pd.DataFrame({'effekt_kw': df_se['value'] / 1000.0})
                    komplett_historikk_df = synkroniser_solaredge_data(api_df)

                    #returnerer sammenflettet historikk
                    maalt_kw = komplett_historikk_df['effekt_kw']
    except Exception as e:
        print(f"SolarEdge feil: {e}")

    if maalt_kw.empty and os.path.exists(SOLAREDGE_HISTORIKK_FIL):
        try:
            historikk_df = les_historikk_med_tidsindeks(SOLAREDGE_HISTORIKK_FIL)
            if 'effekt_kw' in historikk_df.columns:
                maalt_kw = historikk_df['effekt_kw']
        except Exception as e:
            print(f"Klarte ikke lese SolarEdge-historikk: {e}")
        
    return maalt_kw

#Leser kun headeren og de siste 'X' radene av en stor CSV-fil.
def hent_raskt_fra_csv(filnavn: str, dager: int, rader_per_dag: int = 1440) -> pd.DataFrame:
    antall_rader = int(dager * rader_per_dag)
    
    with open(filnavn, 'r', encoding='utf-8') as f:
        #leser headeren som er 4 linjer
        header_linjer = [next(f) for _ in range(4)]
        #deque fylles opp, når den er full, dyttes den ut eldste
        siste_rader = collections.deque(f, maxlen=antall_rader)

    #CR1000 format: Rad 2 inneholder kolonnenavnene
    #Vi slår sammen kolonnenavnene og dataradene til én lang tekststreng.
    csv_tekst = header_linjer[1] + "".join(siste_rader)
    
    #pandas leser strengen istedenfor hele filen
    df = pd.read_csv(io.StringIO(csv_tekst), na_values="NAN", on_bad_lines='skip')
    
    return df

def hent_og_kombiner_data(dager=30) -> pd.DataFrame:
    maalt_kw = hent_solaredge_data()
    try:
        #Antall rader loggeren din skriver på et døgn
        #Må justeres hvis logge-intervall endres
        rader_per_dag = 17280 
        #Bruker samme deque streaming som live dataen
        df_live = hent_raskt_fra_csv(FILNAVN_DATALOGGER, dager=dager, rader_per_dag=rader_per_dag)
        df_live = df_live.rename(columns={
            'SP215_wm2': 'GHI_avg',
            'ST100_T': 'AirTemp_avg',
            'CS241_T': 'CellTemp_avg'
            })
        #formatering
        df_live['TIMESTAMP'] = pd.to_datetime(df_live['TIMESTAMP'])
        df_live.set_index('TIMESTAMP', inplace=True)
        
        df_live.index = df_live.index.tz_localize('Europe/Oslo', ambiguous='NaT', nonexistent='NaT')
        df_live['GHI_avg'] = np.maximum(df_live['GHI_avg'], 0)
        
        #Ignorerer innstråling under 10 W/m2 for å unngå støy og evt. 
        #modellkrasj.
        df_live.loc[df_live['GHI_avg'] < 10, 'GHI_avg'] = 0

        #=============================================
        #Kode for å inkludere vind i målings-dataframe
        #=============================================
        start_str = df_live.index.min().strftime('%Y-%m-%d')
        slutt_str = (df_live.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        df_vind = hent_vind_fra_frost(start_str, slutt_str)
        if not df_vind.empty:
            #Slå sammen vinddataen med datalogger-dataen vår
            df_live = df_live.join(df_vind, how='left')
            # Forward-fill mellom timesmålinger — ingen kunstig fallback-verdi
            df_live['wind_speed'] = df_live['wind_speed'].ffill()
        else:
            #Frost API utilgjengelig, vises som manglende data i brukergrensesnittet
            df_live['wind_speed'] = np.nan
        print('B')
        #BACKUP SENSORDATA
        synkroniser_datalogger_data(df_live)
        
        return_value = bygg_ui_dataframe(df_live, maalt_kw)
        return return_value

    except Exception as e:
        st.warning(f"⚠️ Datafeil ved prosessering av sensordata: {e}")
        return pd.DataFrame()

def _beregn_pvlib_ac_kw(arbeids_df: pd.DataFrame):
    #Felles pvlib-kjerne for bygg_ui_dataframe og kjor_pvlib_simulering.
    #Returnerer (simulert_ac_kw, sapm_fallback_maske).
    site, custom_inverter, mc = bygg_digital_tvilling()
    solpos = site.get_solarposition(arbeids_df.index)
    irrad_components = pvlib.irradiance.erbs(arbeids_df['GHI_avg'], solpos['zenith'], arbeids_df.index)
    
    #CS241 måler modultemperatur (T_mod, bakflate). Konverter til celletemperatur
    #med SAPM: T_cell = T_mod + ΔT × (POA / 1000), der POA er for array 5 (tilt=48°, asimut=246°).
    sapm_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
    poa_cs241 = pvlib.irradiance.get_total_irradiance(
        surface_tilt=48, surface_azimuth=246,
        solar_zenith=solpos['apparent_zenith'], solar_azimuth=solpos['azimuth'],
        dni=irrad_components['dni'].fillna(0).clip(lower=0),
        ghi=arbeids_df['GHI_avg'],
        dhi=irrad_components['dhi'].fillna(0).clip(lower=0),
    )['poa_global'].fillna(0).clip(lower=0)

    t_mod_cs241 = arbeids_df['CellTemp_avg'].copy() if 'CellTemp_avg' in arbeids_df.columns else pd.Series(np.nan, index=arbeids_df.index)
    cell_temp = t_mod_cs241 + sapm_params['deltaT'] * poa_cs241 / 1000.0

    sapm_fallback_maske = t_mod_cs241.isna()   #True der CS241-sensor mangler data
    if sapm_fallback_maske.any():
        #Fallback: estimer T_cell direkte fra SAPM når CS241 er frakoblet
        cell_temp_sapm = pvlib.temperature.sapm_cell(
            poa_global=poa_cs241,
            temp_air=arbeids_df['AirTemp_avg'].fillna(10),
            wind_speed=arbeids_df['wind_speed'].fillna(1.0),
            **sapm_params
        )
        cell_temp = cell_temp.fillna(cell_temp_sapm)

    weather = pd.DataFrame({
        'ghi':              arbeids_df['GHI_avg'],
        'dhi':              irrad_components['dhi'].fillna(0).clip(lower=0),
        'dni':              irrad_components['dni'].fillna(0).clip(lower=0),
        'temp_air':         arbeids_df['AirTemp_avg'],
        'wind_speed':       arbeids_df['wind_speed'],
        'cell_temperature': cell_temp,
    }, index=arbeids_df.index)

    mc.run_model(weather)
    dc_results = mc.results.dc

    #Sandia inverter modell med SE17K skalerte parametre
    p_dc_per_array = [
        dc_results[0]['p_mp'],
        dc_results[1]['p_mp'],
        dc_results[2]['p_mp'] * SE_SHADING_FACTOR,
        dc_results[3]['p_mp'] * SE_SHADING_FACTOR,
        dc_results[4]['p_mp'],
        dc_results[5]['p_mp'],
    ]
    total_dc_watts = sum(p_dc_per_array)
    weighted_v_num = sum(dc_results[i]['v_mp'] * p_dc_per_array[i] for i in range(6))
    safe_total = total_dc_watts.clip(lower=1e-9)
    v_dc_est = (weighted_v_num / safe_total).fillna(custom_inverter['Vdco'])

    simulert_ac_kw = pvlib.inverter.sandia(
        v_dc=v_dc_est,
        p_dc=total_dc_watts,
        inverter=custom_inverter,
    ).fillna(0).clip(lower=0) / 1000.0

    return simulert_ac_kw, sapm_fallback_maske


def bygg_ui_dataframe(df_live: pd.DataFrame, maalt_kw: pd.Series) -> pd.DataFrame:
    if df_live is None or df_live.empty:
        return pd.DataFrame()

    arbeids_df = df_live.copy()
    if 'wind_speed' not in arbeids_df.columns:
        arbeids_df['wind_speed'] = np.nan

    simulert_ac_kw, sapm_fallback_maske = _beregn_pvlib_ac_kw(arbeids_df)

    ui_df = pd.DataFrame(index=arbeids_df.index)
    ui_df['irradiance_wm2'] = arbeids_df['GHI_avg']
    ui_df['ambient_temp_c'] = arbeids_df['AirTemp_avg']
    ui_df['maalt_panel_temp_c'] = arbeids_df['CellTemp_avg']
    ui_df['effekt_kw_teor'] = simulert_ac_kw
    #True der SAPM-estimat ble brukt istedenfor faktisk celletemp-sensor
    ui_df['celltemp_sapm_fallback'] = sapm_fallback_maske

    if maalt_kw is not None and not maalt_kw.empty:
        ui_df = ui_df.join(maalt_kw.rename('effekt_kw'), how='outer')
        siste_gyldige_se = maalt_kw.dropna().index.max()
        ui_df.loc[ui_df['irradiance_wm2'] < 10, 'effekt_kw'] = 0.0
        if pd.notnull(siste_gyldige_se):
            ui_df.loc[ui_df.index > siste_gyldige_se, 'effekt_kw'] = np.nan
    else:
        ui_df['effekt_kw'] = np.nan
        ui_df.loc[ui_df['irradiance_wm2'] < 10, 'effekt_kw'] = 0.0

    ui_df['residual_kw'] = ui_df['effekt_kw'] - ui_df['effekt_kw_teor']
    return ui_df.sort_index()

def les_legacy_eksportdata() -> pd.DataFrame:
    if not LEGACY_FAKTISK_FIL.exists() and not LEGACY_TEORETISK_FIL.exists():
        return pd.DataFrame()

    legacy_df = pd.DataFrame()

    if LEGACY_FAKTISK_FIL.exists():
        faktisk_df = pd.read_csv(LEGACY_FAKTISK_FIL)
        if 'timestamp' in faktisk_df.columns and 'effekt_kw' in faktisk_df.columns:
            faktisk_df['timestamp'] = pd.to_datetime(faktisk_df['timestamp'], errors='coerce')
            faktisk_df = faktisk_df.dropna(subset=['timestamp']).set_index('timestamp')
            legacy_df = legacy_df.join(faktisk_df[['effekt_kw']], how='outer') if not legacy_df.empty else faktisk_df[['effekt_kw']]

    if LEGACY_TEORETISK_FIL.exists():
        teoretisk_df = pd.read_csv(LEGACY_TEORETISK_FIL)
        if 'timestamp' in teoretisk_df.columns and 'effekt_kw_teor' in teoretisk_df.columns:
            teoretisk_df['timestamp'] = pd.to_datetime(teoretisk_df['timestamp'], errors='coerce')
            teoretisk_df = teoretisk_df.dropna(subset=['timestamp']).set_index('timestamp')
            legacy_df = legacy_df.join(teoretisk_df[['effekt_kw_teor']], how='outer') if not legacy_df.empty else teoretisk_df[['effekt_kw_teor']]

    if legacy_df.empty:
        return legacy_df

    legacy_df.index = pd.DatetimeIndex(legacy_df.index)
    legacy_df.sort_index(inplace=True)
    legacy_df['residual_kw'] = legacy_df['effekt_kw'] - legacy_df['effekt_kw_teor']
    return legacy_df

#============================================================
#SIMULERINGS-CACHE  (hindrer at pvlib kjøres på all historikk
#ved hvert eksportkall, kun nye tidspunkter simuleres)
#============================================================
def beregn_modell_hash() -> str:
    """
    Hvis en systemparameter endres, endres hashen
    cache invalideres automatisk og pvlib kjøres på nytt med all historikk.
    """
    site, custom_inverter, mc = bygg_digital_tvilling()

    #Array-konfigurasjon fra den faktiske modellen
    array_params = tuple(
        (
            round(float(a.mount.surface_tilt), 4),
            round(float(a.mount.surface_azimuth), 4),
            int(a.modules_per_string),
        )
        for a in mc.system.arrays
    )
    parametere = (
        SE_SHADING_FACTOR,
        array_params,
        round(float(custom_inverter['Paco']), 2),
        round(float(custom_inverter['Vdco']), 2),
        round(float(custom_inverter['Pdco']), 2),
    )
    return hashlib.md5(str(parametere).encode()).hexdigest()[:10]

def les_simulering_cache() -> pd.DataFrame:
    #Leser eksisterende simuleringscache. Returnerer tom DataFrame om filen ikke finnes.
    if not os.path.exists(SIMULERING_MASTER_FIL):
        return pd.DataFrame()
    try:
        df = les_historikk_med_tidsindeks(SIMULERING_MASTER_FIL)
        if 'effekt_kw_teor' not in df.columns:
            return pd.DataFrame()
        return df[['effekt_kw_teor']]
    except Exception:
        return pd.DataFrame()

def invalider_simulering_cache_om_nodvendig(gjeldende_hash: str) -> bool:
    
    #Sjekker om lagret modell-hash stemmer med gjeldende modell.
    #Hvis hashen er endret (eller mangler): sletter cache + meta-fil slik at alt simuleres på nytt neste gang.
    #Returnerer True om cachen ble invalidert.
    
    if not os.path.exists(SIMULERING_META_FIL):
        if os.path.exists(SIMULERING_MASTER_FIL):
            os.remove(SIMULERING_MASTER_FIL)
        return True
    try:
        with open(SIMULERING_META_FIL, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        lagret_hash = meta.get('model_hash', '')
    except Exception:
        lagret_hash = ''

    if lagret_hash != gjeldende_hash:
        if os.path.exists(SIMULERING_MASTER_FIL):
            os.remove(SIMULERING_MASTER_FIL)
        if os.path.exists(SIMULERING_META_FIL):
            os.remove(SIMULERING_META_FIL)
        return True
    return False

def kjor_pvlib_simulering(sensor_5min_df: pd.DataFrame) -> pd.Series:
    
    #Kjører pvlib-simuleringen på en DataFrame med 5-minutters sensordata og returnerer simulert AC-effekt som en Series (effekt_kw_teor).
    #Kalles kun på tidspunkter som mangler i cache-filen.
    
    if sensor_5min_df.empty:
        return pd.Series(dtype=float)

    arbeids_df = sensor_5min_df.copy()
    if 'wind_speed' not in arbeids_df.columns:
        arbeids_df['wind_speed'] = np.nan

    simulert_ac_kw, _ = _beregn_pvlib_ac_kw(arbeids_df)
    return simulert_ac_kw

def synkroniser_simulering_cache(ny_sim_df: pd.DataFrame, modell_hash: str):
    """
    Fletter ny simuleringsdata inn i cache-filen og oppdaterer meta-filen
    med gjeldende modell-hash.
    """
    if ny_sim_df.empty:
        return
    if os.path.exists(SIMULERING_MASTER_FIL):
        eksisterende = les_simulering_cache()
        oppdatert = pd.concat([eksisterende, ny_sim_df])
    else:
        oppdatert = ny_sim_df
    oppdatert = oppdatert[~oppdatert.index.duplicated(keep='last')]
    oppdatert.sort_index(inplace=True)
    oppdatert.to_csv(SIMULERING_MASTER_FIL)
    with open(SIMULERING_META_FIL, 'w', encoding='utf-8') as f:
        json.dump({'model_hash': modell_hash}, f)

def hent_eller_simuler_for_eksport(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tar inn rå sensordata (5-sekunder) og returnerer en 5-minutters
    DataFrame med effekt_kw_teor.

    Flyten:
      1. Sjekk modell-hash — ugyldiggjør cache hvis modellen er endret.
      2. Resample sensor til 5 min.
      3. Finn 5-min tidspunkter som mangler i cache-filen.
      4. Kjør pvlib KUN på manglende tidspunkter.
      5. Skriv nye resultater til cache-filen.
      6. Returner sensor (5 min) joinet med full cache.
    """
    if sensor_df.empty:
        return pd.DataFrame()

    modell_hash = beregn_modell_hash()
    invalider_simulering_cache_om_nodvendig(modell_hash)

    #Resample til 5 min — gir 105 000 rader/år
    sensor_5min = sensor_df.resample("5min").mean()

    cache_df = les_simulering_cache()

    if cache_df.empty:
        mangler_mask = pd.Series(True, index=sensor_5min.index)
    else:
        mangler_mask = ~sensor_5min.index.isin(cache_df.index)

    #Filtrer bort rader uten nødvendige sensorverdier
    paakrevde_kolonner = ['GHI_avg', 'AirTemp_avg', 'CellTemp_avg']
    manglende_df = sensor_5min.loc[mangler_mask].dropna(subset=paakrevde_kolonner)

    if not manglende_df.empty:
        ny_sim = kjor_pvlib_simulering(manglende_df)
        ny_sim_df = pd.DataFrame({'effekt_kw_teor': ny_sim}, index=manglende_df.index)
        synkroniser_simulering_cache(ny_sim_df, modell_hash)
        #Bygg oppdatert cache i minnet — unngår en ekstra fillesing fra disk
        if cache_df.empty:
            cache_df = ny_sim_df
        else:
            cache_df = pd.concat([cache_df, ny_sim_df])
            cache_df = cache_df[~cache_df.index.duplicated(keep='last')]

    return sensor_5min.join(cache_df[['effekt_kw_teor']], how='left')

@st.cache_data(ttl=FULL_REFRESH_SECONDS, show_spinner=False)
def hent_historiske_data_for_eksport() -> pd.DataFrame:
    try:
        eksport_deler = []

        if Path(FILNAVN_DATALOGGER).exists():
            #Les hele .dat-filen (CR1000-format: 4-linjers header, rad 2 = kolonnenavn)
            with open(FILNAVN_DATALOGGER, 'r', encoding='utf-8') as f:
                header_linjer = [next(f) for _ in range(4)]
                alle_rader    = f.readlines()
            csv_tekst = header_linjer[1] + ''.join(alle_rader)
            sensor_df = pd.read_csv(io.StringIO(csv_tekst), na_values='NAN', on_bad_lines='skip')
            
            #renamer datalogger kolonnene hentet så de matcher med det brukt i koden
            sensor_df = sensor_df.rename(columns={
                'SP215_wm2': 'GHI_avg',
                'ST100_T':   'AirTemp_avg',
                'CS241_T':   'CellTemp_avg',
            })
            sensor_df['TIMESTAMP'] = pd.to_datetime(sensor_df['TIMESTAMP'])
            sensor_df.set_index('TIMESTAMP', inplace=True)
            sensor_df.index = sensor_df.index.tz_localize(
                SITE_TZ, ambiguous='NaT', nonexistent='NaT'
            )
            sensor_df['GHI_avg'] = np.maximum(sensor_df['GHI_avg'], 0)
            sensor_df.loc[sensor_df['GHI_avg'] < 10, 'GHI_avg'] = 0.0

            #Vinddata fra Frost for hele perioden i filen
            start_str = sensor_df.index.min().strftime('%Y-%m-%d')
            slutt_str = (sensor_df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            df_vind = hent_vind_fra_frost(start_str, slutt_str)
            if not df_vind.empty:
                sensor_df = sensor_df.join(df_vind, how='left')
                sensor_df['wind_speed'] = sensor_df['wind_speed'].ffill()
            else:
                sensor_df['wind_speed'] = np.nan

            if not sensor_df.empty:
                #Henter simulert effekt fra cache, kjøres kun på tidspunkter
                #som mangler, og invalideres automatisk om modellen er endret.
                kombinert_5min = hent_eller_simuler_for_eksport(sensor_df)

                if not kombinert_5min.empty:
                    #Rename sensorkolonner til UI-navn
                    ny_df = kombinert_5min.rename(columns={
                        'GHI_avg':       'irradiance_wm2',
                        'AirTemp_avg':   'ambient_temp_c',
                        'CellTemp_avg':  'maalt_panel_temp_c',
                    })

                    #setter inn målte SolarEdge-verdier
                    maalt_kw = pd.Series(dtype=float)
                    if os.path.exists(SOLAREDGE_HISTORIKK_FIL):
                        historikk_df = les_historikk_med_tidsindeks(SOLAREDGE_HISTORIKK_FIL)
                        if 'effekt_kw' in historikk_df.columns:
                            maalt_kw = historikk_df['effekt_kw']

                    if not maalt_kw.empty:
                        ny_df = ny_df.join(maalt_kw.rename('effekt_kw'), how='left')
                    else:
                        ny_df['effekt_kw'] = np.nan

                    ny_df.loc[ny_df['irradiance_wm2'] < 10, 'effekt_kw'] = 0.0
                    ny_df['residual_kw'] = ny_df['effekt_kw'] - ny_df['effekt_kw_teor']
                    ny_df = ny_df.dropna(how="all").sort_index()
                    ny_df.index = ny_df.index.tz_convert(SITE_TZ).tz_localize(None)
                    eksport_deler.append(ny_df)

        legacy_df = les_legacy_eksportdata()
        if not legacy_df.empty:
            eksport_deler.append(legacy_df)

        if not eksport_deler:
            return pd.DataFrame()

        eksport_df = pd.concat(eksport_deler)
        eksport_df = eksport_df[~eksport_df.index.duplicated(keep='last')]
        eksport_df.sort_index(inplace=True)
        eksport_df['residual_kw'] = eksport_df['effekt_kw'] - eksport_df['effekt_kw_teor']
        return eksport_df
    except Exception as e:
        print(f"Feil ved bygging av eksportdata: {e}")
        return pd.DataFrame()

#Cacher historisk data for at bytte mellom visningsmoduser ikke skal lagge
@st.cache_data(ttl=FULL_REFRESH_SECONDS, show_spinner=False) 
def hent_historisk_data() -> pd.DataFrame:
    return hent_og_kombiner_data(dager=30)

#Cache matcher fragment-oppdateringsraten — alle fragmenter deler samme resultat i ett 5s-vindu
@st.cache_data(ttl=LIVE_REFRESH_SECONDS, show_spinner=False)
def hent_live_data_kjapt() -> pd.DataFrame:
    return hent_og_kombiner_data(dager=1)

#=======================
#CSV NEDLASTING HJELPERE
#=======================
def antall_iso_uker(ar: int) -> int:
    return int(pd.Timestamp(year=ar, month=12, day=28).isocalendar().week)

def klargjør_dataframe_for_csv_eksport(dataframe: pd.DataFrame) -> pd.DataFrame:
    eksport_df = dataframe.copy()
    eksport_df.index.name = "timestamp"
    eksport_df = eksport_df.reset_index().rename(columns=EKSPORT_KOLONNENAVN)

    tilgjengelig_avrunding = {
        kolonne: desimaler
        for kolonne, desimaler in EKSPORT_AVRUNDING.items()
        if kolonne in eksport_df.columns
    }
    if tilgjengelig_avrunding:
        eksport_df = eksport_df.round(tilgjengelig_avrunding)

    skjulte_kolonner = [
        "energi_kwh_teor_akk",
        "energi_kwh_akk",
        "energi_avvik_kwh_akk",
    ]
    eksport_df = eksport_df.drop(columns=[kolonne for kolonne in skjulte_kolonner if kolonne in eksport_df.columns])

    return eksport_df

def formater_eksportverdi(kolonne: str, verdi) -> str:
    if verdi is None or pd.isna(verdi):
        return ""

    if kolonne == "Tidspunkt":
        tidspunkt = pd.Timestamp(verdi)
        return tidspunkt.strftime("%d.%m.%Y %H:%M")

    if isinstance(verdi, (int, float, np.integer, np.floating)):
        desimaler = EKSPORT_AVRUNDING.get(kolonne, 3)
        tekst = f"{float(verdi):.{desimaler}f}".replace(".", ",")
        if "," in tekst:
            tekst = tekst.rstrip("0").rstrip(",")
        return tekst

    return str(verdi)

def kolonnebredde_for_eksport(dataframe: pd.DataFrame, kolonne: str) -> int:
    verdilengder = [len(formater_eksportverdi(kolonne, verdi)) for verdi in dataframe[kolonne]]
    maks_lengde = max([len(kolonne), *verdilengder], default=len(kolonne))
    beregnet_bredde = int(maks_lengde * 7 + 18)
    return max(EKSPORT_STANDARD_BREDDE.get(kolonne, 90), min(beregnet_bredde, 260))

def dataframe_til_excel_xml(dataframe: pd.DataFrame) -> bytes:
    eksport_df = klargjør_dataframe_for_csv_eksport(dataframe)
    kolonner = list(eksport_df.columns)
    kolonnebredder = {
        kolonne: kolonnebredde_for_eksport(eksport_df, kolonne)
        for kolonne in kolonner
    }

    stil_per_kolonne = {
        "Tidspunkt": "Tekst",
        "Solinnstråling (W/m²)": "Tall1",
        "Omgivelsestemperatur (°C)": "Tall2",
        "Modultemperatur (°C)": "Tall2",
        "Simulert effekt (kW)": "Tall3",
        "Målt effekt (kW)": "Tall3",
        "Effektavvik (kW)": "Tall3",
        "Simulert energi (kWh)": "Tall3",
        "Målt energi (kWh)": "Tall3",
        "Energiavvik (kWh)": "Tall3",
    }

    kolonne_xml = "".join(
        f'<Column ss:AutoFitWidth="0" ss:Width="{kolonnebredder[kolonne]}"/>'
        for kolonne in kolonner
    )

    header_xml = "".join(
        f'<Cell ss:StyleID="Header"><Data ss:Type="String">{escape(kolonne)}</Data></Cell>'
        for kolonne in kolonner
    )

    #Pre-trekk ut numpy-arrayer og NaN-masker per kolonne
    nan_celle = '<Cell ss:StyleID="Tekst"/>'
    col_data   = {k: eksport_df[k].values for k in kolonner}
    col_isnan  = {k: pd.isna(eksport_df[k].values) for k in kolonner}
    col_style  = {k: stil_per_kolonne.get(k, "Tall3") for k in kolonner}

    rader_xml = []
    for i in range(len(eksport_df)):
        celle_xml = []
        for kolonne in kolonner:
            if col_isnan[kolonne][i]:
                celle_xml.append(nan_celle)
                continue

            verdi = col_data[kolonne][i]

            if kolonne == "Tidspunkt":
                tekst = formater_eksportverdi(kolonne, verdi)
                celle_xml.append(
                    f'<Cell ss:StyleID="Tekst"><Data ss:Type="String">{escape(tekst)}</Data></Cell>'
                )
                continue

            celle_xml.append(
                f'<Cell ss:StyleID="{col_style[kolonne]}">'
                f'<Data ss:Type="Number">{float(verdi):.10f}</Data></Cell>'
            )

        rader_xml.append(f"<Row>{''.join(celle_xml)}</Row>")

    xml_tekst = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<?mso-application progid="Excel.Sheet"?>'
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:x="urn:schemas-microsoft-com:office:excel" '
        'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:html="http://www.w3.org/TR/REC-html40">'
        "<Styles>"
        '<Style ss:ID="Default" ss:Name="Normal"><Alignment ss:Vertical="Center"/></Style>'
        '<Style ss:ID="Header"><Font ss:Bold="1"/><Interior ss:Color="#D9EAF7" ss:Pattern="Solid"/></Style>'
        '<Style ss:ID="Tekst"><NumberFormat ss:Format="@"/></Style>'
        '<Style ss:ID="Tall1"><NumberFormat ss:Format="0.0"/></Style>'
        '<Style ss:ID="Tall2"><NumberFormat ss:Format="0.00"/></Style>'
        '<Style ss:ID="Tall3"><NumberFormat ss:Format="0.000"/></Style>'
        "</Styles>"
        '<Worksheet ss:Name="Historikk"><Table>'
        f"{kolonne_xml}"
        f"<Row>{header_xml}</Row>"
        f"{''.join(rader_xml)}"
        "</Table>"
        '<WorksheetOptions xmlns="urn:schemas-microsoft-com:office:excel">'
        "<FreezePanes/><FrozenNoSplit/><SplitHorizontal>1</SplitHorizontal>"
        "<TopRowBottomPane>1</TopRowBottomPane><ActivePane>2</ActivePane>"
        "</WorksheetOptions></Worksheet></Workbook>"
    )
    return xml_tekst.encode("utf-8")


def dataframe_til_csv(dataframe: pd.DataFrame) -> bytes:
    eksport_df = klargjør_dataframe_for_csv_eksport(dataframe)
    return eksport_df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def formater_sidebar_verdi(verdi, enhet: str = "", desimaler: int = 1) -> str:
    try:
        if verdi is None or pd.isna(verdi):
            raise ValueError
        return f"{float(verdi):.{desimaler}f} {enhet}".strip()
    except (TypeError, ValueError):
        return f"-- {enhet}".strip() if enhet else "--"

def hent_siste_gyldige_verdi(dataframe: pd.DataFrame, kolonne: str):
    if dataframe.empty or kolonne not in dataframe.columns:
        return None, None

    serie = dataframe[kolonne].dropna()
    if serie.empty:
        return None, None

    return float(serie.iloc[-1]), serie.index[-1]

def til_naiv_lokal_tid(ts: pd.Timestamp) -> pd.Timestamp:
    tidspunkt = pd.Timestamp(ts)
    if tidspunkt.tzinfo is not None:
        return tidspunkt.tz_convert(SITE_TZ).tz_localize(None)
    return tidspunkt

def formater_sidebar_tidspunkt(ts) -> str:
    if ts is None:
        return "--"
    return til_naiv_lokal_tid(pd.Timestamp(ts)).strftime("%d.%m.%Y %H:%M")

def bygg_full_eksportindeks(start_dato: pd.Timestamp, slutt_dato: pd.Timestamp, oppløsning: str) -> pd.DatetimeIndex:
    start = til_naiv_lokal_tid(start_dato)
    slutt = til_naiv_lokal_tid(slutt_dato)

    if oppløsning == "15 minutter":
        return pd.date_range(start=start.floor("15min"), end=slutt.floor("15min"), freq="15min")
    if oppløsning == "1 time":
        return pd.date_range(start=start.floor("1h"), end=slutt.floor("1h"), freq="1h")
    if oppløsning == "1 dag":
        return pd.date_range(start=start.normalize(), end=slutt.normalize(), freq="D")
    if oppløsning == "1 uke":
        return pd.date_range(start=start.normalize(), end=slutt.normalize(), freq="7d")
    if oppløsning == "1 måned":
        return pd.date_range(start=start.normalize().replace(day=1), end=slutt.normalize(), freq="MS")

    return pd.DatetimeIndex([])

def beregn_energikolonner_for_eksport(df: pd.DataFrame, oppløsning: str, start_dato: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    effekt_df = df[["effekt_kw_teor", "effekt_kw"]].copy()

    #Samme prinsipp som energivisningen i dashboardet: først timesnitt av effekt, så summeres disse til valgt energioppløsning.
    if oppløsning == "15 minutter":
        energi_df = effekt_df.resample("15min").mean() * 0.25
        energi_df = energi_df.rename(columns={"effekt_kw_teor": "energi_kwh_teor", "effekt_kw": "energi_kwh"})
        energi_df["energi_avvik_kwh"] = energi_df["energi_kwh"] - energi_df["energi_kwh_teor"]
        return energi_df

    if oppløsning == "1 time":
        energi_df = effekt_df.resample("1h").mean()
        energi_df = energi_df.rename(columns={"effekt_kw_teor": "energi_kwh_teor", "effekt_kw": "energi_kwh"})
        energi_df["energi_avvik_kwh"] = energi_df["energi_kwh"] - energi_df["energi_kwh_teor"]
        return energi_df

    if oppløsning not in {"1 dag", "1 uke", "1 måned"}:
        return pd.DataFrame()

    time_df = effekt_df.resample("1h").mean()
    if oppløsning == "1 dag":
        energi_df = time_df.resample("D").sum(min_count=1)
    elif oppløsning == "1 uke":
        energi_df = time_df.resample("7D", origin=til_naiv_lokal_tid(start_dato).normalize()).sum(min_count=1)
    elif oppløsning == "1 måned":
        energi_df = time_df.resample("MS").sum(min_count=1)
    else:
        energi_df = pd.DataFrame(index=effekt_df.index)

    if energi_df.empty:
        return energi_df

    energi_df = energi_df.rename(columns={
        "effekt_kw_teor": "energi_kwh_teor",
        "effekt_kw": "energi_kwh",
    })
    energi_df["energi_avvik_kwh"] = energi_df["energi_kwh"] - energi_df["energi_kwh_teor"]
    return energi_df

def resample_nedlasting_dataframe(df: pd.DataFrame, oppløsning: str, start_dato: pd.Timestamp, slutt_dato: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df

    freq_map = {
        "15 minutter": "15min",
        "1 time": "1h",
        "1 dag": "D",
        "1 uke": "7D",
        "1 måned": "MS"
    }

    freq = freq_map.get(oppløsning, "15min")
    if start_dato is None or slutt_dato is None:
        start = pd.Timestamp(df.index.min())
        slutt = pd.Timestamp(df.index.max())
    else:
        start = til_naiv_lokal_tid(start_dato)
        slutt = til_naiv_lokal_tid(slutt_dato)

    if oppløsning == "1 uke":
        resampled_df = df.resample(freq, origin=start.normalize()).mean()
    else:
        resampled_df = df.resample(freq).mean()

    full_indeks = bygg_full_eksportindeks(start, slutt, oppløsning)
    if len(full_indeks) > 0:
        resampled_df = resampled_df.reindex(full_indeks)

    energi_df = beregn_energikolonner_for_eksport(df, oppløsning, start)
    if len(full_indeks) > 0 and not energi_df.empty:
        energi_df = energi_df.reindex(full_indeks)

    if not energi_df.empty:
        resampled_df = resampled_df.join(energi_df, how="left")

    if {"effekt_kw_teor", "effekt_kw"}.issubset(resampled_df.columns):
        resampled_df["residual_kw"] = resampled_df["effekt_kw"] - resampled_df["effekt_kw_teor"]

    if {"energi_kwh_teor", "energi_kwh"}.issubset(resampled_df.columns):
        resampled_df["energi_avvik_kwh"] = resampled_df["energi_kwh"] - resampled_df["energi_kwh_teor"]

    return resampled_df

#============================
#UI HJELPEFUNKSJONER & THEMES
#============================
def html_attr(tekst: str) -> str:
    return escape(str(tekst), {'"': '&quot;', "'": '&#39;'})

def live_help_icon(tekst: str) -> str:
    return f'<span class="live-tooltip" data-tip="{html_attr(tekst)}">?</span>'

def apply_light_theme() -> None:
    st.markdown(
        """
<style>
    * { transition: none !important; animation: none !important; }
    .block-container { padding-top: 1.05rem !important; padding-bottom: 0.45rem !important; }
    .main-title { margin: 0.08rem 0 0.35rem 0 !important; line-height: 1.05 !important; font-weight: 700 !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        opacity: 1 !important;
        transition: none !important;
    }
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    .live-stats-grid,
    .live-energy-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        column-gap: 2.4rem;
    }
    .live-stats-grid {
        row-gap: 2.10rem;
        margin: 0.55rem 0 2.10rem 0;
    }
    .live-energy-grid {
        margin: 0 0 3.15rem 0;
    }
    .live-stat-label {
        color: inherit;
        font-size: 0.8rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0.42rem;
        display: flex;
        align-items: center;
        gap: 0.32rem;
    }
    .live-stat-value {
        color: inherit;
        font-size: 2.15rem;
        line-height: 1.05;
        font-weight: 400;
        letter-spacing: 0;
    }
    .live-tooltip {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 0.82rem;
        height: 0.82rem;
        border: 1px solid #8b949e;
        border-radius: 999px;
        color: #8b949e;
        font-size: 0.58rem;
        line-height: 1;
        cursor: help;
        flex: 0 0 auto;
    }
    .live-tooltip::after {
        content: attr(data-tip);
        position: absolute;
        left: -0.25rem;
        top: 1.15rem;
        z-index: 1000;
        width: max-content;
        max-width: 36rem;
        white-space: normal;
        background: #2b2d38;
        color: white;
        padding: 0.55rem 0.7rem;
        border-radius: 0.35rem;
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1.45;
        box-shadow: 0 0.35rem 1.2rem rgba(0,0,0,0.25);
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
    }
    .live-stat:nth-child(even) .live-tooltip::after {
        left: auto;
        right: -0.25rem;
    }
    .live-tooltip:hover::after {
        opacity: 1;
        visibility: visible;
    }
    [data-testid="stSidebarHeader"] {
    height: 3rem !important;
    min-height: 0rem !important;
    padding: 0rem !important;
    margin: 0rem !important;
    }
    
    div[data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }
</style>
""", unsafe_allow_html=True)

def apply_screen_theme() -> None:
    st.markdown(
        """
<style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] { height: 100vh !important; overflow: hidden !important; }
    .block-container { max-height: 100vh !important; padding-top: 0.35rem !important; padding-bottom: 0.15rem !important; overflow: hidden !important; }
    .screen-title { margin: 0.02rem 0 0.08rem 0 !important; line-height: 1.06 !important; font-weight: 700 !important; }
    .screen-subtitle { margin: 0 0 0.18rem 0 !important; color: #94a3b8 !important; font-size: 1.06rem !important; }
    
    .screen-metrics { border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; background: rgba(255,255,255,0.05); padding: 0.42rem 0.65rem; margin-bottom: 0.26rem; }
    .screen-top-row { border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; background: rgba(255,255,255,0.05); padding: 0.48rem 0.7rem; margin-bottom: 0.25rem; display: flex; justify-content: space-between; align-items: center; }
    
    .screen-temp-now { display: flex; align-items: flex-start; gap: 0.2rem; }
    .screen-power-label { color: #cbd5e1; font-size: 0.78rem; font-weight: 600; margin-bottom: 0.04rem; }
    
    .screen-temp-value { color: #38bdf8; font-size: 2.4rem; font-weight: 700; line-height: 1; }
    .screen-temp-unit { color: #38bdf8; font-size: 1.05rem; font-weight: 700; line-height: 1.35; margin-top: 0.16rem; }
    
    .screen-right-info { text-align: right; }
    .screen-right-time { color: #94a3b8; font-size: 0.95rem; margin-bottom: 0.08rem; }
    .screen-right-status { color: #38bdf8; font-size: 1.02rem; font-weight: 700; }
    
    .screen-m-row { display: flex; justify-content: space-between; align-items: center; padding: 0.12rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); font-size: 0.97rem; }
    .screen-m-row:last-child { border-bottom: none; }
    .screen-m-label { color: #cbd5e1; }
    .screen-m-value { color: #38bdf8; font-weight: 700; }
    div[data-testid="stButton"] > button {
    padding: 0.2rem 0.8rem !important;
    font-size: 0.85rem !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] { background: #38bdf8 !important; color: #0f172a !important; border: 1px solid #38bdf8 !important; }
    [data-testid="stHeader"] {
    display: none !important;
    }
    [data-testid="stMetricLabel"] {
    white-space: normal !important;
    word-wrap: break-word !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        opacity: 1 !important;
        transition: none !important;
    }
    div[data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }
</style>
""", unsafe_allow_html=True)

def format_norsk_dato(dato: pd.Timestamp) -> str:
    ukedager = ["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"]
    maneder = ["januar", "februar", "mars", "april", "mai", "juni", "juli", "august", "september", "oktober", "november", "desember"]
    return f"{ukedager[dato.weekday()].capitalize()} {dato.day}. {maneder[dato.month - 1]} {dato.year}"

def format_ukedag_tid(ts: pd.Timestamp) -> str:
    ukedager = ["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"]
    return f"{ukedager[ts.weekday()]} kl. {ts:%H:%M}"

def filtrer_periode(dataframe: pd.DataFrame, periode: str, visning: str) -> pd.DataFrame:
    if dataframe.empty: return dataframe
    slutt = dataframe.index.max()
    
    #MODUS 1: ENERGI (kWh)
    if visning == "Energi (kWh)":
        df_hourly = dataframe.resample("1h").mean()
        if "24t" in periode: 
            start = slutt - pd.Timedelta(days=1)
            df_slice = df_hourly[df_hourly.index >= start.floor("1h")]
            idx = pd.date_range(start.floor("1h"), slutt.ceil("1h"), freq="1h")
            return df_slice.reindex(idx)
        if "7d" in periode:
            start = slutt - pd.Timedelta(days=7)
            df_slice = df_hourly[df_hourly.index >= start.floor("D")].resample("D").sum(min_count=1)
            idx = pd.date_range(start.floor("D"), slutt.ceil("D"), freq="D")
            return df_slice.reindex(idx)
        if "30d" in periode:
            start = slutt - pd.Timedelta(days=30)
            df_slice = df_hourly[df_hourly.index >= start.floor("D")].resample("D").sum(min_count=1)
            idx = pd.date_range(start.floor("D"), slutt.ceil("D"), freq="D")
            return df_slice.reindex(idx)
        if "År" in periode:
            start = slutt - pd.Timedelta(days=365)
            df_slice = df_hourly[df_hourly.index >= start.replace(day=1)].resample("MS").sum(min_count=1)
            idx = pd.date_range(start.replace(day=1), slutt, freq="MS")
            return df_slice.reindex(idx)

    #MODUS 2: EFFEKT (kW)
    else:
        if "24t" in periode:
            start = (slutt - pd.Timedelta(days=1)).floor("15min")
            df_slice = dataframe[dataframe.index >= start].resample("15min").mean()
            idx = pd.date_range(start, slutt.floor("15min"), freq="15min")
            return df_slice.reindex(idx)
        if "7d" in periode:
            start = (slutt - pd.Timedelta(days=7)).floor("1h")
            df_slice = dataframe[dataframe.index >= start].resample("1h").mean()
            idx = pd.date_range(start, slutt.floor("1h"), freq="1h")
            return df_slice.reindex(idx)
        if "30d" in periode:
            start = slutt - pd.Timedelta(days=30)
            df_slice = dataframe[dataframe.index >= start].resample("D").max()
            idx = pd.date_range(start.normalize(), slutt.normalize(), freq="D")
            return df_slice.reindex(idx)
        if "År" in periode:
            start = slutt - pd.Timedelta(days=365)
            df_slice = dataframe[dataframe.index >= start].resample("W").max()
            idx = pd.date_range(start.normalize(), slutt.normalize(), freq="W")
            return df_slice.reindex(idx)
            
    return dataframe

#===========
#SKJERMMODUS
#===========
def render_skjermmodus(dataframe: pd.DataFrame, vaer: dict) -> None:
    apply_screen_theme()

    col_knapp, col_tittel, col_tom = st.columns([1, 3, 1])
    
    with col_knapp:
        st.write("")
        st.write("")
        if st.button("← Tilbake", type="primary"):
            st.session_state.skjermmodus = False
            st.rerun()
    with col_tittel:
        st.markdown(
            '<h1 class="main-title screen-title" style="text-align: center;">PV-ytelse NTNU Gløshaugen</h1>',
            unsafe_allow_html=True)
    #col_tom er tom, men balanserer layouten så tittelen blir sentrert
    if dataframe.empty:
        st.warning("Ingen data å vise på skjerm.")
        return

    #====================================
    #Del 1: Oppdaterer live sensorverdier
    #====================================
    @st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
    def vis_live_tv_bokser():
        live_df = hent_live_data_kjapt()

        if live_df.empty:
            st.warning("Laster inn sensordata...")
            return
        #velger de nyeste verdiene
        siste_ghi = live_df['irradiance_wm2'].dropna()
        siste_ute = live_df['ambient_temp_c'].dropna()
        siste_maalt_t = live_df['maalt_panel_temp_c'].dropna()
        siste_effekt = live_df['effekt_kw'].dropna()
        siste_sim_effekt = live_df['effekt_kw_teor'].dropna()

        p_navaerende = float(siste_effekt.iloc[-1]) if not siste_effekt.empty else 0.0
        p_simulert = float(siste_sim_effekt.iloc[-1]) if not siste_sim_effekt.empty else 0.0
        effektavvik = p_simulert - p_navaerende
        ghi_naa = float(siste_ghi.iloc[-1]) if not siste_ghi.empty else 0.0
        maalt_t = float(siste_maalt_t.iloc[-1]) if not siste_maalt_t.empty else 0.0

        vaer_tekst = oversett_vaerkode(vaer.get("weathercode", -1)) if vaer else "Ukjent vær"

        #Øverste "boks" på TV skjerm modus
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1, 1, 1, 0.5])
            with c1:
                st.metric("Simulert effekt (Tvilling)", f"{p_simulert:.2f} kW")
            with c2:
                st.metric("Målt effekt (API)", f"{p_navaerende:.2f} kW")
            with c3:
                st.metric("Residual effekt", f"{effektavvik:.2f} kW")
            with c4:
                st.markdown(f"""
                <div style='text-align: right;'>
                    <div style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 0.2rem;'>{format_ukedag_tid(pd.Timestamp.now(tz=SITE_TZ))}</div>
                    <div style='color: #cbd5e1; font-size: 1.1rem; margin-bottom: 0.2rem;'>{vaer_tekst}</div>
                </div>
                """, unsafe_allow_html=True)

        #Sensorverdiene
        with st.container(border=True):
            m1, m2, m3, _ = st.columns([1, 1, 1, 0.5])
            m1.metric("Solinnstråling", f"{ghi_naa:.0f} W/m²")
            m2.metric("Modultemperatur", f"{maalt_t:.1f} °C", help=MODULTEMPERATUR_HELP)
            m3.metric("Omgivelsestemperatur", f"{float(siste_ute.iloc[-1]):.1f} °C" if not siste_ute.empty else "--")

    vis_live_tv_bokser()

    #==============================
    #oversiktsgraf målt vs simulert
    #==============================
    @st.fragment(run_every=f"{FULL_REFRESH_SECONDS}s")
    def vis_live_tv_graf():
            live_df = hent_historisk_data()
            
            na_tid = pd.Timestamp.now(tz=SITE_TZ)
            siste_tid = live_df.index.max()
            if na_tid > siste_tid: na_tid = siste_tid
    
            alle_dager = pd.DatetimeIndex(live_df.index.normalize().unique()).sort_values()
            aktiv_dag = alle_dager[-1] if len(alle_dager) > 0 else pd.Timestamp.today(tz=SITE_TZ)
            
            dag_start = aktiv_dag
            dag_slutt = dag_start + pd.Timedelta(hours=23, minutes=59) 
            
            #hent ut dagens data
            dag_df = live_df.loc[dag_start:siste_tid].copy()
            #5 min sampling
            time_5 = pd.date_range(dag_start, dag_slutt, freq="5min")
            #resampler og reindexerer
            dag_5 = dag_df.resample("5min").mean().reindex(time_5)
            #fremtid = NaN 
            dag_5.loc[dag_5.index > na_tid, 'effekt_kw_teor'] = np.nan
            #sampler 15min for solaredge data
            time_15 = pd.date_range(dag_start, dag_slutt, freq="15min")
            dag_15 = dag_df.resample("15min").mean().reindex(time_15)
            #fremtiden = NaN
            dag_15.loc[dag_15.index > na_tid, 'effekt_kw'] = np.nan
    
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dag_15.index, 
                y=dag_15["effekt_kw"], 
                mode="lines + markers", 
                name="Målt (SolarEdge)", 
                connectgaps=False, 
                line = dict(color="#00c8f5", width = 1),
                marker=dict(color="#00c8f5", size=8),
                hovertemplate="%{y:.2f} kW"
            ))

            fig.add_trace(go.Scatter(
                x=dag_5.index, 
                y=dag_5["effekt_kw_teor"], 
                mode="lines + markers", 
                name="Simulert (Digital Tvilling)", 
                connectgaps=False, 
                line = dict(color="#ff4500", width = 1),
                marker=dict(color="#ff4500", size = 5), 
                hovertemplate="%{y:.2f} kW"
            ))
            
            fig.update_layout(
                title=f"Effektproduksjon {format_norsk_dato(aktiv_dag)}", 
                xaxis_title="Tidspunkt", 
                yaxis_title="Effekt (kW)", 
                height=470, 
                margin=dict(l=46, r=12, t=44, b=32), 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",  
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            fig.update_xaxes(
                gridcolor="rgba(255,255,255,0.1)", 
                tickangle=45, 
                tickformat="%H:%M",
                range=[dag_start, dag_slutt] 
            )
            
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)", autorange=True, fixedrange=False, rangemode="tozero")
            
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", key="tv_graf")
    vis_live_tv_graf()

#============
#HOVEDPROGRAM
#============
apply_light_theme()

#Hent værmelding som brukes til pynt
vaer_data = hent_live_vaer()

#HENT DATA (Kjører PVLIB)
with st.spinner("Henter historisk data og kjører Digital Tvilling..."):
    df = hent_historisk_data()

if df.empty:
    st.error("Ingen data funnet fra datalogger eller API. Venter...")
    st.stop()

if "skjermmodus" not in st.session_state:
    st.session_state.skjermmodus = False

if st.session_state.skjermmodus:
    render_skjermmodus(df, vaer_data)
    st.stop()

#Auto-refresh logikk
#Timestamp-sjekken er nødvendig: uten den vil st.rerun() starte en ny app-render, som re-registrerer fragmentet og trigger det umiddelbart igjen → uendelig loop.
if "siste_hovedoppdatering_ts" not in st.session_state:
    st.session_state.siste_hovedoppdatering_ts = pd.Timestamp.now().timestamp()

@st.fragment(run_every=f"{FULL_REFRESH_SECONDS}s")
def auto_refresh_hovedvisning():
    na_ts = pd.Timestamp.now().timestamp()
    sist_ts = st.session_state.get("siste_hovedoppdatering_ts", 0.0)
    if na_ts - sist_ts >= FULL_REFRESH_SECONDS:
        st.session_state.siste_hovedoppdatering_ts = na_ts
        hent_historisk_data.clear()
        st.rerun()

auto_refresh_hovedvisning()

#STANDARD WEB VISNING
st.sidebar.header("Været nå (Gløshaugen)")

@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def vis_live_vaer_sidebar():
    værdata = hent_live_vaer()
    live_df = hent_live_data_kjapt()
    siste_vind, _ = hent_siste_gyldige_verdi(live_df, 'wind_speed')

    #Bruk Frost API-måling når tilgjengelig, fall tilbake på MET.no-prognose
    if siste_vind is None and værdata:
        siste_vind = værdata.get("windspeed")

    if værdata and værdata.get("forecast_time") is not None:
        st.caption(f"Tidspunkt: {formater_sidebar_tidspunkt(værdata.get('forecast_time'))}")

    if værdata:
        st.markdown(f"**{oversett_vaerkode(værdata.get('weathercode'))}**")
        st.metric("Vind", formater_sidebar_verdi(siste_vind, "m/s", 1))
    else:
        st.caption("Kunne ikke hente værvarsel akkurat nå.")

    st.caption("MET.no · Frost API, stasjon Voll (vind)")

with st.sidebar:
    vis_live_vaer_sidebar()

st.sidebar.markdown("---")

st.sidebar.header("Sensormålinger (Taket)")

@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def vis_live_sensorer():
    live_df = hent_live_data_kjapt()

    siste_ghi, tid_ghi = hent_siste_gyldige_verdi(live_df, 'irradiance_wm2')
    siste_ute, tid_ute = hent_siste_gyldige_verdi(live_df, 'ambient_temp_c')
    siste_maalt_t, tid_maalt_t = hent_siste_gyldige_verdi(live_df, 'maalt_panel_temp_c')

    tilgjengelige_tider = [tid for tid in [tid_ghi, tid_ute, tid_maalt_t] if tid is not None]
    if tilgjengelige_tider:
        st.caption(f"Tidspunkt: {max(tilgjengelige_tider):%d.%m.%Y %H:%M:%S}")
    else:
        st.caption("Ingen sensordata tilgjengelig ennå.")

    st.metric("Pyranometer", formater_sidebar_verdi(siste_ghi, "W/m²", 0))
    st.metric("Modultemperatur", formater_sidebar_verdi(siste_maalt_t, "°C", 1), help=MODULTEMPERATUR_HELP)
    st.metric("Omgivelsestemperatur", formater_sidebar_verdi(siste_ute, "°C", 1))

    #Advarsel hvis SAPM-fallback er aktiv for siste datapunkt
    if 'celltemp_sapm_fallback' in live_df.columns:
        siste_fallback = live_df['celltemp_sapm_fallback'].dropna()
        if not siste_fallback.empty and bool(siste_fallback.iloc[-1]):
            st.warning("⚠️ Modultemperatursensor ikke tilgjengelig — bruker SAPM-estimat for simulering.")

with st.sidebar:
    vis_live_sensorer()

    #Knapp bytte visningsmodus
    st.markdown("---")
    st.markdown("### Visning")
    if st.button("Skjermmodus", use_container_width=True):
        st.session_state.skjermmodus = True
        st.rerun()
st.markdown('<h1 class="main-title">Digital Tvilling: NTNU Gløshaugen</h1>', unsafe_allow_html=True)

tab_vis, tab_det = st.tabs(["Oversikt", "Detaljer & Statistikk"])

with tab_vis:
    venstre, hoyre = st.columns([1.1, 1.9], gap="large")

    with hoyre:
        valg_col1, valg_col2 = st.columns(2)
        with valg_col1:
            visningstype = st.radio("Datatype", ["Effekt (kW)", "Energi (kWh)"], horizontal=True)
        with valg_col2:
            periode = st.radio("Periode", PERIODE_OPTIONS, horizontal=True, index=0)
            
        vis_df = filtrer_periode(df, periode, visningstype)
        
        if not vis_df.empty:
            fig = go.Figure()
            
            if visningstype == "Energi (kWh)":
                fig.add_trace(go.Bar(x=vis_df.index, y=vis_df["effekt_kw"], name="Målt (SolarEdge)", marker_color="#00c8f5", marker_line_width=0, hovertemplate="%{y:.1f} kWh"))
                fig.add_trace(go.Bar(x=vis_df.index, y=vis_df["effekt_kw_teor"], name="Simulert (Digital Tvilling)", marker_color="#ff4500", marker_line_width=0, hovertemplate="%{y:.1f} kWh"))
                fig.update_layout(barmode="group", bargap=0.2)
                y_tittel = "Energi (kWh)"
                graf_tittel = "Faktisk vs Simulert Energiproduksjon"
                
                if "År" in periode: fig.update_xaxes(tickformat="%b %Y")
                elif "30d" in periode or "7d" in periode: fig.update_xaxes(tickformat="%d. %b")
                else: fig.update_xaxes(tickformat="%H:%M")
                
            else:
                if "24t" in periode or "7d" in periode:
                    fig.add_trace(go.Scatter(x=vis_df.index, y=vis_df["effekt_kw"], name="Målt (SolarEdge)", mode="lines", line=dict(color="#00c8f5", width=2), hovertemplate="%{y:.2f} kW"))
                    fig.add_trace(go.Scatter(x=vis_df.index, y=vis_df["effekt_kw_teor"], name="Simulert (Digital Tvilling)", mode="lines", line=dict(color="#ff4500", width=2), hovertemplate="%{y:.2f} kW"))
                    y_tittel = "Effekt (kW)"
                else:
                    fig.add_trace(go.Bar(x=vis_df.index, y=vis_df["effekt_kw"], name="Målt Maks (SolarEdge)", marker_color="#00c8f5", marker_line_width=0, hovertemplate="%{y:.2f} kW"))
                    fig.add_trace(go.Bar(x=vis_df.index, y=vis_df["effekt_kw_teor"], name="Simulert Maks (Tvilling)", marker_color="#ff4500", marker_line_width=0, hovertemplate="%{y:.2f} kW"))
                    fig.update_layout(barmode="group", bargap=0.3)
                    y_tittel = "Maks Effekt (kW)"
                
                graf_tittel = "Faktisk vs Simulert Effektproduksjon"
                
            fig.update_layout(
                title=graf_tittel, 
                yaxis_title=y_tittel, 
                height=350, 
                margin=dict(l=54, r=18, t=48, b=38), 
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                hovermode="x unified" 
            )
            
            fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)", tickangle=0)
            fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown(f"**Oppsummering for valgt periode ({periode})**")
            
            df_energi = filtrer_periode(df, periode, "Energi (kWh)")
            tot_maalt = df_energi["effekt_kw"].sum()
            tot_sim = df_energi["effekt_kw_teor"].sum()
            avvik = tot_maalt - tot_sim
            ratio = 0.0 if tot_sim <= 0 else (tot_maalt / tot_sim) * 100
            
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Totalt Målt Energi", f"{tot_maalt:.1f} kWh")
            sc2.metric("Totalt Simulert Energi", f"{tot_sim:.1f} kWh")
            sc3.metric("Avvik", f"{avvik:.1f} kWh")
            sc4.metric("Ytelsesratio", f"{ratio:.1f} %", help=YTELSESRATIO_PERIODE_HELP)
            

        else:
            st.warning("Ikke nok data for valgt periode. Venter på mer data fra loggeren.")

    with venstre:
        st.subheader("Sanntidsdata")
        
        #Setter denne til 7 sekunder (LIVE_REFRESH_SECONDS + 2) for å spre lasten
        @st.fragment(run_every=f"{LIVE_REFRESH_SECONDS + 2}s")
        def vis_live_sanntidsdata():
            live_df = hent_live_data_kjapt()
            
            if live_df.empty or 'effekt_kw' not in live_df.columns:
                st.info("Venter på sensordata...")
                return
            
            siste_effekt = live_df['effekt_kw'].dropna()
            siste_sim_effekt = live_df['effekt_kw_teor'].dropna()
            siste_ghi = live_df['irradiance_wm2'].dropna()
            
            p_faktisk = float(siste_effekt.iloc[-1]) if not siste_effekt.empty else 0.0
            p_simulert = float(siste_sim_effekt.iloc[-1]) if not siste_sim_effekt.empty else 0.0
            effektavvik = p_simulert - p_faktisk
            ghi_naa = float(siste_ghi.iloc[-1]) if not siste_ghi.empty else 0.0
            
            ratio = 0.0 if p_simulert <= 0 else (p_faktisk / p_simulert) * 100
            
            if ghi_naa < 20:
                ratio_tekst = "-"
            elif ghi_naa < 50:
                #For lite sol til å gi meningsfull ratio
                ratio_tekst = f"{ratio:.1f}%"
            else:
                ratio_tekst = f"{ratio:.1f}%"

            st.markdown(
                f"""
                <div class="live-stats-grid">
                    <div class="live-stat">
                        <div class="live-stat-label">Simulert effekt</div>
                        <div class="live-stat-value">{p_simulert:.2f} kW</div>
                    </div>
                    <div class="live-stat">
                        <div class="live-stat-label">Målt effekt</div>
                        <div class="live-stat-value">{p_faktisk:.2f} kW</div>
                    </div>
                    <div class="live-stat">
                        <div class="live-stat-label">Ytelsesratio {live_help_icon(YTELSESRATIO_LIVE_HELP)}</div>
                        <div class="live-stat-value">{html_attr(ratio_tekst)}</div>
                    </div>
                    <div class="live-stat">
                        <div class="live-stat-label">Residual effekt</div>
                        <div class="live-stat-value">{effektavvik:.2f} kW</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            #Advarsel hvis SAPM-fallback er aktiv for siste datapunkt
            if 'celltemp_sapm_fallback' in live_df.columns:
                siste_fallback = live_df['celltemp_sapm_fallback'].dropna()
                if not siste_fallback.empty and bool(siste_fallback.iloc[-1]):
                    st.warning(
                        "⚠️ **Modultemperatursensor ikke tilgjengelig** — "
                        "simulert effekt er basert på SAPM-estimat for modultemperatur "
                        "(lufttemp + innstråling + vind) og kan avvike noe fra faktisk verdi."
                    )

            midnatt = pd.Timestamp.now(tz = SITE_TZ).normalize()
            dagens_df = live_df[live_df.index >= midnatt]

            #Resample begge til 15-min intervaller før integrering, så dt antakelsen (0.25 h) er konsistent og NaN gaps håndteres likt
            daily_15 = dagens_df.resample("15min").mean()
            energi_faktisk  = float(daily_15["effekt_kw"].sum() * 0.25)
            energi_simulert = float(daily_15["effekt_kw_teor"].sum() * 0.25)

            #TMY-basert referanse (typisk dag fra PVGIS)
            tmy_dagsprod = hent_tmy_dagsproduksjon()
            idag = pd.Timestamp.now(tz=SITE_TZ)
            tmy_forventet = tmy_dagsprod.get((idag.month, idag.day))

            if tmy_forventet and tmy_forventet > 0:
                daglig_prosent   = (energi_faktisk / tmy_forventet) * 100
                referanse_kwh    = tmy_forventet
                referanse_tekst  = f"Typisk dag: {tmy_forventet:.1f} kWh"
                tittel_tekst     = (
                    f"Ytelse i dag: {daglig_prosent:.1f}% av typisk dag"
                    f"  ({energi_faktisk:.1f} / {tmy_forventet:.1f} kWh)"
                )
            else:
                #Fallback: bruk simulert hvis TMY ikke er tilgjengelig
                daglig_prosent  = (energi_faktisk / energi_simulert * 100) if energi_simulert > 0 else 0.0
                referanse_kwh   = energi_simulert
                referanse_tekst = f"Simulert: {energi_simulert:.1f} kWh"
                tittel_tekst    = (
                    f"Ytelse i dag: {daglig_prosent:.1f}% av simulert"
                    f"  ({energi_faktisk:.1f} / {energi_simulert:.1f} kWh)"
                )
                st.caption("⚠️ TMY-data ikke tilgjengelig — bruker simulert som referanse. "
                           "Sjekk terminalvinduet for feilmelding.")

            st.markdown(
                f"""
                <div class="live-energy-grid">
                    <div class="live-stat">
                        <div class="live-stat-label">Energi i dag — Målt</div>
                        <div class="live-stat-value">{energi_faktisk:.1f} kWh</div>
                    </div>
                    <div class="live-stat">
                        <div class="live-stat-label">Forventet (typisk dag) {live_help_icon(TYPISK_DAG_HELP)}</div>
                        <div class="live-stat-value">{referanse_kwh:.1f} kWh</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            maks_verdi = max(energi_faktisk, referanse_kwh) if referanse_kwh > 0 else max(energi_faktisk, 1)

            #Blå stolpe = forventet full dag (TMY), rød strek = målt så langt
            fig_bar = go.Figure(go.Bar(
                x=[referanse_kwh],
                y=[""],
                orientation='h',
                marker=dict(color="#00c8f5"),
                width=0.4,
                hovertemplate=f"{referanse_tekst}<extra></extra>"
            ))
            fig_bar.add_vline(
                x=energi_faktisk,
                line_width=2,
                line_dash="dash",
                line_color="#ff4500",
                annotation_text=f"Målt: {energi_faktisk:.1f} kWh",
                annotation_position="bottom right",
            )
            fig_bar.update_layout(
                barmode="stack",
                title=tittel_tekst,
                height=90,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                           range=[0, maks_verdi * 1.15]),
                yaxis=dict(showticklabels=False, showgrid=False),
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="daglig_ytelse_bar")
        vis_live_sanntidsdata()

with tab_det:
    
    st.subheader("Digital Tvilling - Informasjon")
    st.markdown("""
    Dette dashboardet sammenligner live data fra SolarEdge-anlegget på NTNU Elektro med en **fysikkbasert Digital Tvilling**.
    Modellen kjører kontinuerlig en ModelChain via pvlib, som løser:
    * **CEC Single Diode Model** (6 parametere)
    * **Sandia Inverter Model** (SE20K coefficients scaled to SE17K via datasheet)
    * **Erbs-modellen** for DNI/DHI dekomponering
    """)
    st.subheader("Anleggsdata")
    st.markdown(
        f"**Installert effekt:** {SYSTEM_STC_KW:.2f} kWp  \n"
        f"**Paneler:** 62 x Panasonic 325W  \n"
        f"**Inverter:** SE17K"
    )
    st.markdown("---")
    st.subheader("Eksporter Historisk Data")
    st.markdown(
        "Filen bygges fra kombinert historikk og inneholder simulert effekt, målt effekt, avvik, temperatur og "
        "solinnstråling."
        "Energikolonner vises ved dag-, uke- og månedsoppløsning og følger samme beregningsprinsipp som energigrafen i dashboardet."
    )
    def hent_alt_fra_fil_for_periode(start_dato=None, slutt_dato=None):
        """Henter ferdig kombinert historikk for eksport."""
        try:
            full_df = hent_historiske_data_for_eksport()
            if full_df.empty:
                return pd.DataFrame()
            
            if start_dato and slutt_dato:
                start = til_naiv_lokal_tid(pd.Timestamp(start_dato))
                slutt = til_naiv_lokal_tid(pd.Timestamp(slutt_dato))
                mask = (full_df.index >= start) & (full_df.index <= slutt)
                return full_df.loc[mask]
            
            return full_df
        except Exception as e:
            st.error(f"Kunne ikke lese master-loggfil: {e}")
            return pd.DataFrame()
    @st.fragment
    def vis_nedlastingsmeny():
        eksport_historikk = hent_historiske_data_for_eksport()
        if eksport_historikk.empty:
            st.warning("Ingen historiske eksportdata tilgjengelig.")
            return
        siste_tidspunkt = eksport_historikk.index.max()
        tilgjengelige_ar = list(range(eksport_historikk.index.min().year, max(pd.Timestamp.today().year, siste_tidspunkt.year) + 1))
        siste_iso = siste_tidspunkt.isocalendar()
        #Nedlastingsalternativer inkludert Livstid
        nedlastingstype = st.radio(
            "Hvor mye data vil du laste ned?", 
            ["Enkelt dag", "Hel uke", "Hel måned", "Helt år", "Livstid (Hele loggen)"], 
            horizontal=True
        )
        #Bestemmer hvilke oppløsninger som gir mening for valgt periode
        if nedlastingstype == "Enkelt dag":
            oppløsning_valg = ["15 minutter", "1 time"]
        elif nedlastingstype == "Hel uke":
            oppløsning_valg = ["1 time", "1 dag"]
        elif nedlastingstype == "Hel måned":
            oppløsning_valg = ["1 time", "1 dag", "1 uke"]
        elif nedlastingstype == "Helt år":
            oppløsning_valg = ["1 dag", "1 uke", "1 måned"]
        else: # Livstid
            oppløsning_valg = ["1 dag", "1 uke", "1 måned"]

        #Lar brukeren velge nedlastingsfrekvens
        valgt_oppløsning = st.selectbox("Velg dataoppløsning (nedsampling)", oppløsning_valg)
        nedlasting_df = pd.DataFrame()
        nedlasting_etikett = ""
        start = None
        slutt = None

        #Hent data basert på valgt periode
        if nedlastingstype == "Enkelt dag":
            valgt_dag = st.date_input(
                "Velg dato",
                value=siste_tidspunkt.date(),
                max_value=pd.Timestamp.today().date()
            )
            #Definerer start og slutt for hele døgnet
            start = pd.Timestamp(valgt_dag, tz = SITE_TZ).replace(hour=0, minute=0, second=0)
            slutt = pd.Timestamp(valgt_dag, tz = SITE_TZ).replace(hour=23, minute=59, second=59)
            
            nedlasting_df = hent_alt_fra_fil_for_periode(start, slutt)
            nedlasting_etikett = f"dag_{start.strftime('%Y-%m-%d')}"

        elif nedlastingstype == "Hel uke":
            uke_col1, uke_col2 = st.columns(2)
            with uke_col1:
                valgt_uke_ar = st.selectbox("Velg år", tilgjengelige_ar, index=len(tilgjengelige_ar)-1)
            uke_ops = list(range(1, antall_iso_uker(valgt_uke_ar) + 1))
            with uke_col2:
                valgt_uke = st.selectbox("Velg uke", uke_ops, index=int(siste_iso.week)-1)
            
            #Beregner mandag og søndag i valgt uke
            start = pd.Timestamp.fromisocalendar(valgt_uke_ar, valgt_uke, 1).tz_localize(SITE_TZ).replace(hour=0, minute=0, second=0)
            slutt = start + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
            
            nedlasting_df = hent_alt_fra_fil_for_periode(start, slutt)
            nedlasting_etikett = f"uke_{valgt_uke_ar}_Uke{valgt_uke:02d}"

        elif nedlastingstype == "Hel måned":
            måned_col1, måned_col2 = st.columns(2)
            with måned_col1:
                valgt_måned_ar = st.selectbox("Velg år", tilgjengelige_ar, index=len(tilgjengelige_ar)-1, key="dl_maned_ar")
            with måned_col2:
                valgt_måned = st.selectbox("Velg måned", list(range(1, 13)), index=siste_tidspunkt.month - 1, format_func=lambda m: MÅNED_NAVN[m - 1])
            
            #Beregner første og siste sekund i måneden
            start = pd.Timestamp(year=valgt_måned_ar, month=valgt_måned, day=1, tz = SITE_TZ).replace(hour=0, minute=0, second=0)
            slutt = (start + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)
            
            nedlasting_df = hent_alt_fra_fil_for_periode(start, slutt)
            nedlasting_etikett = f"måned_{valgt_måned_ar}_{valgt_måned:02d}"

        elif nedlastingstype == "Helt år":
            valgt_ar = st.selectbox("Velg år", tilgjengelige_ar, index=len(tilgjengelige_ar)-1, key="dl_ar_ar")
            
            #Beregner fra 1. januar til 31. desember
            start = pd.Timestamp(year=valgt_ar, month=1, day=1, tz = SITE_TZ).replace(hour=0, minute=0, second=0)
            slutt = pd.Timestamp(year=valgt_ar, month=12, day=31, tz = SITE_TZ).replace(hour=23, minute=59, second=59)
            nedlasting_df = hent_alt_fra_fil_for_periode(start, slutt)
            nedlasting_etikett = str(valgt_ar)
            
        else: #Livstid (Hele loggen)
            #kaller funksjonen uten start/slutt for å få hele filen
            nedlasting_df = hent_alt_fra_fil_for_periode() 
            nedlasting_etikett = "Livstid"
            if not nedlasting_df.empty:
                start = pd.Timestamp(nedlasting_df.index.min())
                slutt = pd.Timestamp(nedlasting_df.index.max())
        #komprimeringen (Downsampling)
        if start is not None and slutt is not None:
            nedlasting_df = resample_nedlasting_dataframe(nedlasting_df, valgt_oppløsning, start, slutt)
        if nedlasting_df.empty:
            st.warning("Ingen data funnet for valgt periode. Prøv et annet datoområde.")
        else:
            st.info(f"Klar for nedlasting: Filen inneholder {len(nedlasting_df)} rader med data for valgt periode.")
            st.download_button(
                label="Last ned Excel-fil",
                data=dataframe_til_excel_xml(nedlasting_df),
                file_name=f"NTNU_Digital_Tvilling_{nedlasting_etikett}.xml",
                mime="application/vnd.ms-excel",
                use_container_width=True,
                type="primary"
            )
    #Kjører selve fragmentet inni fanen
    vis_nedlastingsmeny()


