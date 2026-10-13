import pandas as pd
import numpy as np
import pvlib
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.ivtools.sdm import fit_cec_sam


SITE_LAT = 63.4183333
SITE_LON = 10.4008333
SITE_TZ = 'Europe/Oslo'
ALTITUDE = 60


class DigitalTwin:
    def __init__(self):
        self.site: Location    = Location(SITE_LAT, SITE_LON, tz=SITE_TZ, altitude=ALTITUDE)
        self.inverter          = None
        self.model: ModelChain = None
        self.shading_factor    = 0.554

    def build_model(self) -> None:
        
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
        self.inverter = se20k.copy()
        self.inverter['Paco'] = 17000                  #AC rating
        self.inverter['Pdco'] = se20k['Pdco'] * k      #Skalert DC input ved rated AC
        self.inverter['Vdco'] = 750                    #Nominell DC spenning
        self.inverter['Pso']  = se20k['Pso']  * k      #skalert oppstart-threshhold
        self.inverter['Pnt']  = 2.5                    #max nattestid-tap
        self.inverter['C0']   = se20k['C0']   / k      #invers-skalert
    

        arrays = [
            Array(FixedMount(15, 67),  module_parameters=module_parameters, temperature_model_parameters=temp_open, modules_per_string=12),
            Array(FixedMount(14, 247), module_parameters=module_parameters, temperature_model_parameters=temp_open, modules_per_string=12),
            Array(FixedMount(55, 157), module_parameters=module_parameters, temperature_model_parameters=temp_open, modules_per_string=7),
            Array(FixedMount(55, 157), module_parameters=module_parameters, temperature_model_parameters=temp_open, modules_per_string=7),
            Array(FixedMount(48, 246), module_parameters=module_parameters, temperature_model_parameters=temp_open, modules_per_string=12),
            Array(FixedMount(45, 180), module_parameters=module_parameters, temperature_model_parameters=temp_open, modules_per_string=12)
        ]
        
        system = PVSystem(arrays=arrays, inverter_parameters=self.inverter)

        self.model = ModelChain(system, self.site, aoi_model='physical', spectral_model='no_loss')


    def calculate_power_idk(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        #Felles pvlib-kjerne for bygg_ui_dataframe og kjor_pvlib_simulering.
        #Returnerer (simulert_ac_kw, sapm_fallback_maske).
        solpos = self.site.get_solarposition(df.index)
        irrad_components = pvlib.irradiance.erbs(df['GHI_avg'], solpos['zenith'], df.index)
        
        #CS241 måler modultemperatur (T_mod, bakflate). Konverter til celletemperatur
        #med SAPM: T_cell = T_mod + ΔT × (POA / 1000), der POA er for array 5 (tilt=48°, asimut=246°).
        sapm_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
        poa_cs241 = pvlib.irradiance.get_total_irradiance(
            surface_tilt=48, surface_azimuth=246,
            solar_zenith=solpos['apparent_zenith'], solar_azimuth=solpos['azimuth'],
            dni=irrad_components['dni'].fillna(0).clip(lower=0),
            ghi=df['GHI_avg'],
            dhi=irrad_components['dhi'].fillna(0).clip(lower=0),
        )['poa_global'].fillna(0).clip(lower=0)
    
        t_mod_cs241 = df['CellTemp_avg'].copy() if 'CellTemp_avg' in df.columns else pd.Series(np.nan, index=df.index)
        cell_temp = t_mod_cs241 + sapm_params['deltaT'] * poa_cs241 / 1000.0
    
        sapm_fallback_maske = t_mod_cs241.isna()   #True der CS241-sensor mangler data
        if sapm_fallback_maske.any():
            #Fallback: estimer T_cell direkte fra SAPM når CS241 er frakoblet
            cell_temp_sapm = pvlib.temperature.sapm_cell(
                poa_global=poa_cs241,
                temp_air=df['AirTemp_avg'].fillna(10),
                wind_speed=df['wind_speed'].fillna(1.0),
                **sapm_params
            )
            cell_temp = cell_temp.fillna(cell_temp_sapm)
    
        weather = pd.DataFrame({
            'ghi':              df['GHI_avg'],
            'dhi':              irrad_components['dhi'].fillna(0).clip(lower=0),
            'dni':              irrad_components['dni'].fillna(0).clip(lower=0),
            'temp_air':         df['AirTemp_avg'],
            'wind_speed':       df['wind_speed'],
            'cell_temperature': cell_temp,
        }, index=df.index)
    
        self.model.run_model(weather)
        dc_results = self.model.results.dc
    
        #Sandia inverter modell med SE17K skalerte parametre
        p_dc_per_array = [
            dc_results[0]['p_mp'],
            dc_results[1]['p_mp'],
            dc_results[2]['p_mp'] * self.shading_factor,
            dc_results[3]['p_mp'] * self.shading_factor,
            dc_results[4]['p_mp'],
            dc_results[5]['p_mp'],
        ]
        total_dc_watts = sum(p_dc_per_array)
        weighted_v_num = sum(dc_results[i]['v_mp'] * p_dc_per_array[i] for i in range(6))
        safe_total = total_dc_watts.clip(lower=1e-9)
        v_dc_est = (weighted_v_num / safe_total).fillna(self.inverter['Vdco'])
    
        simulert_ac_kw = pvlib.inverter.sandia(
            v_dc=v_dc_est,
            p_dc=total_dc_watts,
            inverter=self.inverter,
        ).fillna(0).clip(lower=0) / 1000.0
    
        return simulert_ac_kw, sapm_fallback_maske


    def calculate_power(self, df: pd.DataFrame) -> pd.DataFrame:
        weather = pd.DataFrame({
            'ghi':        df['ghi'].clip(lower=0),
            'dni':        df['dni'].clip(lower=0),
            'dhi':        df['dhi'].clip(lower=0),
            'temp_air':   df['temp_air'],
            'wind_speed': df['wind_speed'].clip(lower=0),
        }, index=df.index)

        self.model.run_model(weather)
        dc = self.model.results.dc

        p_dc = [
            dc[0]['p_mp'],
            dc[1]['p_mp'],
            dc[2]['p_mp'] * self.shading_factor,
            dc[3]['p_mp'] * self.shading_factor,
            dc[4]['p_mp'],
            dc[5]['p_mp'],
        ]

        total_dc = sum(p_dc)
        v_num    = sum(dc[i]['v_mp'] * p_dc[i] for i in range(6))
        safe_tot = total_dc.clip(lower=1e-9)
        v_dc     = (v_num / safe_tot).fillna(self.inverter['Vdco'])

        ac_kw = (pvlib.inverter.sandia(
            v_dc=v_dc, 
            p_dc=total_dc, 
            inverter=self.inverter
        ).fillna(0).clip(lower=0) / 1000.0)

        gruppert = ac_kw.groupby([ac_kw.index.month, ac_kw.index.day]).sum()

        df = ac_kw.to_frame(name='power_kw')

        return df
        