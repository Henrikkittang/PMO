


import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#######################
# Python Variables
#######################

def load_data():

    df_price = pd.read_csv('./data/price.csv')
    df_power_s1 = pd.read_csv('./data/power_s1.csv')
    df_power_s2 = pd.read_csv('./data/power_s2.csv')
    df_power_s3 = pd.read_csv('./data/power_s3.csv')


    df_power_s1.iloc[:48, df_power_s1.columns.get_loc('power_kw')] = (
        df_power_s3['power_kw'].iloc[:48].to_numpy()
    )

    df_power_s2.iloc[:48, df_power_s2.columns.get_loc('power_kw')] = (
        df_power_s3['power_kw'].iloc[:48].to_numpy()
    )

    power_dfs = {'s1': df_power_s1, 's2': df_power_s2, 's3': df_power_s3}


    n_steps = len(df_price)

    # PV production, indexed by (scenario, time)
    pv_data = {
        (s, t): float(power_dfs[s]['power_kw'].iloc[t])
        for s in power_dfs
        for t in range(n_steps)
    }

    return df_price['price_nok_kwh'], pv_data


def build_model(scenarios:list, pv_data, price):
        
    #######################
    # Setup
    #######################

    model = pyo.ConcreteModel()


   

    #######################
    # Constant Parameteres
    #######################

    # Time Steps
    model.T = pyo.RangeSet(0, len(price) - 1 ) # 96 [15 min]

    # Extra time step for energy integral
    model.T_battery = pyo.RangeSet(0, len(price))

    model.S = pyo.Set(initialize=scenarios)

    model.spot_price = pyo.Param(model.T, initialize=dict(enumerate(price)))

    model.P_line = pyo.Param(initialize=5)            # kW
    model.P_max = pyo.Param(initialize=5)             # kW
    model.battery_E_max = pyo.Param(initialize=20)    # kWh
    model.battery_E_0 = pyo.Param(initialize=0)       # kWh

    model.battery_eff =  pyo.Param(initialize=0.95)    # %
    model.bat_cycle_cost = pyo.Param(initialize=0.35)  # NOK/kWh 
    model.dt = pyo.Param(initialize=0.25)              # hours

   

    # Expected PV power at each time step [kW]
    model.PV_P_prod = pyo.Param(model.S, model.T, initialize=pv_data)


    #######################
    # Decision Variables
    #######################


    # Battery energy level 
    model.battery_E = pyo.Var(model.S, model.T_battery, initialize=0, bounds=(0, model.battery_E_max), domain=pyo.NonNegativeReals) # kWh

    # Power in/out of battery
    model.battery_charge_P   = pyo.Var(model.S, model.T, initialize=0, bounds=(0, model.P_max), domain=pyo.NonNegativeReals) # kW
    model.battery_discharge_P = pyo.Var(model.S, model.T, initialize=0, bounds=(0, model.P_max), domain=pyo.NonNegativeReals) # kW

    model.is_charging = pyo.Var(model.S, model.T, initialize=0, domain=pyo.Binary) # 0|1

    # Power in/out of grid
    model.grid_import_P = pyo.Var(model.S, model.T, initialize=0, bounds=(0, model.P_line), domain=pyo.NonNegativeReals) # kW
    model.grid_export_P = pyo.Var(model.S, model.T, initialize=0, bounds=(0, model.P_line), domain=pyo.NonNegativeReals) # kW

    model.loss_P = pyo.Var(model.S, model.T, initialize=0, bounds=(0, None),        domain=pyo.NonNegativeReals) # kW

    model.is_importing = pyo.Var(model.S, model.T, initialize=0, domain=pyo.Binary) # 0|1



    #######################
    # Constraints
    #######################
    # Battery capacity constraint
    def initial_battery_constraint(model, s):
        return model.battery_E[s, 0] == model.battery_E_0


    # Energy balance constraint
    def energy_balance_constraint(model, s, t):
        return (model.PV_P_prod[s, t] + model.grid_import_P[s, t] + model.battery_discharge_P[s, t]) == (model.battery_charge_P[s, t] + model.grid_export_P[s, t] + model.loss_P[s, t])


    # Charging/discharging constraint
    def charge_constraint(model, s, t):
        return model.battery_charge_P[s, t] <= model.P_max * model.is_charging[s, t]

    def discharge_constraint(model, s, t):
        return model.battery_discharge_P[s, t] <= model.P_max* (1 - model.is_charging[s, t])

    # Import/export constraint
    def import_constraint(model, s, t):
        return model.grid_import_P[s, t] <= model.P_line* model.is_importing[s, t]

    def export_constraint(model, s, t):
        return model.grid_export_P[s, t] <= model.P_line* (1 - model.is_importing[s, t])


    # Battery capacity constraint
    def battery_capacity_constraint(model, s, t):
        return model.battery_E[s, t + 1] == model.battery_E[s, t] + model.battery_eff*model.battery_charge_P[s, t]*model.dt - (1/model.battery_eff)*model.battery_discharge_P[s, t]*model.dt


    model.initial_battery        = pyo.Constraint(model.S, rule=initial_battery_constraint)
    model.energy_balance         = pyo.Constraint(model.S, model.T, rule=energy_balance_constraint)
    model.charge_limit           = pyo.Constraint(model.S, model.T, rule=charge_constraint)
    model.discharge_limit        = pyo.Constraint(model.S, model.T, rule=discharge_constraint)
    model.battery_energy_balance = pyo.Constraint(model.S, model.T, rule=battery_capacity_constraint)
    model.import_limit           = pyo.Constraint(model.S, model.T, rule=import_constraint)
    model.export_limit           = pyo.Constraint(model.S, model.T, rule=export_constraint)


    return model
    

def eva_objective(model, s):

    # Maximize the revenue from selling energy
    def objective_rule(model):
        return model.dt*sum( model.spot_price[t]*( (model.grid_export_P[s, t]) -  model.grid_import_P[s, t] )  - model.bat_cycle_cost*model.battery_discharge_P[s, t]
            for t in model.T )
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


def sbs_objective(model, s):

    # Maximize the revenue from selling energy
    def objective_rule(model):
        return model.dt*sum( model.spot_price[s, t]*( (model.grid_export_P[s, t]) -  model.grid_import_P[s, t] )  - model.bat_cycle_cost*model.battery_discharge_P[s, t]
            for t in model.T )
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

def asj_objective(model):

    # First-stage profit: counted once, because decisions are shared
    model.cost_fs = pyo.Expression(
        expr=model.dt * sum(
            model.spot_price[t] * (model.grid_export_P['fs', t] - model.grid_import_P['fs', t] )
            - model.bat_cycle_cost * model.battery_discharge_P['fs', t]
            for t in model.T_FS
        )
    )

    # Expected second-stage profit: only actual scenarios
    model.cost_ss = pyo.Expression(
        expr=sum(
            model.pi[s] * model.dt * sum(
                model.spot_price[t] * (model.grid_export_P[s, t] - model.grid_import_P[s, t] )
                - model.bat_cycle_cost * model.battery_discharge_P[s, t] 
                for t in model.T_SS
            )
            for s in model.S
        )
    )

    model.objective = pyo.Objective(
        expr=model.cost_fs + model.cost_ss,
        sense=pyo.maximize
    )




solver = SolverFactory('glpk')


price, pv_data = load_data()

pi = {'s1': 1/3, 's2': 1/3, 's3': 1/3}
scenarios = ['s1', 's2', 's3']


method = 'EVA'

if method == 'EVA':
    

    pv_data_eva = {
        ('EVA', t): sum(
            pi[s] * pv_data[s, t]
            for s in scenarios
        )
        for t in range(len(price))
}
    model = build_model(['EVA'], pv_data_eva, price)

    eva_objective(model, 'EVA')


    results = solver.solve(model, tee=False)

    print('---------------------------')
    print('Status:', results.solver.status)
    print('Termination:', results.solver.termination_condition)
    print(f'Objective value: {pyo.value(model.objective):.2f} kr')


elif method == 'SBS':
    pass
elif method == 'ASJ':
    pass