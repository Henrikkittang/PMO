


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

    return df_price['price_nok_kwh'], pv_data, power_dfs


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
        return model.dt*sum( model.spot_price[t]*( (model.grid_export_P[s, t]) -  model.grid_import_P[s, t] )  - model.bat_cycle_cost*model.battery_discharge_P[s, t]
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
            for s in ['s1', 's2', 's3']
        )
    )

    model.objective = pyo.Objective(
        expr=model.cost_fs + model.cost_ss,
        sense=pyo.maximize
    )




if __name__ == "__main__":


    solver = SolverFactory('glpk')


    price, pv_data, power_dfs = load_data()

    pi = {'s1': 1/3, 's2': 1/3, 's3': 1/3}
    scenarios = ['s1', 's2', 's3']


    method = 'ASJ'

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


        dt = pyo.value(model.dt)
        cycle_cost = pyo.value(model.bat_cycle_cost)
        P_line = pyo.value(model.P_line)

        # Extract the optimised EVA battery schedule
        P_charge = np.array([ pyo.value(model.battery_charge_P['EVA', t]) for t in model.T ])
        P_discharge = np.array([ pyo.value(model.battery_discharge_P['EVA', t]) for t in model.T ])
        P_import = np.array([ pyo.value(model.grid_import_P['EVA', t]) for t in model.T ])
        P_export = np.array([ pyo.value(model.grid_export_P['EVA', t]) for t in model.T ])
        price = np.array([ pyo.value(model.spot_price[t]) for t in model.T ])
        P_loss = np.array([pyo.value(model.loss_P['EVA', t])for t in model.T])


        scenario_profit = {}

        for s, df in power_dfs.items():
            pv = df["power_kw"].to_numpy()

            net = pv + P_discharge - P_charge

            net_export = np.minimum(np.maximum(net, 0), P_line)
            net_import = np.minimum(np.maximum(-net, 0), P_line)

            if np.any(-net > P_line + 1e-6):
                print(f"Warning: EVA battery schedule infeasible for scenario {s}: grid import limit exceeded.")

            # Check battery charging/discharging constraints separately if needed
            net_revenue_t = price * (net_export - net_import) - cycle_cost * P_discharge

            scenario_profit[s] = np.sum(net_revenue_t) * dt

        # Expected profit using the scenario probabilities
        expected_profit = sum(
            pi[s] * scenario_profit[s]
            for s in scenarios
        )

        print("Scenario profits:")
        for s, profit in scenario_profit.items():
            print(f"{s}: {profit:.2f} NOK")

        print(f"Expected realised EVA profit: {expected_profit:.2f} NOK")

    elif method == 'SBS':

        results = {}
        summary = []

        for s in scenarios:
            pv = power_dfs[s]['power_kw'].to_numpy()

            # Prepare PV data for this scenario
            pv_data_s = {
                ('SBS', t): float(pv[t])
                for t in range(len(price))
            }

            # Build and solve a separate model for each scenario
            model = build_model(['SBS'], pv_data_s, price)
            sbs_objective(model, 'SBS')

            res = solver.solve(model, tee=False)

            if res.solver.termination_condition != pyo.TerminationCondition.optimal:
                raise RuntimeError(
                    f'SBS model for scenario {s} did not solve optimally: '
                    f'{res.solver.termination_condition}'
                )

            revenue = pyo.value(model.objective)

            # Extract results
            results[s] = {
                'model': model,
                'objective': revenue,
                'battery_E': [ pyo.value(model.battery_E['SBS', t]) for t in model.T_battery ],
                'battery_charge_P': [ pyo.value(model.battery_charge_P['SBS', t]) for t in model.T ],
                'battery_discharge_P': [ pyo.value(model.battery_discharge_P['SBS', t]) for t in model.T ],
                'battery_P': [ pyo.value(model.battery_charge_P['SBS', t]) - pyo.value(model.battery_discharge_P['SBS', t]) for t in model.T ],
                'grid_P': [ pyo.value(model.grid_import_P['SBS', t]) - pyo.value(model.grid_export_P['SBS', t]) for t in model.T ],
                'pv_P': list(pv),
                'loss_P': [ pyo.value(model.loss_P['SBS', t]) for t in model.T ],
            }

            # Scenario energy and gross PV value
            dt = pyo.value(model.dt)

            pv_energy = sum(pv) * dt

            pv_value = sum(
                pv[t] * price[t] * dt
                for t in range(len(price))
            )

            summary.append({
                'Scenario': s,
                'Probability': pi[s],
                'Optimal revenue (NOK)': revenue,
                'PV energy (kWh)': pv_energy,
                'Gross PV value (NOK)': pv_value,
                'Weighted revenue (NOK)': pi[s] * revenue,
            })

        #######################
        # Scenario-by-scenario results
        #######################

        df_results = pd.DataFrame(summary)

        print('\nScenario-by-scenario results:')
        print(df_results.to_string(
            index=False,
            float_format=lambda x: f'{x:.2f}'
        ))

        R_WS = sum(
            pi[s] * results[s]['objective']
            for s in scenarios
        )

        print(f'\nExpected wait-and-see revenue: {R_WS:.2f} NOK')

    elif method == 'ASJ':


        results = {}
        summary = []

        n_steps = len(price)
        n_fs = n_steps // 2

        asj_scenarios = ['fs', 's1', 's2', 's3']

        # PV data for all scenarios.
        # The fs profile uses s3, which is the common first-stage profile.
        pv_data_asj = {
            ('fs', t): float(power_dfs['s3']['power_kw'].iloc[t])
            for t in range(n_steps)
        }

        pv_data_asj.update({
            (s, t): float(power_dfs[s]['power_kw'].iloc[t])
            for s in scenarios
            for t in range(n_steps)
        })

        # Build one joint model
        model = build_model(asj_scenarios, pv_data_asj, price)

        # Time stages
        model.T_FS = pyo.RangeSet(0, n_fs - 1)
        model.T_SS = pyo.RangeSet(n_fs, n_steps - 1)

        # Battery energy includes the stage boundary (n_fs)
        model.T_FS_E = pyo.RangeSet(0, n_fs)

        # Scenario probabilities
        model.pi = pyo.Param(
            model.S,
            initialize={
                'fs': 0.0,
                's1': pi['s1'],
                's2': pi['s2'],
                's3': pi['s3'],
            }
        )

        # First-stage decisions must be identical in all actual scenarios
        def nonanticipativity_rule(m, s, t):
            if s == 'fs':
                return pyo.Constraint.Skip

            return m.battery_charge_P[s, t] == m.battery_charge_P['fs', t]

        model.nonanticipativity_charge = pyo.Constraint(
            model.S, model.T_FS, rule=nonanticipativity_rule
        )

        def nonanticipativity_discharge_rule(m, s, t):
            if s == 'fs':
                return pyo.Constraint.Skip

            return m.battery_discharge_P[s, t] == m.battery_discharge_P['fs', t] 

        model.nonanticipativity_discharge = pyo.Constraint(
            model.S, model.T_FS, rule=nonanticipativity_discharge_rule
        )

        def nonanticipativity_import_rule(m, s, t):
            if s == 'fs':
                return pyo.Constraint.Skip

            return m.grid_import_P[s, t] == m.grid_import_P['fs', t] 

        model.nonanticipativity_import = pyo.Constraint(
            model.S, model.T_FS, rule=nonanticipativity_import_rule
        )

        def nonanticipativity_export_rule(m, s, t):
            if s == 'fs':
                return pyo.Constraint.Skip

            return m.grid_export_P[s, t] == m.grid_export_P['fs', t]
            

        model.nonanticipativity_export = pyo.Constraint(
            model.S, model.T_FS, rule=nonanticipativity_export_rule
        )

        def nonanticipativity_loss_rule(m, s, t):
            if s == 'fs':
                return pyo.Constraint.Skip

            return m.loss_P[s, t] == m.loss_P['fs', t]

        model.nonanticipativity_loss = pyo.Constraint(
            model.S, model.T_FS, rule=nonanticipativity_loss_rule
        )

        def nonanticipativity_charging_binary_rule(m, s, t):
            if s == 'fs':
                return pyo.Constraint.Skip

            return m.is_charging[s, t] == m.is_charging['fs', t]

        model.nonanticipativity_charging_binary = pyo.Constraint(
            model.S, model.T_FS, rule=nonanticipativity_charging_binary_rule
        )

        def nonanticipativity_import_binary_rule(m, s, t):
            if s == 'fs':
                return pyo.Constraint.Skip

            return m.is_importing[s, t] == m.is_importing['fs', t]

        model.nonanticipativity_import_binary = pyo.Constraint(
            model.S, model.T_FS, rule=nonanticipativity_import_binary_rule
        )

        # Battery energy must also agree through the stage boundary
        def nonanticipativity_energy_rule(m, s, t):
            if s == 'fs':
                return pyo.Constraint.Skip

            return m.battery_E[s, t] == m.battery_E['fs', t]

        model.nonanticipativity_energy = pyo.Constraint(model.S, model.T_FS_E, rule=nonanticipativity_energy_rule)

        # Objective and solve
        asj_objective(model)

        res = solver.solve(model, tee=False)

    

        revenue = pyo.value(model.objective)
        dt = pyo.value(model.dt)

        #######################
        # Extract results
        #######################

        for s in scenarios:
            pv = power_dfs[s]['power_kw'].to_numpy()

            scenario_revenue = dt * sum(
                pyo.value(model.spot_price[t]) * (
                    pyo.value(model.grid_export_P[s, t])
                    - pyo.value(model.grid_import_P[s, t])
                )
                - pyo.value(model.bat_cycle_cost)
                * pyo.value(model.battery_discharge_P[s, t])
                for t in model.T
            )

            results[s] = {
                'model': model,
                'objective': scenario_revenue,
                'battery_E': [
                    pyo.value(model.battery_E[s, t])
                    for t in model.T_battery
                ],
                'battery_charge_P': [
                    pyo.value(model.battery_charge_P[s, t])
                    for t in model.T
                ],
                'battery_discharge_P': [
                    pyo.value(model.battery_discharge_P[s, t])
                    for t in model.T
                ],
                'battery_P': [
                    pyo.value(model.battery_charge_P[s, t])
                    - pyo.value(model.battery_discharge_P[s, t])
                    for t in model.T
                ],
                'grid_P': [
                    pyo.value(model.grid_import_P[s, t])
                    - pyo.value(model.grid_export_P[s, t])
                    for t in model.T
                ],
                'pv_P': list(pv),
                'loss_P': [
                    pyo.value(model.loss_P[s, t])
                    for t in model.T
                ],
            }

            pv_energy = sum(pv) * dt
            pv_value = sum(
                pv[t] * price.iloc[t] * dt
                for t in range(n_steps)
            )

            summary.append({
                'Scenario': s,
                'Probability': pi[s],
                'Revenue under ASJ decisions (NOK)': scenario_revenue,
                'PV energy (kWh)': pv_energy,
                'Gross PV value (NOK)': pv_value,
                'Weighted revenue (NOK)': pi[s] * scenario_revenue,
            })

        df_results = pd.DataFrame(summary)

        print('\nScenario results under ASJ decisions:')
        print(df_results.to_string(
            index=False,
            float_format=lambda x: f'{x:.2f}'
        ))

        R_ASJ = sum(
            pi[s] * results[s]['objective']
            for s in scenarios
        )

        print(f'\nExpected ASJ revenue: {R_ASJ:.2f} NOK')
        print(f'ASJ objective value: {revenue:.2f} NOK')



