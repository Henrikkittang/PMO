import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#######################
# Setup
#######################

df_price = pd.read_csv('./data/price.csv')
df_power_s1 = pd.read_csv('./data/power_s1.csv')
df_power_s2 = pd.read_csv('./data/power_s2.csv')
df_power_s3 = pd.read_csv('./data/power_s3.csv')

# Check that every series covers the same horizon
n_steps = len(df_price)

df_power_s1.loc[:47, 'power_kw'] = df_power_s3['power_kw'].iloc[:48]
df_power_s2.loc[:47, 'power_kw'] = df_power_s3['power_kw'].iloc[:48]


power_dfs = {'s1': df_power_s1, 's2': df_power_s2, 's3': df_power_s3}


# xs = range(len(df_power_s1))
# 
# fig, ax = plt.subplots()
# 
# # Original power
# ax.plot(xs, df_power_s1['power_kw'], label='Scenario 1', linewidth=2)
# ax.plot(xs, df_power_s2['power_kw'], label='Scenario 2', linewidth=2)
# ax.plot(xs, df_power_s3['power_kw'], label='Scenario 3', linewidth=2)
# ax.axvline(x=47, color='black', linestyle='--', linewidth=1.5)
# ax.plot(xs[:48], df_power_s1['power_kw'][:48], label='Shared production', linewidth=2, color='black')
# 
# plt.xlim(xs[0], xs[-1])
# plt.xlabel('Timesteps [15 min]')
# plt.ylabel('Power (kW)')
# plt.title('PV production scenarios')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# 
# quit()
 
model = pyo.ConcreteModel()


#######################
# Parameters
#######################

model.P_max = pyo.Param(initialize=5)             # kW
model.P_line_max = pyo.Param(initialize=5.375)             # kW
model.battery_E_max = pyo.Param(initialize=20)    # kWh
model.battery_E_0 = pyo.Param(initialize=0)       # kWh

model.battery_eff =  pyo.Param(initialize=0.95)
model.bat_cycle_cost = pyo.Param(initialize=0.33)  # NOK/kWh throughput
model.dt = pyo.Param(initialize=0.25)             # hours

# Scenarios and probabilities
model.S = pyo.Set(initialize=['s1', 's2', 's3'])
model.pi = pyo.Param(model.S, initialize={'s1': 1/3, 's2': 1/3, 's3': 1/3})

# Time intervals and battery states
model.T = pyo.RangeSet(0, n_steps-1)

model.T_FS = pyo.RangeSet(0, n_steps//2-1)
model.T_SS = pyo.RangeSet(n_steps//2, n_steps-1)

model.T_battery_FS = pyo.RangeSet(0, n_steps)
model.T_battery_SS = pyo.RangeSet(n_steps//2, n_steps)

# Electricity prices
model.spot_price = pyo.Param(model.T, initialize=dict(enumerate(df_price['price_nok_kwh'])))

# PV production, indexed by (scenario, time)
pv_data = {
    (s, t): float(power_dfs[s]['power_kw'].iloc[t])
    for s in power_dfs
    for t in range(n_steps)
}

model.PV_P_prod = pyo.Param(model.S, model.T, initialize=pv_data)


#######################
# Expected Value Approximation
#######################


model.P_PV_EV  = pyo.Param(model.T, initialize=lambda model, t: sum(model.pi[s] * model.PV_P_prod[s, t] for s in model.S))
model.E_PV_EV  = pyo.Param(initialize=sum(model.pi[s] * model.PV_P_prod[s, t] * model.dt for t in model.T for s in model.S))
model.price_EV = pyo.Param(initialize=sum(model.pi[s] * model.spot_price[t] * model.PV_P_prod[s, t] * model.dt for t in model.T for s in model.S))

print('---------------------------')
print(f'Exptected Energy PV {pyo.value(model.E_PV_EV):.2f} kWh')
print(f'Exptected Price {pyo.value(model.price_EV):.2f} kr')
print('---------------------------')


#######################
# Decision Variables
#######################

#  ---- First stage ---- Defined over first half of T


# Battery energy level 
model.battery_E_FS = pyo.Var(model.T_battery_FS, initialize=0, bounds=(0, model.battery_E_max), domain=pyo.Reals) # kWh

# Power in/out of battery
model.battery_charge_P_FS    = pyo.Var(model.T_FS, initialize=0, bounds=(0, model.P_max), domain=pyo.Reals) # kW
model.battery_discharge_P_FS = pyo.Var(model.T_FS, initialize=0, bounds=(0, model.P_max), domain=pyo.Reals) # kW

model.is_charging_FS = pyo.Var(model.T_FS, initialize=0, domain=pyo.Binary) # 0|1

# Power in/out of grid
model.grid_import_P_FS = pyo.Var(model.T_FS, initialize=0, bounds=(0, None), domain=pyo.Reals) # kW
model.grid_export_P_FS = pyo.Var(model.T_FS, initialize=0, bounds=(0, None), domain=pyo.Reals) # kW
model.is_importing_FS  = pyo.Var(model.T_FS, initialize=0, domain=pyo.Binary) # 0|1


# ---- Second stage  ---- Defined over second half of T and S

# Battery energy level 
model.battery_E_SS = pyo.Var(model.S, model.T_battery_SS, initialize=0, bounds=(0, model.battery_E_max), domain=pyo.Reals) # kWh

# Power in/out of battery
model.battery_charge_P_SS    = pyo.Var(model.S, model.T_SS, initialize=0, bounds=(0, model.P_max), domain=pyo.Reals) # kW
model.battery_discharge_P_SS = pyo.Var(model.S, model.T_SS, initialize=0, bounds=(0, model.P_max), domain=pyo.Reals) # kW
model.is_charging_SS = pyo.Var(model.S, model.T_SS, initialize=0, domain=pyo.Binary) # 0|1

# Power in/out of grid
model.grid_import_P_SS = pyo.Var(model.S, model.T_SS, initialize=0, bounds=(0, None), domain=pyo.Reals) # kW
model.grid_export_P_SS = pyo.Var(model.S, model.T_SS, initialize=0, bounds=(0, None), domain=pyo.Reals) # kW
model.is_importing_SS  = pyo.Var(model.S, model.T_SS, initialize=0, domain=pyo.Binary) # 0|1




#######################
# Constraints
#######################

#  ---- First stage ----

model.initial_battery_constraint = pyo.Constraint(
    expr=model.battery_E_FS[0] == model.battery_E_0
)

# Energy balance constraint
def energy_balance_constraint_fs(model, t):
    return (model.PV_P_prod['s1', t] + model.grid_import_P_FS[t] + model.battery_discharge_P_FS[t]) == (model.battery_charge_P_FS[t] + model.grid_export_P_FS[t])
model.energy_balance_fs = pyo.Constraint(model.T_FS, rule=energy_balance_constraint_fs)


# Charging/discharging constraint
def charge_constraint_fs(model, t):
    return model.battery_charge_P_FS[t] <= model.P_max * model.is_charging_FS[t]
model.charge_limit_fs = pyo.Constraint(model.T_FS, rule=charge_constraint_fs)

def discharge_constraint_fs(model, t):
    return model.battery_discharge_P_FS[t] <= model.P_max* (1 - model.is_charging_FS[t])
model.discharge_limit_fs = pyo.Constraint(model.T_FS, rule=discharge_constraint_fs)


# Battery capacity constraint
def battery_capacity_constraint_fs(model, t):
    return model.battery_E_FS[t + 1] == model.battery_E_FS[t] + model.battery_eff*model.battery_charge_P_FS[t]*model.dt - (1/model.battery_eff)*model.battery_discharge_P_FS[t]*model.dt
model.battery_energy_balance_fs = pyo.Constraint(model.T_FS, rule=battery_capacity_constraint_fs)


# Import/export constraint
def import_constraint_fs(model, t):
    return model.grid_import_P_FS[t] <= model.P_line_max* model.is_importing_FS[t]
model.import_limit_fs = pyo.Constraint(model.T_FS, rule=import_constraint_fs)

def export_constraint_fs(model, t):
    return model.grid_export_P_FS[t] <= model.P_line_max* model.is_importing_FS[t]
model.export_limit_fs = pyo.Constraint(model.T_FS, rule=export_constraint_fs)




#  ---- Second stage ----

# Energy balance constraint
def energy_balance_constraint_ss(model, s, t):
    return (model.PV_P_prod[s, t] + model.grid_import_P_SS[s, t] + model.battery_discharge_P_SS[s, t]) == (model.battery_charge_P_SS[s, t] + model.grid_export_P_SS[s, t])
model.energy_balance_ss = pyo.Constraint(model.S, model.T_SS, rule=energy_balance_constraint_ss)


# Charging/discharging constraint
def charge_constraint_ss(model, s, t):
    return model.battery_charge_P_SS[s, t] <= model.P_max * model.is_charging_SS[s, t]
model.charge_limit_ss = pyo.Constraint(model.S, model.T_SS, rule=charge_constraint_ss)

def discharge_constraint_ss(model, s, t):
    return model.battery_discharge_P_SS[s, t] <= model.P_max* (1 - model.is_charging_SS[s, t])
model.discharge_limit_ss = pyo.Constraint(model.S, model.T_SS, rule=discharge_constraint_ss)


# Battery capacity constraint
def battery_capacity_constraint_ss(model, s, t):
    return model.battery_E_SS[s, t + 1] == model.battery_E_SS[s, t] + model.battery_eff*model.battery_charge_P_SS[s, t]*model.dt - (1/model.battery_eff)*model.battery_discharge_P_SS[s, t]*model.dt
model.battery_energy_balance_ss = pyo.Constraint(model.S, model.T_SS, rule=battery_capacity_constraint_ss)



# Import/export constraint
def import_constraint_ss(model, s, t):
    return model.grid_import_P_SS[s, t] <= model.P_line_max * model.is_importing_SS[s, t]
model.import_limit_ss = pyo.Constraint(model.S, model.T_SS, rule=import_constraint_ss)

def export_constraint_ss(model, s, t):
    return model.grid_export_P_SS[s, t] <= model.P_line_max* (1 - model.is_importing_SS[s, t])
model.export_limit_ss = pyo.Constraint(model.S, model.T_SS, rule=export_constraint_ss)


#  ---- Link between stages ----

# Battery capacity constraint
def battery_energy_link_constraint(model, s):
    return model.battery_E_SS[s, n_steps // 2] == model.battery_E_FS[n_steps // 2]
model.battery_energy_link = pyo.Constraint(model.S, rule=battery_energy_link_constraint)


#######################
# Objective
#######################
model.FirstStageCost = pyo.Expression(
    expr=model.dt * sum(
        model.spot_price[t] * (model.grid_export_P_FS[t] - model.grid_import_P_FS[t])
        - model.bat_cycle_cost * model.battery_charge_P_FS[t]
        for t in model.T_FS
    )
)

model.SecondStageCost = pyo.Expression(
    expr=sum(
        model.pi[s] * model.dt * sum(
            model.spot_price[t] * (model.grid_export_P_SS[s, t] - model.grid_import_P_SS[s, t])
            - model.bat_cycle_cost * model.battery_charge_P_SS[s, t]
            for t in model.T_SS
        )
        for s in model.S
    )
)

model.objective = pyo.Objective(
    expr=model.FirstStageCost + model.SecondStageCost,
    sense=pyo.maximize
)

#######################
# Results
#######################

solver = SolverFactory('glpk')
results = solver.solve(model, tee=False)


print('---------------------------')
print("Status:", results.solver.status)
print("Termination:", results.solver.termination_condition)
print('---------------------------')

# if results.solver.termination_condition == pyo.TerminationCondition.optimal:
#     print('Optimal solution found!')
#     print('Objective value:', pyo.value(model.objective))
# 
#     for t in model.T:
#         print(
#             f't={t:2d} | '
#             f'Battery E={pyo.value(model.battery_E[t]):6.2f} kWh | '
#             f'Battery P={pyo.value(model.battery_charge_P[t]) + pyo.value(model.battery_discharge_P[t]):6.2f} kW | '
#             f'Grid P={pyo.value(model.grid_import_P[t]) + pyo.value(model.grid_export_P[t]):6.2f} kW'
#         )
# else:
#     print('Solver status:', results.solver.status)
#     print('Termination:', results.solver.termination_condition)

print('---------------------------')
print(f'First stage value: {pyo.value(model.FirstStageCost):.2f} kr')
print(f'Second stage value: {pyo.value(model.SecondStageCost):.2f} kr')
print(f'Objective value: {pyo.value(model.objective):.2f} kr')
print('---------------------------')




#######################
# Data
#######################

scenario = 's2'
half = n_steps // 2

# Time indices
xs = list(range(n_steps))
price = [pyo.value(model.spot_price[t]) for t in model.T]

# Battery energy: states are indexed 0..n_steps
battery_E = (
    [pyo.value(model.battery_E_FS[t]) for t in range(half)]
    + [pyo.value(model.battery_E_SS[scenario, t]) for t in range(half, n_steps)]
)

# Battery power: positive = charging, negative = discharging
battery_P = (
    [
        pyo.value(model.battery_charge_P_FS[t])
        - pyo.value(model.battery_discharge_P_FS[t])
        for t in model.T_FS
    ]
    + [
        pyo.value(model.battery_charge_P_SS[scenario, t])
        - pyo.value(model.battery_discharge_P_SS[scenario, t])
        for t in model.T_SS
    ]
)

# Grid power: positive = importing, negative = exporting
grid_P = (
    [
        pyo.value(model.grid_import_P_FS[t])
        - pyo.value(model.grid_export_P_FS[t])
        for t in model.T_FS
    ]
    + [
        pyo.value(model.grid_import_P_SS[scenario, t])
        - pyo.value(model.grid_export_P_SS[scenario, t])
        for t in model.T_SS
    ]
)

# PV production: first-stage values are common decisions, but PV itself
# is scenario-dependent. Use scenario-specific PV throughout.
pv_P = [ pyo.value(model.PV_P_prod[scenario, t]) for t in model.T ]

# Energy and gross PV value for the selected scenario
dt = pyo.value(model.dt)

pv_E = sum(pv_P) * dt
pv_price = sum(
    pv_P[t] * price[t] * dt
    for t in range(n_steps)
)

print('---------------------------')
print(f'Scenario: {scenario}')
print(f'PV energy: {pv_E:.2f} kWh')
print(f'PV value at spot price: {pv_price:.2f} NOK')
print('---------------------------')


#######################
# Plot
#######################

fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)

plots = [
    (axes[0, 0], battery_E, 'Battery Energy', 'Energy [kWh]'),
    (axes[0, 1], battery_P, 'Battery Power', 'Power [kW]'),
    (axes[1, 0], grid_P, 'Grid Power', 'Power [kW]'),
    (axes[1, 1], pv_P, 'PV Production', 'Power [kW]'),
]
fig.suptitle("Scenatio: " + scenario, fontsize=16)

for ax, data, title, ylabel in plots:
    ax.step(xs, data, label=title, color='tab:blue', where='post')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    ax.axvline(x=n_steps//2, color='red', linestyle=':', label='Stage transition')
    ax.axhline(y=pyo.value(model.P_max), color='black', linestyle='--', linewidth=0.8, label='Power limit')
    ax.axhline(y=pyo.value(-model.P_max), color='black', linestyle='--', linewidth=0.8)


    ax2 = ax.twinx()
    ax2.step(
        xs, price,
        label='Spot Price',
        color='tab:orange',
        linestyle='--',
        where='post'
    )
    ax2.set_ylabel('Price [NOK/kWh]', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

axes[1, 0].set_xlabel('Time step [15 min]')
axes[1, 1].set_xlabel('Time step [15 min]')

fig.tight_layout()
plt.show()