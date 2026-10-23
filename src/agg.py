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
# # Original power
# plt.plot(xs, df_power_s1['power_kw'], label='S1', linewidth=2)
# plt.plot(xs, df_power_s2['power_kw'], label='S2', linewidth=2)
# plt.plot(xs, df_power_s3['power_kw'], label='S3', linewidth=2)
# 
# plt.xlabel('Timestep')
# plt.ylabel('Power (kW)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

 
model = pyo.ConcreteModel()


#######################
# Parameters
#######################

model.P_max = pyo.Param(initialize=5)             # kW
model.battery_E_max = pyo.Param(initialize=20)    # kWh
model.battery_E_0 = pyo.Param(initialize=0)       # kWh

model.battery_eff =  pyo.Param(initialize=0.95)
model.bat_cycle_cost = pyo.Param(initialize=0.65)  # NOK/kWh throughput
model.dt = pyo.Param(initialize=0.25)             # hours

# Scenarios and probabilities
model.S = pyo.Set(initialize=['fs', 's1', 's2', 's3'])
model.pi = pyo.Param(model.S, initialize={'fs': 1, 's1': 1/3, 's2': 1/3, 's3': 1/3})

# Time intervals and battery states
model.T = pyo.RangeSet(0, n_steps-1)
model.T_battery = pyo.RangeSet(0, n_steps)

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


# Add the first-stage reference production
pv_data.update({
    ('fs', t): float(df_power_s1['power_kw'].iloc[t])
    for t in range(n_steps)
})

model.PV_P_prod = pyo.Param(model.S, model.T, initialize=pv_data)


#######################
# Expected Value Approximation
#######################

scenarios = ['s1', 's2', 's3']

# Expected PV power at each time step [kW]
model.P_PV_EV = pyo.Param(
    model.T,
    initialize=lambda m, t: sum(
        model.pi[s] * model.PV_P_prod[s, t]
        for s in scenarios
    )
)

# Expected total PV energy [kWh]
model.E_PV_EV = pyo.Param(
    initialize=sum(
        model.pi[s] * model.PV_P_prod[s, t] * model.dt
        for s in scenarios
        for t in model.T
    )
)

# Expected gross PV value at spot prices [NOK]
model.price_EV = pyo.Param(
    initialize=sum(
        model.pi[s] * model.spot_price[t]
        * model.PV_P_prod[s, t] * model.dt
        for s in scenarios
        for t in model.T
    )
)

print('---------------------------')
print(f'Expected PV energy: {pyo.value(model.E_PV_EV):.2f} kWh')
print(f'Expected PV value: {pyo.value(model.price_EV):.2f} NOK')
print('---------------------------')


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
model.grid_import_P = pyo.Var(model.S, model.T, initialize=0, bounds=(0, model.P_max), domain=pyo.NonNegativeReals) # kW
model.grid_export_P = pyo.Var(model.S, model.T, initialize=0, bounds=(0, None),        domain=pyo.NonNegativeReals) # kW




#######################
# Constraints
#######################


# Energy balance constraint
def energy_balance_constraint(model, s, t):
    return (model.PV_P_prod[s, t] + model.grid_import_P[s, t] + model.battery_discharge_P[s, t]) == (model.battery_charge_P[s, t] + model.grid_export_P[s, t])


# Charging/discharging constraint
def charge_constraint(model, s, t):
    return model.battery_charge_P[s, t] <= model.P_max * model.is_charging[s, t]

def discharge_constraint(model, s, t):
    return model.battery_discharge_P[s, t] <= model.P_max* (1 - model.is_charging[s, t])


# Battery capacity constraint
def battery_capacity_constraint(model, s, t):
    return model.battery_E[s, t + 1] == model.battery_E[s, t] + model.battery_eff*model.battery_charge_P[s, t]*model.dt - (1/model.battery_eff)*model.battery_discharge_P[s, t]*model.dt




#  ---- First stage ----

model.initial_battery_constraint = pyo.Constraint(
    expr=model.battery_E['fs', 0] == model.battery_E_0
)


model.energy_balance_fs         = pyo.Constraint(['fs'], model.T_FS, rule=energy_balance_constraint)
model.charge_limit_fs           = pyo.Constraint(['fs'], model.T_FS, rule=charge_constraint)
model.discharge_limit_fs        = pyo.Constraint(['fs'], model.T_FS, rule=discharge_constraint)
model.battery_energy_balance_fs = pyo.Constraint(['fs'], model.T_FS, rule=battery_capacity_constraint)


#  ---- Second stage ----

model.energy_balance_ss         = pyo.Constraint(model.S, model.T_SS, rule=energy_balance_constraint)
model.charge_limit_ss           = pyo.Constraint(model.S, model.T_SS, rule=charge_constraint)
model.discharge_limit_ss        = pyo.Constraint(model.S, model.T_SS, rule=discharge_constraint)
model.battery_energy_balance_ss = pyo.Constraint(model.S, model.T_SS, rule=battery_capacity_constraint)


#  ---- Link between stages ----

# Battery capacity constraint
def battery_energy_link_constraint(model, s):
    return model.battery_E[s, n_steps // 2] == model.battery_E['fs', n_steps // 2]
model.battery_energy_link = pyo.Constraint(['s1', 's2', 's3'], rule=battery_energy_link_constraint)


# First-stage decisions constraint
def first_stage_charge_link(model, s, t):
    return model.battery_charge_P[s, t] == model.battery_charge_P['fs', t]

def first_stage_discharge_link(model, s, t):
    return model.battery_discharge_P[s, t] == model.battery_discharge_P['fs', t]

def first_stage_charging_link(model, s, t):
    return model.is_charging[s, t] == model.is_charging['fs', t]

def first_stage_import_link(model, s, t):
    return model.grid_import_P[s, t] == model.grid_import_P['fs', t]

def first_stage_export_link(model, s, t):
    return model.grid_export_P[s, t] == model.grid_export_P['fs', t]

model.first_stage_charge_link    = pyo.Constraint(['s1', 's2', 's3'], model.T_FS, rule=first_stage_charge_link)
model.first_stage_discharge_link = pyo.Constraint(['s1', 's2', 's3'], model.T_FS, rule=first_stage_discharge_link)
model.first_stage_charging_link  = pyo.Constraint(['s1', 's2', 's3'], model.T_FS, rule=first_stage_charging_link)
model.first_stage_import_link    = pyo.Constraint(['s1', 's2', 's3'], model.T_FS, rule=first_stage_import_link)
model.first_stage_export_link    = pyo.Constraint(['s1', 's2', 's3'], model.T_FS, rule=first_stage_export_link)





#######################
# Objective
#######################

# First-stage profit: counted once, because decisions are shared
model.cost_fs = pyo.Expression(
    expr=model.dt * sum(
        model.spot_price[t] * model.grid_export_P['fs', t]
        - 0.5 * model.grid_import_P['fs', t]
        - model.bat_cycle_cost * model.battery_discharge_P['fs', t]
        for t in model.T_FS
    )
)

# Expected second-stage profit: only actual scenarios
model.cost_ss = pyo.Expression(
    expr=sum(
        model.pi[s] * model.dt * sum(
            model.spot_price[t] * model.grid_export_P[s, t]
            - 0.5 * model.grid_import_P[s, t]
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

#######################
# Results
#######################

solver = SolverFactory('glpk')

results = solver.solve(model, tee=False)

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
print(f'Objective value: {pyo.value(model.objective):.2f} kr')
print('---------------------------')



#######################
# Data
#######################

scenario = 's3'
half = n_steps // 2
dt = pyo.value(model.dt)

# Time indices
xs = list(range(n_steps))
price = [pyo.value(model.spot_price[t]) for t in model.T]

# Battery energy: states are indexed 0..n_steps
# Use the 'fs' trajectory for the first stage and the
# selected scenario for the second stage.
battery_E = (
    [pyo.value(model.battery_E['fs', t]) for t in range(half)]
    + [pyo.value(model.battery_E[scenario, t]) for t in range(half, n_steps)]
)

# Include the final battery state at n_steps
battery_E.append(pyo.value(model.battery_E[scenario, n_steps]))

# Battery power: positive = charging, negative = discharging
battery_P = (
    [
        pyo.value(model.battery_charge_P['fs', t])
        - pyo.value(model.battery_discharge_P['fs', t])
        for t in range(half)
    ]
    + [
        pyo.value(model.battery_charge_P[scenario, t])
        - pyo.value(model.battery_discharge_P[scenario, t])
        for t in range(half, n_steps)
    ]
)

print(list(pyo.value(model.grid_import_P['fs', t]) for t in range(half)))
print(list(pyo.value(model.grid_export_P['fs', t]) for t in range(half)))
print('----------------------')
print(list(pyo.value(model.grid_import_P['s1', t]) for t in range(half, n_steps)))
print(list(pyo.value(model.grid_export_P['s1', t]) for t in range(half, n_steps)))

# Grid power: positive = importing, negative = exporting
grid_P = (
    [
        pyo.value(model.grid_import_P['fs', t])
        - pyo.value(model.grid_export_P['fs', t])
        for t in range(half)
    ]
    + [
        pyo.value(model.grid_import_P[scenario, t])
        - pyo.value(model.grid_export_P[scenario, t])
        for t in range(half, n_steps)
    ]
)

# PV production: use the selected actual scenario throughout
pv_P = [
    pyo.value(model.PV_P_prod[scenario, t])
    for t in model.T
]

# Energy and gross PV value for the selected scenario
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

for i, (ax, data, title, ylabel) in enumerate(plots):
    # Energy has n_steps + 1 points; power has n_steps
    x_plot = list(range(len(data)))

    ax.step(x_plot, data, label=title, color='tab:blue', where='post')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    ax.axvline(
        x=half, color='red', linestyle=':',
        label='Stage transition'
    )

    if title != 'Battery Energy':
        ax.axhline(
            y=pyo.value(model.P_max), color='black',
            linestyle='--', linewidth=0.8, label='Power limit'
        )
        ax.axhline(
            y=-pyo.value(model.P_max), color='black',
            linestyle='--', linewidth=0.8
        )

    # Spot price on the secondary axis
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
