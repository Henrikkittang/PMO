import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import matplotlib.pyplot as plt

#######################
# Python Variables
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


# PV production, indexed by (scenario, time)
pv_data = {
    (s, t): float(power_dfs[s]['power_kw'].iloc[t])
    for s in power_dfs
    for t in range(n_steps)
}

pi = {'s1': 1/3, 's2': 1/3, 's3': 1/3}

senarios = ['s1', 's2', 's3']

#######################
# Setup
#######################

model = pyo.ConcreteModel()




#######################
# Constant Parameteres
#######################

# Time Steps
model.T = pyo.RangeSet(0, len(df_price['price_nok_kwh']) - 1 ) # 96 [15 min]

# Extra time step for energy integral
model.T_battery = pyo.RangeSet(0, len(df_price['price_nok_kwh']))

# Expected PV power at each time step [kW]
model.PV_P_EV = pyo.Param(model.T,
    initialize=lambda model, t: sum(
        pi[s] * pv_data[s, t]
        for s in senarios
    )
)

model.spot_price    = pyo.Param(model.T, initialize=dict(enumerate(df_price['price_nok_kwh'])))


model.P_line = pyo.Param(initialize=5)            # kW
model.P_max = pyo.Param(initialize=5)             # kW
model.battery_E_max = pyo.Param(initialize=20)    # kWh
model.battery_E_0 = pyo.Param(initialize=0)       # kWh

model.battery_eff =  pyo.Param(initialize=0.95)    # %
model.bat_cycle_cost = pyo.Param(initialize=0.35)  # NOK/kWh 
model.dt = pyo.Param(initialize=0.25)              # hours


#######################
# Decision Variables
#######################

# Battery energy level 
model.battery_E = pyo.Var(model.T_battery, initialize=model.battery_E_0, bounds=(0, model.battery_E_max), within=pyo.NonNegativeReals) # kWh

# Power in/out of battery
model.battery_charge_P    = pyo.Var(model.T, initialize=0, bounds=(0, model.P_max)) # kW
model.battery_discharge_P = pyo.Var(model.T, initialize=0, bounds=(0, model.P_max)) # kW

model.is_charging = pyo.Var(model.T, initialize=0, domain=pyo.Binary) # 0|1

# Power in/out of grid
model.grid_import_P = pyo.Var(model.T, initialize=0, bounds=(0, model.P_line)) # kW
model.grid_export_P = pyo.Var(model.T, initialize=0, bounds=(0, model.P_line)) # kW

model.loss_P = pyo.Var(model.T, initialize=0, bounds=(0, None)) # kW




#######################
# Constraints
#######################

model.initial_battery_constraint = pyo.Constraint(
    expr=model.battery_E[0] == model.battery_E_0
)

# Energy balance constraint
def energy_balance_constraint(model, t):
    return (model.PV_P_EV[t] + model.grid_import_P[t] + model.battery_discharge_P[t] - model.loss_P[t]) == (model.battery_charge_P[t] + model.grid_export_P[t])
model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_constraint)


# Charging/discharging constraint
def charge_constraint(model, t):
    return model.battery_charge_P[t] <= model.P_max * model.is_charging[t]
model.charge_limit = pyo.Constraint(model.T, rule=charge_constraint)

def discharge_constraint(model, t):
    return model.battery_discharge_P[t] <= model.P_max* (1 - model.is_charging[t])
model.discharge_limit = pyo.Constraint(model.T, rule=discharge_constraint)



# Battery capacity constraint
def battery_capacity_constraint(model, t):
    return model.battery_E[t + 1] == model.battery_E[t] + model.battery_eff*model.battery_charge_P[t]*model.dt - (1/model.battery_eff)*model.battery_discharge_P[t]*model.dt
model.battery_energy_balance = pyo.Constraint(model.T, rule=battery_capacity_constraint)


#######################
# Objective
#######################

# Maximize the revenue from selling energy
def objective_rule(model):
    return model.dt*sum( model.spot_price[t]*( (model.grid_export_P[t]) -  model.grid_import_P[t] - model.loss_P[t]) - 
                   model.bat_cycle_cost*(model.battery_charge_P[t] - model.battery_discharge_P[t] ) 
            for t in model.T )
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)




#######################
# Results
#######################

solver = SolverFactory('glpk')

results = solver.solve(model, tee=False)

print('---------------------------')
print('Status:', results.solver.status)
print('Termination:', results.solver.termination_condition)
print(f'Objective value: {pyo.value(model.objective):.2f} kr')


#######################
# Data
#######################

xs    = [t for t in model.T]
price = [pyo.value(model.spot_price[t]) for t in model.T]
loss = [pyo.value(model.loss_P[t]) for t in model.T]


battery_E = [pyo.value(model.battery_E[t]) for t in model.T]
battery_P = [pyo.value(model.battery_charge_P[t])- pyo.value(model.battery_discharge_P[t]) for t in model.T]
grid_P = [pyo.value(model.grid_import_P[t]) - pyo.value(model.grid_export_P[t]) for t in model.T ]
pv_P = [pyo.value(model.PV_P_EV[t]) for t in model.T]

pv_E = sum(pv_P)*model.dt
pv_price = sum( pv_P[t]*price[t]*model.dt for t in model.T )

print('---------------------------')
print('PV energy:', pv_E)
print('PV price:', pv_price)
print('Loss', loss)
print('---------------------------')


#######################
# Plot
#######################

fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)

plots = [
    (axes[0, 0], battery_E, "Battery Energy", "Energy [kWh]"),
    (axes[0, 1], battery_P, "Battery Power", "Power [kW]"),
    (axes[1, 0], grid_P, "Grid Power", "Power [kW]"),
    (axes[1, 1], pv_P, "PV Production", "Power [kW]"),
]

for ax, data, title, ylabel in plots:
    # Primary axis
    ax.step(xs, data, label=title, color="tab:blue", where="post")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # Secondary axis: electricity price
    ax2 = ax.twinx()
    ax2.step(xs, price, label="Spot Price", color="tab:orange", linestyle="--", where="post")
    ax2.set_ylabel("Price [NOK/kWh]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

axes[1, 0].set_xlabel("Time step [15 min]")
axes[1, 1].set_xlabel("Time step [15 min]")

fig.tight_layout()
plt.show()