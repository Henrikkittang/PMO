import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import matplotlib.pyplot as plt

#######################
# Python Variables
#######################


P_max            = 5    # kW
battery_E_max    = 20   # kWh
battery_E_0      = 0    # kWh
battery_eff      = 0.95 # %
battery_cyc_cost = 1.5  # NOK/kWh

dt = 0.25


df_price = pd.read_csv('./data/price.csv')
df_power = pd.read_csv('./data/power.csv')

#######################
# Setup
#######################

model = pyo.ConcreteModel()

# Time Steps
model.T = pyo.RangeSet(0, len(df_price['price_nok_kwh']) - 1 ) # 96 [15 min]

# Extra time step for energy integral
model.T_battery = pyo.RangeSet(0, len(df_price['price_nok_kwh']))



#######################
# Decision Variables
#######################

# Battery energy level 
model.battery_E = pyo.Var(model.T_battery, initialize=battery_E_0, bounds=(0, battery_E_max), within=pyo.NonNegativeReals) # kWh

# Power in/out of battery
model.battery_charge_P    = pyo.Var(model.T, initialize=0, bounds=(0, P_max)) # kW
model.battery_discharge_P = pyo.Var(model.T, initialize=0, bounds=(0, P_max)) # kW

model.is_charging = pyo.Var(model.T, initialize=0, domain=pyo.Binary) # 0|1

# Power in/out of grid
model.grid_import_P = pyo.Var(model.T, initialize=0, bounds=(0, P_max)) # kW
model.grid_export_P = pyo.Var(model.T, initialize=0, bounds=(0, None)) # kW




#######################
# Constant Parameteres
#######################

# Price of electricy
model.spot_price    = pyo.Param(model.T, initialize=dict(enumerate(df_price['price_nok_kwh'])))

# Oower production of solar panels
model.PV_P_prod = pyo.Param(model.T, initialize=dict(enumerate(df_power['power_kw'])), within=pyo.NonNegativeReals) # NOK/kWh

model.bat_cycle_cost = pyo.Param(initialize=0)    # NOK/kWh
model.bat_charge_eff = pyo.Param(initialize=0.95) # %




#######################
# Constraints
#######################

model.initial_battery_constraint = pyo.Constraint(
    expr=model.battery_E[0] == battery_E_0
)

# Energy balance constraint
def energy_balance_constraint(model, t):
    return (model.PV_P_prod[t] + model.grid_import_P[t] + model.battery_discharge_P[t]) == (model.battery_charge_P[t] + model.grid_export_P[t])
model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_constraint)


# Charging/discharging constraint
def charge_constraint(model, t):
    return model.battery_charge_P[t] <= P_max * model.is_charging[t]
model.charge_limit = pyo.Constraint(model.T, rule=charge_constraint)

def discharge_constraint(model, t):
    return model.battery_discharge_P[t] <= P_max* (1 - model.is_charging[t])
model.discharge_limit = pyo.Constraint(model.T, rule=discharge_constraint)



# Battery capacity constraint
def battery_capacity_constraint(model, t):
    return model.battery_E[t + 1] == model.battery_E[t] + battery_eff*model.battery_charge_P[t]*dt - (1/battery_eff)*model.battery_discharge_P[t]*dt
model.battery_energy_balance = pyo.Constraint(model.T, rule=battery_capacity_constraint)


#######################
# Objective
#######################

# Maximize the revenue from selling energy
def objective_rule(model):
    return dt*sum( model.spot_price[t]*( (model.grid_export_P[t])) - model.grid_import_P[t]*0.5 - battery_cyc_cost*(model.battery_charge_P[t] - model.battery_discharge_P[t]) for t in model.T )
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)




#######################
# Results
#######################

solver = SolverFactory('glpk')

results = solver.solve(model, tee=True)

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
print('Objective value:', pyo.value(model.objective))
print('---------------------------')


#######################
# Data
#######################

xs    = [t for t in model.T]
price = [pyo.value(model.spot_price[t]) for t in model.T]

battery_E = [pyo.value(model.battery_E[t]) for t in model.T]
battery_P = [pyo.value(model.battery_charge_P[t])- pyo.value(model.battery_discharge_P[t]) for t in model.T]
grid_P = [pyo.value(model.grid_import_P[t]) - pyo.value(model.grid_export_P[t]) for t in model.T ]
pv_P = [pyo.value(model.PV_P_prod[t]) for t in model.T]

pv_E = sum(pv_P)*dt
pv_price = sum( pv_P[t]*price[t]*dt for t in model.T )

print('---------------------------')
print('PV energy:', pv_E)
print('PV price:', pv_price)
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