import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import matplotlib.pyplot as plt

#######################
# Python Variables
#######################

dt = 0.25

df_price = pd.read_csv('./data/price.csv')
df_power = pd.read_csv('./data/power.csv')

print(df_price.columns)
print(df_price['price_nok_kwh'])
print(df_power.columns)
print(df_power['power_kw'])


#######################
# Setup
#######################

model = pyo.ConcreteModel()

# Time Steps
model.T = pyo.RangeSet(0, 24*4-1) # 96 [15 min]

# Time steps minus last step
model.T_battery = pyo.RangeSet(model.T.first(), model.T.last() - 1)



#######################
# Decision Variables
#######################

# Battery energy level 
model.battery_E = pyo.Var(model.T, initialize=0, bounds=(0, 20), within=pyo.NonNegativeReals) # kWh
model.battery_E[0] = 10

# Power in/out of battery
model.battery_P = pyo.Var(model.T, initialize=0, bounds=(-10, 10)) # kW

# Binary variable for charging
model.is_charging = pyo.Var(model.T, initialize=0, domain=pyo.Binary) # 0|1

# Power in/out of grid
model.grid_P = pyo.Var(model.T, initialize=0) # kW


#######################
# Constant Parameteres
#######################


# Oower production of solar panels
model.spot_price    = pyo.Param(model.T, initialize=dict(enumerate(df_price['price_nok_kwh'])), within=pyo.NonNegativeReals)

# Price of electricy
model.PV_P_prod = pyo.Param(model.T, initialize=dict(enumerate(df_power['power_kw']))) # NOK/kWh

model.bat_cycle_cost = pyo.Param(initialize=0)    # NOK/kWh
model.bat_charge_eff = pyo.Param(initialize=0.95) # %




#######################
# Constraints
#######################

# Energy production cannot be created from nothing or come from nothing
def energy_balance_constraint(model, t):
    return (model.PV_P_prod[t] - model.battery_P[t] - model.grid_P[t]) == 0
model.energy_balance = pyo.Constraint(model.T, rule=energy_balance_constraint)


# Positive power (charging) only if is_charging = 1
def charge_constraint(model, t):
    return model.battery_P[t] <= model.battery_P[t].ub * model.is_charging[t]
model.charge_limit = pyo.Constraint(model.T, rule=charge_constraint)

# Negative power (discharging) only if is_charging = 0
def discharge_constraint(model, t):
    return model.battery_P[t] >= model.battery_P[t].lb * (1 - model.is_charging[t])
model.discharge_limit = pyo.Constraint(model.T, rule=discharge_constraint)


# Battery capacity cannot be discharged below 0 or over E_max
def battery_capacity_constraint(model, t):
    return model.battery_E[t + 1]  == model.battery_E[t] + model.battery_P[t] * dt
model.battery_energy_balance = pyo.Constraint(model.T_battery, rule=battery_capacity_constraint)




#######################
# Objective
#######################

def objective_rule(model):
    return sum( model.spot_price[t]*( model.grid_P[t] )*dt for t in model.T )
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)




#######################
# Results
#######################

solver = SolverFactory('glpk')

results = solver.solve(model, tee=True)

if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print('Optimal solution found!')
    print('Objective value:', pyo.value(model.objective))

    for t in model.T:
        print(
            f't={t:2d} | '
            f'Battery E={pyo.value(model.battery_E[t]):6.2f} kWh | '
            f'Battery P={pyo.value(model.battery_P[t]):6.2f} kW | '
            f'Grid P={pyo.value(model.grid_P[t]):6.2f} kW'
        )
else:
    print('Solver status:', results.solver.status)
    print('Termination:', results.solver.termination_condition)





#######################
# Plot
#######################

battery_E = [pyo.value(model.battery_E[t]) for t in model.T]
battery_P = [pyo.value(model.battery_P[t]) for t in model.T]
grid_P    = [pyo.value(model.grid_P[t]) for t in model.T]
pv_P      = [pyo.value(model.PV_P_prod[t]) for t in model.T]


plt.plot(battery_E, label='Battery E')
plt.plot(battery_P, label='Battery P')
plt.plot(grid_P, label='Grid P')
plt.plot(pv_P, label='PV P')


plt.legend()
plt.grid()
plt.show()