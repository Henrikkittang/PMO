
import pyomo.environ as pyo
import pandas as pd
import matplotlib.pyplot as plt

#######################
# Data
#######################

df_price = pd.read_csv('./data/price.csv')
power_dfs = {
    's1': pd.read_csv('./data/power_s1.csv'),
    's2': pd.read_csv('./data/power_s2.csv'),
    's3': pd.read_csv('./data/power_s3.csv'),
}

n_steps = len(df_price)
dt = 0.25
scenarios = ['s1', 's2', 's3']
pi = {'s1': 1/3, 's2': 1/3, 's3': 1/3}

# Use the same first-stage PV profile in all scenarios.
for s in ['s1', 's2']:
    power_dfs[s].loc[:47, 'power_kw'] = (
        power_dfs['s3']['power_kw'].iloc[:48].to_numpy()
    )

# Check that all input series cover the same horizon.
assert all(len(df) == n_steps for df in power_dfs.values())

prices = df_price['price_nok_kwh'].to_numpy()

#######################
# Model builder
#######################

def build_model(pv_values):
    m = pyo.ConcreteModel()

    # Sets
    m.T = pyo.RangeSet(0, n_steps - 1)
    m.T_battery = pyo.RangeSet(0, n_steps)

    # Parameters
    m.P_line = pyo.Param(initialize=5)
    m.P_max = pyo.Param(initialize=5)
    m.battery_E_max = pyo.Param(initialize=20)
    m.battery_E_0 = pyo.Param(initialize=0)
    m.battery_eff = pyo.Param(initialize=0.95)
    m.bat_cycle_cost = pyo.Param(initialize=0.35)
    m.dt = pyo.Param(initialize=dt)

    m.spot_price = pyo.Param(
        m.T,
        initialize={t: float(prices[t]) for t in range(n_steps)}
    )

    m.PV_P_prod = pyo.Param(
        m.T,
        initialize={t: float(pv_values[t]) for t in range(n_steps)}
    )

    # Decision variables
    m.battery_E = pyo.Var(
        m.T_battery, domain=pyo.NonNegativeReals,
        bounds=(0, 20)
    )
    m.battery_charge_P = pyo.Var(
        m.T, domain=pyo.NonNegativeReals, bounds=(0, 5)
    )
    m.battery_discharge_P = pyo.Var(
        m.T, domain=pyo.NonNegativeReals, bounds=(0, 5)
    )
    m.is_charging = pyo.Var(m.T, domain=pyo.Binary)

    m.grid_import_P = pyo.Var(
        m.T, domain=pyo.NonNegativeReals, bounds=(0, 5)
    )
    m.grid_export_P = pyo.Var(
        m.T, domain=pyo.NonNegativeReals, bounds=(0, 5)
    )
    m.loss_P = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # Initial battery energy
    m.initial_battery_constraint = pyo.Constraint(
        expr=m.battery_E[0] == m.battery_E_0
    )

    # Power balance
    def energy_balance(m, t):
        return (
            m.PV_P_prod[t]
            + m.grid_import_P[t]
            + m.battery_discharge_P[t]
            == m.battery_charge_P[t]
            + m.grid_export_P[t]
            + m.loss_P[t]
        )

    m.energy_balance = pyo.Constraint(m.T, rule=energy_balance)

    # No simultaneous charging and discharging
    def charge_limit(m, t):
        return m.battery_charge_P[t] <= (
            m.P_max * m.is_charging[t]
        )

    def discharge_limit(m, t):
        return m.battery_discharge_P[t] <= (
            m.P_max * (1 - m.is_charging[t])
        )

    m.charge_limit = pyo.Constraint(m.T, rule=charge_limit)
    m.discharge_limit = pyo.Constraint(m.T, rule=discharge_limit)

    # Battery energy dynamics
    def battery_dynamics(m, t):
        return (
            m.battery_E[t + 1]
            == m.battery_E[t]
            + m.battery_eff * m.battery_charge_P[t] * m.dt
            - (1 / m.battery_eff)
            * m.battery_discharge_P[t] * m.dt
        )

    m.battery_energy_balance = pyo.Constraint(
        m.T, rule=battery_dynamics
    )

    # Objective: market revenue minus battery throughput cost.
    # This charges the cycling cost on both charging and discharging.
    m.objective = pyo.Objective(
        expr=m.dt * sum(
            m.spot_price[t] * (m.grid_export_P[t] - m.grid_import_P[t])
            - m.bat_cycle_cost * m.battery_discharge_P[t]
    
            for t in m.T
        ),
        sense=pyo.maximize
    )

    return m


#######################
# Solve each scenario independently
#######################

solver = pyo.SolverFactory('glpk')
results = {}
summary = []

for s in scenarios:
    pv = power_dfs[s]['power_kw'].to_numpy()

    # A separate model for each scenario
    m = build_model(pv)
    res = solver.solve(m, tee=False)

    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f'{s} did not solve optimally: '
            f'{res.solver.termination_condition}'
        )

    revenue = pyo.value(m.objective)

    # Extract results
    results[s] = {
        'model': m,
        'objective': revenue,
        'battery_E': [
            pyo.value(m.battery_E[t])
            for t in m.T_battery
        ],
        'battery_charge_P': [
            pyo.value(m.battery_charge_P[t])
            for t in m.T
        ],
        'battery_discharge_P': [
            pyo.value(m.battery_discharge_P[t])
            for t in m.T
        ],
        'battery_P': [
            pyo.value(m.battery_charge_P[t])
            - pyo.value(m.battery_discharge_P[t])
            for t in m.T
        ],
        'grid_P': [
            pyo.value(m.grid_import_P[t])
            - pyo.value(m.grid_export_P[t])
            for t in m.T
        ],
        'pv_P': list(pv),
        'loss_P': [
            pyo.value(m.loss_P[t])
            for t in m.T
        ],
    }

    pv_energy = sum(pv) * dt
    pv_value = sum(
        pv[t] * prices[t] * dt for t in range(n_steps)
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
print(df_results.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

R_WS = sum(
    pi[s] * results[s]['objective']
    for s in scenarios
)

print(f'\nExpected wait-and-see revenue: {R_WS:.2f} NOK')

#######################
# Plot each scenario
#######################

xs = list(range(n_steps))

for s in scenarios:
    r = results[s]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    fig.suptitle(
        f'Scenario-by-scenario optimisation: {s}',
        fontsize=16
    )

    plots = [
        (axes[0, 0], r['battery_E'][:-1],
         'Battery Energy', 'Energy [kWh]'),
        (axes[0, 1], r['battery_P'],
         'Battery Power', 'Power [kW]'),
        (axes[1, 0], r['grid_P'],
         'Grid Power', 'Power [kW]'),
        (axes[1, 1], r['pv_P'],
         'PV Production', 'Power [kW]'),
    ]

    for ax, data, title, ylabel in plots:
        ax.step(xs, data, label=title, where='post')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.step(
            xs, prices, label='Spot Price',
            linestyle='--', where='post'
        )
        ax2.set_ylabel('Price [NOK/kWh]')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

    axes[1, 0].set_xlabel('Time step [15 min]')
    axes[1, 1].set_xlabel('Time step [15 min]')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()
