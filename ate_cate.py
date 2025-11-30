# Import libraries
import os, warnings, random
import dowhy
import econml
from dowhy import CausalModel
import pandas as pd
import numpy as np
np.int = np.int32
from sklearn.preprocessing import PolynomialFeatures
from zepid.graphics import EffectMeasurePlot
import matplotlib.pyplot as plt
import scipy.stats as stats
from zepid.graphics import EffectMeasurePlot
from itertools import product
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs, geom_point, geom_hline, theme_linedraw, theme, element_rect, theme_light, element_line, element_text, geom_errorbarh, geom_vline, theme_bw, element_blank
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.preprocessing import minmax_scale
from econml.dml import LinearDML
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler


# Set seeds to make the results more reproducible
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#%%

# Create DataFrame for ATE results - CORRECTED: 4 DataFrames
pd.set_option('display.precision', 5)
pd.set_option('display.float_format', '{:.5f}'.format)

# Create DataFrames
data_fire_ATE_all = pd.DataFrame(0.0, index=range(0, 8), columns=['ATE', 'ci_lower', 'ci_upper'])
data_fire_ATE_all = data_fire_ATE_all.astype({'ATE': 'float64', 'ci_lower': 'float64', 'ci_upper': 'float64'})

data_nofire_ATE_all = pd.DataFrame(0.0, index=range(0, 8), columns=['ATE', 'ci_lower', 'ci_upper'])
data_nofire_ATE_all = data_nofire_ATE_all.astype({'ATE': 'float64', 'ci_lower': 'float64', 'ci_upper': 'float64'})

# Dictionary to access DataFrames
ate_dataframes = {
    'all': {'fire': data_fire_ATE_all, 'nofire': data_nofire_ATE_all},
}

#%%

data = pd.read_csv("D:/top50_data.csv", encoding='latin-1')

# 1. Label Encoding DANE
le = LabelEncoder()
data['DANE_labeled'] = le.fit_transform(data['DANE'])
scaler = MinMaxScaler()
data['DANE_normalized'] = scaler.fit_transform(
    data[['DANE_labeled']]
)

# 2. Label Encoding DANE_year
le_year = LabelEncoder()
data['DANE_year_labeled'] = le_year.fit_transform(data['DANEYear'])
scaler_year = MinMaxScaler()
data['DANE_year_normalized'] = scaler_year.fit_transform(
    data[['DANE_year_labeled']]
)

# Convert aerosol and pop density columns to binary based on median
columns = data.columns[9:18] 
medians = data[columns].median()
data[columns] = data[columns].apply(
        lambda x: (x > medians[x.name]).astype(int)
    )

data['DUSMASS25'] = data['DUSMASS25'] / 10


data_fire = data[data['wildfire'] == 1]
data_nofire = data[data['wildfire'] == 0]

# Define outcomes
outcomes_all = ['excess_all_1', 'excess_all_2', 'excess_all_3', 'excess_all_4', 'excess_all_5', 'excess_all_6', 'excess_all_7', 'excess_all_8']

all_outcomes = {
    'all': outcomes_all,
}

#%%

def get_dag_string(outcome_var):

    return f"""graph[directed 1
                node[id "DUSMASS25" label "DUSMASS25"]
                node[id "{outcome_var}" label "{outcome_var}"]
                node[id "BCSMASS" label "BCSMASS"]
                node[id "DMSSMASS" label "DMSSMASS"]
                node[id "SO4SMASS" label "SO4SMASS"]
                node[id "SO2SMASS" label "SO2SMASS"]
                node[id "OCSMASS" label "OCSMASS"]
                node[id "DUSMASS" label "DUSMASS"]
                node[id "PBLH" label "PBLH"]
                node[id "TLML" label "TLML"]
                node[id "SPEED" label "SPEED"]
                node[id "pop_density" label "pop_density"]
                node[id "DANE_normalized" label "DANE_normalized"]
                node[id "DANE_year_normalized" label "DANE_year_normalized"]


                edge[source "BCSMASS" target "DUSMASS25"]
                edge[source "BCSMASS" target "DUSMASS"]
                edge[source "BCSMASS" target "DMSSMASS"]
                edge[source "BCSMASS" target "SO4SMASS"]
                edge[source "BCSMASS" target "SO2SMASS"]
                edge[source "BCSMASS" target "OCSMASS"]

                edge[source "DMSSMASS" target "DUSMASS25"]
                edge[source "DMSSMASS" target "DUSMASS"]
                edge[source "DMSSMASS" target "SO4SMASS"]
                edge[source "DMSSMASS" target "SO2SMASS"]
                edge[source "DMSSMASS" target "OCSMASS"]

                edge[source "SO4SMASS" target "DUSMASS25"]
                edge[source "SO4SMASS" target "DUSMASS"]
                edge[source "SO4SMASS" target "SO2SMASS"]
                edge[source "SO4SMASS" target "OCSMASS"]

                edge[source "SO2SMASS" target "DUSMASS25"]
                edge[source "SO2SMASS" target "DUSMASS"]
                edge[source "SO2SMASS" target "OCSMASS"]

                edge[source "OCSMASS" target "DUSMASS25"]
                edge[source "OCSMASS" target "DUSMASS"]

                edge[source "DUSMASS" target "DUSMASS25"]

                edge[source "TLML" target "PBLH"]
                edge[source "TLML" target "SPEED"]
                edge[source "SPEED" target "PBLH"]

                edge[source "TLML" target "DUSMASS25"]
                edge[source "TLML" target "BCSMASS"]
                edge[source "TLML" target "DMSSMASS"]
                edge[source "TLML" target "SO4SMASS"]
                edge[source "TLML" target "SO2SMASS"]
                edge[source "TLML" target "OCSMASS"]
                edge[source "TLML" target "DUSMASS"]

                edge[source "SPEED" target "DUSMASS25"]
                edge[source "SPEED" target "BCSMASS"]
                edge[source "SPEED" target "DMSSMASS"]
                edge[source "SPEED" target "SO4SMASS"]
                edge[source "SPEED" target "SO2SMASS"]
                edge[source "SPEED" target "OCSMASS"]
                edge[source "SPEED" target "DUSMASS"]

                edge[source "PBLH" target "DUSMASS25"]
                edge[source "PBLH" target "BCSMASS"]
                edge[source "PBLH" target "DMSSMASS"]
                edge[source "PBLH" target "SO4SMASS"]
                edge[source "PBLH" target "SO2SMASS"]
                edge[source "PBLH" target "OCSMASS"]
                edge[source "PBLH" target "DUSMASS"]

                edge[source "pop_density" target "DUSMASS25"]
                edge[source "pop_density" target "BCSMASS"]
                edge[source "pop_density" target "DMSSMASS"]
                edge[source "pop_density" target "SO4SMASS"]
                edge[source "pop_density" target "SO2SMASS"]
                edge[source "pop_density" target "OCSMASS"]
                edge[source "pop_density" target "DUSMASS"]
                edge[source "pop_density" target "PBLH"]
                edge[source "pop_density" target "SPEED"]

                edge[source "DUSMASS25" target "{outcome_var}"]

                edge[source "BCSMASS" target "{outcome_var}"]
                edge[source "DMSSMASS" target "{outcome_var}"]
                edge[source "SO4SMASS" target "{outcome_var}"]
                edge[source "SO2SMASS" target "{outcome_var}"]
                edge[source "OCSMASS" target "{outcome_var}"]
                edge[source "DUSMASS" target "{outcome_var}"]

                edge[source "TLML" target "{outcome_var}"]
                edge[source "pop_density" target "{outcome_var}"]      
                
                edge[source "DANE_normalized" target "{outcome_var}"]
                edge[source "DANE_year_normalized" target "{outcome_var}"] 
      
                  
            ]"""

# --- Function to get specific model parameters ---
def get_model_params(dataset_name, outcome_name):
    """
    Returns a dictionary with specific parameters for a dataset and outcome.
    """
    params_map = {
        ('fire', 'excess_all_1'): {"n_estimators": 3700, "max_depth": 31, "min_samples_leaf": 90},
        ('fire', 'excess_all_2'): {"n_estimators": 3400, "max_depth": 35, "min_samples_leaf": 80},
        ('fire', 'excess_all_3'): {"n_estimators": 3700, "max_depth": 31, "min_samples_leaf": 90},
        ('fire', 'excess_all_4'): {"n_estimators": 3500, "max_depth": 39, "min_samples_leaf": 68},
        ('fire', 'excess_all_5'): {"n_estimators": 3600, "max_depth": 32, "min_samples_leaf": 60},
        ('fire', 'excess_all_6'): {"n_estimators": 3600, "max_depth": 33, "min_samples_leaf": 100},
        ('fire', 'excess_all_7'): {"n_estimators": 3700, "max_depth": 31, "min_samples_leaf": 90},
        ('fire', 'excess_all_8'): {"n_estimators": 3600, "max_depth": 33, "min_samples_leaf": 100},
        
        ('nofire', 'excess_all_1'): {"n_estimators": 3400, "max_depth": 35, "min_samples_leaf": 80},
        ('nofire', 'excess_all_2'): {"n_estimators": 3400, "max_depth": 35, "min_samples_leaf": 80},
        ('nofire', 'excess_all_3'): {"n_estimators": 3600, "max_depth": 32, "min_samples_leaf": 60},
        ('nofire', 'excess_all_4'): {"n_estimators": 3400, "max_depth": 35, "min_samples_leaf": 80},
        ('nofire', 'excess_all_5'): {"n_estimators": 3600, "max_depth": 33, "min_samples_leaf": 100},
        ('nofire', 'excess_all_6'): {"n_estimators": 3600, "max_depth": 33, "min_samples_leaf": 100},
        ('nofire', 'excess_all_7'): {"n_estimators": 3600, "max_depth": 32, "min_samples_leaf": 60},
        ('nofire', 'excess_all_8'): {"n_estimators": 3600, "max_depth": 33, "min_samples_leaf": 100},
    }
    
    key = (dataset_name, outcome_name)
    if key in params_map:
        return params_map[key]
    else:

        print(f"Warning: No specific parameters found for dataset='{dataset_name}', outcome='{outcome_name}'. Using default values.")
        return {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 60} # Default options


# Define labels for outcomes
labels = [
    'Excess of accumulated deaths up to 1 day after',
    'Excess of accumulated deaths up to 2 days after',
    'Excess of accumulated deaths up to 3 days after',
    'Excess of accumulated deaths up to 4 days after',
    'Excess of accumulated deaths up to 5 days after',   
    'Excess of accumulated deaths up to 6 days after',
    'Excess of accumulated deaths up to 7 days after',
    'Excess of accumulated deaths up to 8 days after'
]

# Create a mapping dictionary: outcome_name -> label
outcome_to_label = {
    f'excess_all_{i+1}': label
    for i, label in enumerate(labels)
}


def process_outcomes(data_fire, data_nofire, outcomes_dict, ate_dataframes, outcome_to_label):
    """
    Processes all outcomes for fire and nofire groups
    """
    
    for group, outcomes in outcomes_dict.items():
        print(f"\nProcessing group: {group}")
        
        # Get specific DataFrames for this group
        data_fire_ATE_current = ate_dataframes[group]['fire']
        data_nofire_ATE_current = ate_dataframes[group]['nofire']
        
        for i, outcome in enumerate(outcomes):
            print(f"\nProcessing outcome: {outcome}")
            
            # Filter data for the current outcome
            data_fire_current = data_fire[[
                           'DUSMASS25', 'BCSMASS', 'DMSSMASS', 'DUSMASS', 'OCSMASS', 'SO2SMASS', 'SO4SMASS',
                           'TLML', 'pop_density', 'DANE_normalized', 'DANE_year_normalized', outcome]].dropna()
            
            data_nofire_current = data_nofire[[
                           'DUSMASS25', 'BCSMASS', 'DMSSMASS', 'DUSMASS', 'OCSMASS', 'SO2SMASS', 'SO4SMASS',
                           'TLML', 'pop_density', 'DANE_normalized', 'DANE_year_normalized', outcome]].dropna()
            
            # Create DAG string with the specific outcome
            dag_string = get_dag_string(outcome)
            
            # Create causal models
            model_fire = CausalModel(
                data=data_fire_current,
                treatment=['DUSMASS25'],
                outcome=[outcome],
                effect_modifiers=['TLML', 'DANE_normalized', 'DANE_year_normalized'],
                common_causes=['BCSMASS', 'DMSSMASS', 'DUSMASS', 'OCSMASS', 'SO2SMASS', 'SO4SMASS', 'TLML', 'pop_density'],
                graph=dag_string
            )    
            
            model_nofire = CausalModel(
                data=data_nofire_current,
                treatment=['DUSMASS25'],
                outcome=[outcome],
                effect_modifiers=['TLML', 'DANE_normalized', 'DANE_year_normalized'],
                common_causes=['BCSMASS', 'DMSSMASS', 'DUSMASS', 'OCSMASS', 'SO2SMASS', 'SO4SMASS', 'TLML', 'pop_density'],
                graph=dag_string
            )    
            
            # Identify effects
            identified_estimand_fire = model_fire.identify_effect(proceed_when_unidentifiable=None)
            identified_estimand_nofire = model_nofire.identify_effect(proceed_when_unidentifiable=None)
            
            print(f"Identified estimand for {outcome} (fire):")
            print(identified_estimand_fire)
            
            print(f"Identified estimand for {outcome} (no fire):")
            print(identified_estimand_nofire)
            
            # --- Get specific parameters for Fire ---
            params_fire = get_model_params('fire', outcome)
            print(f"Using parameters for Fire - {outcome}: {params_fire}")
            
            # Estimate effects for Fire
            estimate_fire = model_fire.estimate_effect(
                identified_estimand_fire,
                effect_modifiers=['TLML', 'DANE_normalized', 'DANE_year_normalized'],
                method_name="backdoor.econml.dml.CausalForestDML",
                confidence_intervals=True,
                method_params={
                    "init_params": {
                        "model_y": "auto",
                        "model_t": "auto",
                        "discrete_outcome": True,
                        "discrete_treatment": False,
                        **params_fire, # Unpack specific parameters
                        "cv": 5,
                        "random_state": 123
                    },
                }
            )
            
            # --- Get specific parameters for NoFire ---
            params_nofire = get_model_params('nofire', outcome)
            print(f"Using parameters for NoFire - {outcome}: {params_nofire}")
            
            # Estimate effects for NoFire
            estimate_nofire = model_nofire.estimate_effect(
                identified_estimand_nofire,
                effect_modifiers=['TLML', 'DANE_normalized', 'DANE_year_normalized'],
                method_name="backdoor.econml.dml.CausalForestDML",
                confidence_intervals=True,
                method_params={
                    "init_params": {
                        "model_y": "auto",
                        "model_t": "auto",
                        "discrete_outcome": True,
                        "discrete_treatment": False,
                        **params_nofire, # Unpack specific parameters
                        "cv": 5,
                        "random_state": 123
                    },
                }
            )
            
            #  ATE and CI for fire
                        
            # Define effect modifier variables
            effect_modifiers = ['TLML', 'DANE_normalized', 'DANE_year_normalized']            
            
            estimator_fire = estimate_fire.estimator.estimator
            X_data_fire = data_fire_current[effect_modifiers].dropna()
            ate_fire = estimator_fire.ate(X=X_data_fire)
            ate_ci_fire = estimator_fire.ate_interval(X=X_data_fire, alpha=0.05)
            ci_lower_fire = ate_ci_fire[0]
            ci_upper_fire = ate_ci_fire[1]
            
            data_fire_ATE_current.at[i, 'ATE'] = ate_fire
            data_fire_ATE_current.at[i, 'ci_lower'] = ci_lower_fire
            data_fire_ATE_current.at[i, 'ci_upper'] = ci_upper_fire
            
            # Calculate ATE and CI for nofire
            estimator_nofire = estimate_nofire.estimator.estimator
            X_data_nofire = data_nofire_current[effect_modifiers].dropna()  
            ate_nofire = estimator_nofire.ate(X=X_data_nofire)
            ate_ci_nofire = estimator_nofire.ate_interval(X=X_data_nofire, alpha=0.05)
            ci_lower_nofire = ate_ci_nofire[0]
            ci_upper_nofire = ate_ci_nofire[1]
            
            data_nofire_ATE_current.at[i, 'ATE'] = ate_nofire
            data_nofire_ATE_current.at[i, 'ci_lower'] = ci_lower_nofire
            data_nofire_ATE_current.at[i, 'ci_upper'] = ci_upper_nofire
                                    
            # Print ATE DataFrames
            print(f"Data fire ATE for {group} - {outcome}:")
            print(data_fire_ATE_current)
            print(f"Data nofire ATE for {group} - {outcome}:")
            print(data_nofire_ATE_current)                   
            
            # Figure 3
            print(f"\nGenerating CATE plot for {outcome} - Fire...")
            
            # Get descriptive label
            label_text = outcome_to_label.get(outcome, outcome)
            title_text_fire = f"{label_text}"
            
            temp_fire = data_fire_current['TLML']
            min_temp_fire = temp_fire.min()
            max_temp_fire = temp_fire.max()
            delta_fire = (max_temp_fire - min_temp_fire) / 100
            temp_grid_fire = np.arange(min_temp_fire, max_temp_fire + delta_fire - 0.001, delta_fire)
            
            DANE_encoded_mean_fire = data_fire_current['DANE_normalized'].mean()
            DANE_year_encoded_mean_fire = data_fire_current['DANE_year_normalized'].mean()
            
            X_test_grid_fire_temp = np.column_stack([
                temp_grid_fire,
                np.full_like(temp_grid_fire, DANE_encoded_mean_fire),
                np.full_like(temp_grid_fire, DANE_year_encoded_mean_fire),
            ])
            
            treatment_effect_fire_temp = estimator_fire.effect(X_test_grid_fire_temp)
            hte_lower_fire_temp, hte_upper_fire_temp = estimator_fire.effect_interval(X_test_grid_fire_temp, alpha=0.05)
            
            plot_data_temp_fire = pd.DataFrame({
                'x': temp_grid_fire,
                'treatment_effect': treatment_effect_fire_temp.flatten(),
                'ci_lower': hte_lower_fire_temp.flatten(),
                'ci_upper': hte_upper_fire_temp.flatten()
            })
            
            
            # Create Fire plot
            cate_plot_fire_temp = (
                ggplot(plot_data_temp_fire)
                + aes(x='x', y='treatment_effect')
                + geom_line(color='red', size=1)
                + geom_ribbon(aes(ymin='ci_lower', ymax='ci_upper'), alpha=0.2, fill='red')
                + labs(
                    x='Temperature °C',
                    y='Effect of PM2.5 on excess CVD deaths',
                    title=title_text_fire
                )
                + geom_hline(yintercept=0, color="blue", linetype="dashed", size=0.8)
                + theme(
                    plot_title=element_text(hjust=0.5, size=12),
                    axis_title_x=element_text(size=10),
                    axis_title_y=element_text(size=10)
                )
            )
            
            print(f"CATE plot for {outcome} (temperature) - Fire:")
            print(cate_plot_fire_temp)
            
            # Figure 4
            print(f"\nGenerating CATE plot for {outcome} - NoFire...")
            
            title_text_nofire = f"{label_text}"
            
            
            temp_nofire = data_nofire_current['TLML']
            min_temp_nofire = temp_nofire.min()
            max_temp_nofire = temp_nofire.max()
            delta_nofire = (max_temp_nofire - min_temp_nofire) / 100
            temp_grid_nofire = np.arange(min_temp_nofire, max_temp_nofire + delta_nofire - 0.001, delta_nofire)
            
            DANE_encoded_mean_nofire = data_nofire_current['DANE_normalized'].mean()
            DANE_year_encoded_mean_nofire = data_nofire_current['DANE_year_normalized'].mean()
            
            X_test_grid_nofire_temp = np.column_stack([
                temp_grid_nofire,
                np.full_like(temp_grid_nofire, DANE_encoded_mean_nofire),
                np.full_like(temp_grid_nofire, DANE_year_encoded_mean_nofire),
            ])
            
            treatment_effect_nofire_temp = estimator_nofire.effect(X_test_grid_nofire_temp)
            hte_lower_nofire_temp, hte_upper_nofire_temp = estimator_nofire.effect_interval(X_test_grid_nofire_temp, alpha=0.05)
            
            plot_data_temp_nofire = pd.DataFrame({
                'x': temp_grid_nofire,
                'treatment_effect': treatment_effect_nofire_temp.flatten(),
                'ci_lower': hte_lower_nofire_temp.flatten(),
                'ci_upper': hte_upper_nofire_temp.flatten()
            })
            
            
            # Create NoFire plot
            cate_plot_nofire_temp = (
                ggplot(plot_data_temp_nofire)
                + aes(x='x', y='treatment_effect')
                + geom_line(color='orange', size=1)
                + geom_ribbon(aes(ymin='ci_lower', ymax='ci_upper'), alpha=0.2, fill='orange')
                + labs(
                    x='Temperature °C',
                    y='Effect of PM2.5 on excess CVD deaths',
                    title=title_text_nofire
                )
                + geom_hline(yintercept=0, color="blue", linetype="dashed", size=0.8)
                + theme(
                    plot_title=element_text(hjust=0.5, size=12),
                    axis_title_x=element_text(size=10),
                    axis_title_y=element_text(size=10)
                )
            )
            
            print(f"CATE plot for {outcome} (temperature) - No Fire:")
            print(cate_plot_nofire_temp)
            
                       
            # Refutation tests for fire
            print(f"Refutation tests for {outcome} (fire):")
            random_fire = model_fire.refute_estimate(identified_estimand_fire, estimate_fire,
                                                    method_name="random_common_cause", random_state=123, num_simulations=50) 
            print(random_fire)
            
            subset_fire = model_fire.refute_estimate(identified_estimand_fire, estimate_fire,
                                                     subset_fraction=0.1, method_name="data_subset_refuter", random_state=123, num_simulations=50)
            print(subset_fire)
            
            placebo_fire = model_fire.refute_estimate(identified_estimand_fire, estimate_fire,
                                                     method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=50)
            print(placebo_fire)
            
            
            
            # Refutation tests for nofire
            print(f"Refutation tests for {outcome} (no fire):")
            random_nofire = model_nofire.refute_estimate(identified_estimand_nofire, estimate_nofire,
                                                        method_name="random_common_cause", random_state=123, num_simulations=50) 
            print(random_nofire)
            
            subset_nofire = model_nofire.refute_estimate(identified_estimand_nofire, estimate_nofire,
                                                     subset_fraction=0.1, method_name="data_subset_refuter", random_state=123, num_simulations=50)
            print(subset_nofire) 
            
            placebo_nofire = model_nofire.refute_estimate(identified_estimand_nofire, estimate_nofire,
                                                         method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=50)
            print(placebo_nofire)

    return ate_dataframes

# Process all outcomes
ate_dataframes,  = process_outcomes(
    data_fire, data_nofire, all_outcomes, ate_dataframes, outcome_to_label
)

# Save DataFrames
ate_dataframes['all']['fire'].to_csv("D:/data_fire_ATE.csv", index=False)
ate_dataframes['all']['nofire'].to_csv("D:/data_nofire_ATE.csv", index=False)

print("DataFrames saved:")
print("\nAll:")
print("Fire ATE:", ate_dataframes['all']['fire'])
print("No Fire ATE:", ate_dataframes['all']['nofire'])


#%%

# Fig2
labels = [
    'Excess of accumulated deaths up to 1 day after',
    'Excess of accumulated deaths up to 2 days after',
    'Excess of accumulated deaths up to 3 days after',
    'Excess of accumulated deaths up to 4 days after',
    'Excess of accumulated deaths up to 5 days after',
    'Excess of accumulated deaths up to 6 days after',
    'Excess of accumulated deaths up to 7 days after',
    'Excess of accumulated deaths up to 8 days after'
]

# Colors for each effect
colors = ['red', 'red', 'red', 'red', 'red', 'red', 'red', 'red']

# Vertical positions for each effect
y_positions = np.arange(len(labels))

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))


fire = ate_dataframes['all']['fire']


# Draw horizontal lines (confidence intervals)
for i, (ATE, ci_lower, ci_upper) in enumerate(fire.values):
    # Draw horizontal line
    ax.plot([ci_lower, ci_upper], [y_positions[i], y_positions[i]], 
            color=colors[i], linewidth=2.5)
    
    # Draw point at center (ATE)
    ax.plot(ATE, y_positions[i], marker='s', markersize=8, 
            color=colors[i], markerfacecolor=colors[i], markeredgecolor='red', 
            markeredgewidth=1)

# Add Y-axis labels
ax.set_yticks(y_positions)
ax.set_yticklabels(labels)

# Configure X-axis
ax.set_xlabel('Average Treatment Effect (ATE)')
ax.set_xlim(-1,1)  # Adjust according to your data
ax.grid(True, alpha=0.3)

# Vertical line at 0
ax.axvline(x=0, color='blue', linestyle='--', alpha=0.7)

# Title
ax.set_title('a', fontsize=14, pad=20)  # Changed the title

# Adjust margins
plt.tight_layout()

# Show figure
plt.show()


# Define labels for each row (from bottom to top)
labels = [
    'Excess of accumulated deaths up to 1 day after',
    'Excess of accumulated deaths up to 2 days after',
    'Excess of accumulated deaths up to 3 days after',
    'Excess of accumulated deaths up to 4 days after',
    'Excess of accumulated deaths up to 5 days after',
    'Excess of accumulated deaths up to 6 days after',
    'Excess of accumulated deaths up to 7 days after',
    'Excess of accumulated deaths up to 8 days after'
]

# Colors for each effect
colors = ['orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange']  

# Vertical positions for each effect
y_positions = np.arange(len(labels))

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Access the 'nofire' dataframe within the 'ate_dataframes' dictionary
nofire = ate_dataframes['all']['nofire']

# Draw horizontal lines (confidence intervals)
for i, (ATE, ci_lower, ci_upper) in enumerate(nofire.values):
    # Draw horizontal line
    ax.plot([ci_lower, ci_upper], [y_positions[i], y_positions[i]], 
            color=colors[i], linewidth=2.5)
    
    # Draw point at center (ATE)
    ax.plot(ATE, y_positions[i], marker='s', markersize=8, 
            color=colors[i], markerfacecolor=colors[i], markeredgecolor='orange', 
            markeredgewidth=1)

# Add Y-axis labels
ax.set_yticks(y_positions)
ax.set_yticklabels(labels)

# Configure X-axis
ax.set_xlabel('Average Treatment Effect (ATE)')
ax.set_xlim(-1,1)  # Adjust according to your data
ax.grid(True, alpha=0.3)

# Vertical line at 0
ax.axvline(x=0, color='blue', linestyle='--', alpha=0.7)

# Title
ax.set_title('b', fontsize=14, pad=20)  # Changed the title

# Adjust margins
plt.tight_layout()

# Show figure
plt.show()