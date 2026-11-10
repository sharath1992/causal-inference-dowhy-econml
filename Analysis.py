### import libraries ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import dowhy
from dowhy import CausalModel
from econml.dml import CausalForestDML
from Ipython.display import Image, display
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures


### Read data ###

df = pd.read_csv("hr_causal_dataset.csv")


### Data Audit and Cleaning ###

# Duplicate Employee Records

print("--- Duplicates ---")
df[df['employee_id'].duplicated(keep=False)].sort_values(by='employee_id')

# Remove duplicates

df = df.drop_duplicates(subset='employee_id', keep='first')

# Fix any bad data

# Negative Tenure Years

df['tenure_years'] = df['tenure_years'].abs()

# Invalid Base Salary

valid_salary = df['base_salary'] > 30000  #we will keep $30k as the min. threshold for valid base salary
salary_medians = df[valid_salary].groupby('job_level')['base_salary'].median()

for level, md in salary_medians.items():
    condition = (df['base_salary'] <= 30000) & (df['job_level'] == level)
    df.loc[condition, 'base_salary'] = md

df.loc[df['base_salary'] <= 30000, 'base_salary'] = df['base_salary'].median()

# Invalid Age

invalid_age = (df['age'] > 100 | df['age'] < 18)

print(f"Detected {invalid_age.sum()} age anamolies")

current_year = 2026

estimated_fixed_age = (
    26 + (current_year - df.loc[invalid_age, 'hiring_cohort']) + df.loc[invalid_age, 'tenure_years']
).astype(int)   # assume entry age for hiring as 26

df.loc[invalid_age, 'age'] = estimated_fixed_age

print(f"Cleaned Age Range: {df['age'].min()} to {df['age'].max()}")
print(f"Cleaned Base Salary Range: ${df['base_salary'].min():,.2f} to ${df['base_salary'].max():,.2f}")

# Standardizing Department Names

df['department'] = df['department'].astype(str).str.strip().str.title()

dept_mapping = {
    "Data Eng": "Data Engineering",
    "Date engineering": "Data Engineering",
    "Cloud Infra": "Cloud Infrastructure"
}

df['department'] = df['department'].replace(dept_mapping)

print(f"Department Names after cleaning: {df['department'].unique()}")

# Missing Counts

print(f"Missing Row Counts:{df.isnull().sum()}")

# Handling Missing Engagement Scores (MNAR)

df['engagement_missing_flag'] = df['engagement_score'].isnull().astype(int)

# Impute using median

df['engagement_score'] = df['engagement_score'].fillna(
    df.groupby(['department', 'job_level'])['engagement_score'].transform('median')
)
df['engagement_score'] = df['engagement_score'].fillna(df['engagement_score'].median())

print(f"Missing engagement scores after imputation: {df['engagement_score'].isnull().sum()}")


### EDA ###

# Calculate Naive Effect

treated_group = df[df['enrolled_in_program'] == 1]
control_group = df[df['enrolled_in_program'] == 0]

naive_effect = treated_group['performance_score_12m'].mean() - control_group['performance_score_12m'].mean()

print(f"Naive Causal Effect: {naive_effect}")

# Confounders - check covariate balance

treatment_col = 'enrolled_in_program'

confounders = [
    'tenure_years',
    'base_salary',
    'manager_quality_score',
    'commute_minutes',
    'engagement_score'
]

balance_results = []

for cf in confounders:
    mean_treated = df.loc[df[treatment_col] == 1, cf].mean()
    mean_control = df.loc[df[treatment_col] == 0, cf].mean()

    n_t = (df[treatment_col] == 1).sum()
    n_c = (df[treatment_col] == 0).sum()

    var_t = df.loc[df[treatment_col] == 1, cf].var()
    var_c = df.loc[df[treatment_col] == 0, cf].var()

    stdev = np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2))

    smd = (mean_treated - mean_control) / stdev if stdev != 0 else 0

    balance_results.append(
        {
            'confounder': cf,
            'Treated Mean': round(mean_treated, 2),
            'Control Mean': round(mean_control, 2),
            'Absolute SMD': round(abs(smd), 3)
        }
    )

print(f"Covariate Balance Results: {balance_results}")

balance_df = pd.DataFrame(balance_results)
print(balance_df.to_string(index=False))
print("Absolute SMD > 0.25 typically indicates meaningful imbalance between treatment groups")
print("High imbalance confirms the presence of strong selection bias requiring causal adjustment")

# Propensity Score

X = df[confounders]
y = df[treatment_col]

ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X,y)

df['propensity_score'] = ps_model.predict_proba(X)[:,1]

treated_ps = df.loc[df[treatment_col] == 1, 'propensity_score']
control_ps = df.loc[df[treatment_col] == 0, 'propensity_score']

print(f"Treated Propensity Score Range: [{treated_ps.min()}, {treated_ps.max()}]")
print(f"Control Propensity Score Range: [{control_ps.min()}, {control_ps.max()}]")

min_overlap = max(treated_ps.min(), control_ps.min())
max_overlap = min(treated_ps.max(), control_ps.max())

print(f"Overlap Region: [{min_overlap}, {max_overlap}]")

if treated_ps.min() < control_ps.min() or treated_ps.max() > control_ps.max():
    print("Minor differences in extreme tails detected. As long as there is good overlap in the central density region (common support), DoWhy and EconML " \
    "estimators can proceed safely. Extreme outliers outside common support can be trimmed if necessary")
else:
    print("Excellent common support across the distribution.")


### Causal Modeling ###

# DAG

causal_dag = """
    digraph {
        tenure_years -> enrolled_in_program;
        base_salary -> enrolled_in_program;
        manager_quality_score -> enrolled_in_program;
        engagement_score -> enrolled_in_program;
        job_level -> enrolled_in_program;
        
        tenure_years -> performance_score_12m;
        base_salary -> performance_score_12m;
        manager_quality_score -> performance_score_12m;
        engagement_score -> performance_score_12m;
        job_level -> performance_score_12m;
        
        enrolled_in_program -> performance_score_12m;
    }
    """

# Causal Model

model = CausalModel(
    data=df,
    treatment='enrolled_in_program',
    outcome='performance_score_12m',
    common_causes=[
        'tenure_years', 
        'base_salary', 
        'manager_quality_score', 
        'engagement_score', 
        'job_level'
    ],
    graph=causal_dag
)

# DAG Image Export

model.view_model(layout="dot", file_name="causal_model")
display(Image(filename="causal_model.png"))

# Estimand

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

print(f"Estimand Results from dowhy: {identified_estimand}")


### Causal Estimation using EconML ###

# Training Double ML Estimator using Backdoor estimation method

dml_estimate = model.estimate_effect(
    identified_estimand, 
    method_name="backdoor.econml.dml.DML",
    control_value=0,
    treatment_value=1,
    method_params={
        "init_params": {
            'model_y': GradientBoostingRegressor(random_state=42),
            'model_t': GradientBoostingRegressor(random_state=42),
            "model_final": LassoCV(fit_intercept=False),
            'featurizer': PolynomialFeatures(degree=2, include_bias=True)
        },
        "fit_params": {}
    }
)

print("\n--- DOUBLE ML ESTIMATION RESULTS ---")
print(dml_estimate)
print(f"Bias-Corrected Average Treatment Effect (ATE): {dml_estimate.value}")

# Causal Forest DML

job_mapping = {
    'L1 - Associate': 1,
    'L2 - Mid': 2,
    'L3 - Senior': 3,
    'L4 - Staff': 4,
    'L5 - Principal': 5
}

df['job_level_numeric'] = df['job_level'].map(job_mapping)

feature_cols = ['tenure_years', 'base_salary', 'manager_quality_score', 'engagement_score', 'job_level_numeric']

Xx = df[feature_cols].values

T = df['enrolled_in_program'].values

Y = df['performance_score_12m'].values

causal_forest = CausalForestDML(
    model_y=GradientBoostingRegressor(random_state=42),
    model_t=GradientBoostingRegressor(random_state=42), # Using regressor for binary T to maintain consistency
    n_estimators=1000,
    min_samples_leaf=10,
    random_state=42
)

causal_forest.fit(Y, T, X=Xx)

individual_effects = causal_forest.effect(Xx)

df['individual_program_lift'] = individual_effects

print(f"Mean Individual Causal Lift: {individual_effects.mean()}")

# Confidence Intervals for Causal Forest

lower_bounds, upper_bounds = causal_forest.effect_interval(Xx, alpha=0.05)

df['cate_lower'] = lower_bounds
df['cate_upper'] = upper_bounds

print(df[['tenure_years', 'job_level_numeric', 'individual_program_lift', 'cate_lower', 'cate_upper']].head())

# Group individual lifts by job level to see where the program works best

job_level_effect = df.groupby('job_level_numeric')['individual_program_lift'].agg(['mean', 'median', 'std', 'count'])
print("\n--- Treatment Effect by Job Level ---")
print(job_level_effect)

# Analyze effect by Tenure brackets

df['tenure_bracket'] = pd.cut(df['tenure_years'], bins=[0, 2, 5, 10, 30], labels=['<2 yrs', '2-5 yrs', '5-10 yrs', '10+ yrs'])
tenure_effect = df.groupby('tenure_bracket', observed=False)['individual_program_lift'].agg(['mean', 'median', 'count'])
print("\n--- Treatment Effect by Tenure ---")
print(tenure_effect)

# Analyze effect by Manager Quality Score quartiles

df['manager_quartile'] = pd.qcut(df['manager_quality_score'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
manager_effect = df.groupby('manager_quartile', observed=False)['individual_program_lift'].agg(['mean', 'median', 'count'])
print("\n--- Treatment Effect by Manager Quality Quartile ---")
print(manager_effect)


### Robustness and Refutation Checks ###

# Placebo Treatment Refuter: Randomly shuffles treatment assignment

placebo_refutation = model.refute_estimate(
    identified_estimand, 
    dml_estimate, 
    method_name="placebo_treatment_refuter", 
    placebo_type="permute",
    num_simulations=20
)

print("\n[1] Placebo Treatment Refuter:")
print(placebo_refutation)

# Random Common Cause Refuter: Adds a simulated random confounder to the data

random_cause_refutation = model.refute_estimate(
    identified_estimand, 
    dml_estimate, 
    method_name="random_common_cause",
    num_simulations=20
)

print("\n[2] Random Common Cause Refuter:")
print(random_cause_refutation)

