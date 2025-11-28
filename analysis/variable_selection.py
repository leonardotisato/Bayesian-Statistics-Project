# IMPORT
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/updated_data.csv")
excluded_cols = ['prov_name', 'fem_empl_rate', 'urban_type']
df = df.drop(columns=excluded_cols)

corr_matrix = df.corr()

# avoid duplicate pairs and self-correlation
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

threshold = 0.89

high_corr_pairs = (
    upper.stack()
         .reset_index()
         .rename(columns={'level_0': 'var1', 'level_1': 'var2', 0: 'corr'})
)

high_corr_pairs = (
    high_corr_pairs[high_corr_pairs['corr'].abs() >= threshold]
    .sort_values(by='corr', key=np.abs, ascending=False)
)

print("Highly correlated pairs")
print(high_corr_pairs.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

var_counts = (
    pd.concat([high_corr_pairs["var1"], high_corr_pairs["var2"]])
      .value_counts()
)

var_presence_list = list(var_counts.items())

for var, count in var_presence_list:
    print(f"({var}, {count})")
    
# we will talk about which one actually to drop

var_to_drop = ['empl_rate', 'inact_rate', 'male_edu_rate', 'other_per_capita_expenditure', 'adj_net_migration_rate']

df_reduced = df.drop(columns=var_to_drop)

# Other variables to consider dropping based on domain knowledge
to_drop = ['part_rate', 'unempl_rate']

df_reduced = df_reduced.drop(columns=to_drop)


selected_covariates = list(df_reduced.columns)

df_full = pd.read_csv("../data/cleaned_ECEC_labour_2022 dataset.csv")
target_col = "fem_empl_rate"

df_model = df_full[[target_col] + selected_covariates].dropna()

print("Selected covariates used in the model:")
print(selected_covariates)


X = df_model[selected_covariates].to_numpy()
y = df_model[target_col].to_numpy()


# standardize predictors and the target
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)

y_mean = y.mean()
y_std = y.std()
y = y / 100
y_scaled = np.log(y / (1 - y))

n_obs, n_feat = X_scaled.shape
print(f"\nNumber of observations: {n_obs}, number of predictors: {n_feat}")

print("X_data shape:", X_scaled.shape)
print("y_data shape:", y_scaled.shape)

# MODEL

with pm.Model() as bl_model:
    X_data = pm.Data("X_data", X_scaled)
    y_data = pm.Data("y_data", y_scaled)

    lambda_ = pm.Gamma("lambda", alpha=0.1, beta=0.1)
    sigma = pm.HalfNormal("sigma", sigma=1.0)

    beta = pm.Laplace(
        "beta",
        mu=0.0,
        b=sigma / lambda_,
        shape=n_feat
    )

    mu = pm.math.dot(X_data, beta)

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

    trace = pm.sample(
        draws=2000,
        tune=2000,
        target_accept=0.9,
        return_inferencedata=True,
        random_seed=42
    )
    
# RESULTS with 95% CI

summary_beta_95 = az.summary(trace, var_names=["beta"], hdi_prob=0.95)

summary_beta_95.index = selected_covariates

summary_beta_95["contains_zero"] = (
    (summary_beta_95["hdi_2.5%"] <= 0) &
    (summary_beta_95["hdi_97.5%"] >= 0)
)

summary_beta_95 = summary_beta_95.sort_values("mean", key=np.abs, ascending=False)

print(summary_beta_95[["mean", "hdi_2.5%", "hdi_97.5%", "contains_zero"]])

# RESULTS with 90% CI
summary_beta_95 = az.summary(trace, var_names=["beta"], hdi_prob=0.90)

summary_beta_95.index = selected_covariates

summary_beta_95["contains_zero"] = (
    (summary_beta_95["hdi_5%"] <= 0) &
    (summary_beta_95["hdi_95%"] >= 0)
)

summary_beta_95 = summary_beta_95.sort_values("mean", key=np.abs, ascending=False)

print(summary_beta_95[["mean", "hdi_5%", "hdi_95%", "contains_zero"]])

## general results
# az.rcParams["plot.max_subplots"] = 100   
# az.plot_trace(trace, compact=False)

az.plot_trace(trace)
plt.tight_layout()

beta_samples = trace.posterior["beta"].stack(samples=("chain", "draw")).values

plt.figure(figsize=(12,4))
sns.kdeplot(beta_samples.T, legend=False)
plt.title("Posterior density of β_{j}")
plt.xlabel("Values of β_{j}")
plt.yticks([])     
plt.gca().tick_params(axis='y', length=0)

plt.show()

plt.figure(figsize=(8,6))
summary_beta_95["mean"].sort_values().plot(kind='barh')
plt.grid(True)
plt.show()
