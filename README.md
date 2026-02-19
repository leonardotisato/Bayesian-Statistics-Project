# Bayesian Statistics Project
This repository contains the final project developed for the **Bayesian Statistics** course at **Politecnico di Milano (PoliMi)**, taught by **Prof. Alessandra Guglielmi**, for the 2025-26 academic year. 

## Project Overview
This project investigates how **Early Childhood Education and Care (ECEC)** availability relates to **women’s labour market participation** across **Italian provinces (year 2022)**. We model the (logit-transformed) provincial female employment rate using a **Bayesian spatial regression** with a **CAR (Leroux) prior** to capture residual spatial dependence, and we explore **model-based spatial clustering** via a **sparse Bayesian finite mixture** where latent component memberships define province-level clusters. The workflow includes data preparation, EDA and spatial autocorrelation checks, **variable selection** (Bayesian Lasso with spatial effects), **label assignment** through posterior clustering summaries, and final inference on selected predictors and spatial effects. In the reported results, the posterior supports **a single cluster** (i.e., no evidence of distinct province groups with different covariate effects), so inference reduces to a global spatial regression with remaining geographic disparities absorbed by the spatial random effect. 

## Repository Structure

The project is organized into the following directories:

*   **`src/`**: Contains the core notebooks for data preparation, exploratory data analysis (EDA), variable selection, label assignment, and cluster specific parameter estimation.
*   **`sandbox/`**: Experimental code and analysis, organized by topic (e.g., `mixture_model`, `variable_selection`, `analysis`).
*   **`doc/`**: Documentation and presentation materials.
*   **`data/`**: Datasets used in the project.

## Tutors
*   Simone Colombara
*   Giulio Beltramin

## Authors
*   Tommaso Baresi
*   Luca Perego
*   Paul Poupeau
*   Leonardo Tisato
*   Alexis Toppè
*   Chi Huan Tuan

## References


- Park, T., & Casella, G. (2008). **The Bayesian Lasso**. *Journal of the American Statistical Association*, 103, 681–686. https://doi.org/10.1198/016214508000000337

- Wade, S., & Ghahramani, Z. (2019). **Bayesian cluster analysis: Point estimation and credible balls**.

- Mozdzen, A., Cremaschi, A., Cadonna, A., Guglielmi, A., & Kastner, G. (2022). **Bayesian modeling and clustering for spatio-temporal areal data: An application to Italian unemployment**. *Spatial Statistics*, 52, 100715. https://doi.org/10.1016/j.spasta.2022.100715

- Malsiner-Walli, G., Frühwirth-Schnatter, S., & Grün, B. (2014). **Model-based clustering based on sparse finite Gaussian mixtures**. *Statistics and Computing*, 26(1), 303. https://doi.org/10.1007/S11222-014-9500-2
