# Bayesian Statistics Project
This repository contains the final project developed for the **Bayesian Statistics** course at **Politecnico di Milano (PoliMi)**, taught by **Prof. Alessandra Guglielmi**. 

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

- Lee, D. (2013). **CARBayes: An R Package for Bayesian Spatial Modeling with Conditional Autoregressive Priors**. *Journal of Statistical Software*, 55(13), 1–24. https://doi.org/10.18637/JSS.V055.I13

- Park, T., & Casella, G. (2008). **The Bayesian Lasso**. *Journal of the American Statistical Association*, 103, 681–686. https://doi.org/10.1198/016214508000000337

- Wade, S., & Ghahramani, Z. (2019). **Bayesian cluster analysis: Point estimation and credible balls**.

- Besag, J. (1974). **Spatial Interaction and the Statistical Analysis of Lattice Systems**. *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 192–225. https://doi.org/10.1111/j.2517-6161.1974.tb00999.x

- Beraha, M., Pegoraro, M., Peli, R., & Guglielmi, A. (2021). **Spatially dependent mixture models via the logistic multivariate CAR prior**. *Spatial Statistics*, 46, 100548. https://doi.org/10.1016/j.spasta.2021.100548

- Mozdzen, A., Cremaschi, A., Cadonna, A., Guglielmi, A., & Kastner, G. (2022). **Bayesian modeling and clustering for spatio-temporal areal data: An application to Italian unemployment**. *Spatial Statistics*, 52, 100715. https://doi.org/10.1016/j.spasta.2022.100715

- Moran, P. A. P. (1950). **Notes on Continuous Stochastic Phenomena**. *Biometrika*, 37(1/2), 17. https://doi.org/10.2307/2332142

- Anselin, L. (1995). **Local Indicators of Spatial Association—LISA**. *Geographical Analysis*, 27(2), 93–115. https://doi.org/10.1111/j.1538-4632.1995.tb00338.x

- Malsiner-Walli, G., Frühwirth-Schnatter, S., & Grün, B. (2014). **Model-based clustering based on sparse finite Gaussian mixtures**. *Statistics and Computing*, 26(1), 303. https://doi.org/10.1007/S11222-014-9500-2

- Leroux, B. G., Lei, X., & Breslow, N. (2000). **Estimation of Disease Rates in Small Areas: A new Mixed Model for Spatial Dependence**. In M. E. Halloran & D. Berry (Eds.), *Statistical Models in Epidemiology, the Environment, and Clinical Trials* (pp. 179–191). Springer. https://doi.org/10.1007/978-1-4612-1284-3_4

- Binder, D. A. (1978). **Bayesian cluster analysis**. *Biometrika*, 65(1), 31–38. https://doi.org/10.1093/biomet/65.1.31

- Duncan, E. W., Cramb, S. M., Baade, P. D., Mengersen, K. L., Saunders, T., & Aitken, J. F. (2024). **Developing a Cancer Atlas using Bayesian Methods: A Practical Guide for Application and Interpretation** (2nd ed.). Queensland University of Technology (QUT) and Cancer Council Queensland. https://atlas.cancer.org.au/ebook/ebook2/Index.html
