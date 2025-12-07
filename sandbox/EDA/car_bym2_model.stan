data {
  int<lower=1> N;
  int<lower=1> P;
  vector[N] y;
  matrix[N, P] X;

  int<lower=1> N_edges;
  array[N_edges] int<lower=1> node1;
  array[N_edges] int<lower=1> node2;
  array[N] int<lower=0> num_neighbors;

  real<lower=0> scaling_factor; // computed in Python
}

parameters {
  vector[P] beta;

  // BYM2 parameters
  vector[N] theta;              // unstructured
  vector[N] phi;                // structured ICAR
  real<lower=0> sigma;          // overall SD
  real<lower=0, upper=1> rho;   // mixing proportion

  real<lower=0> sigma_y;        // likelihood SD
}

transformed parameters {
  vector[N] u;

  // scaled BYM2 random effect
  u = sigma * (
        sqrt(1 - rho) * theta +
        sqrt(rho / scaling_factor) * phi
      );
}

model {

  // Likelihood
  y ~ normal(X * beta + u, sigma_y);

  // Priors
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  sigma_y ~ normal(0, 1);
  rho ~ beta(0.5, 0.5);   // default BYM2 prior

  theta ~ normal(0, 1);

  // ICAR prior on phi: joint distribution
  target += -0.5 * dot_self(phi[node1] - phi[node2]);

  // soft sum-to-zero constraint
  sum(phi) ~ normal(0, 0.001 * N);
}

generated quantities {
  vector[N] u_struct = sigma * sqrt(rho / scaling_factor) * phi;
  vector[N] u_unstruct = sigma * sqrt(1 - rho) * theta;
}
