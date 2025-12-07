data {
  int<lower=1> N;
  int<lower=1> P;
  vector[N] y;
  matrix[N, P] X;

  int<lower=1> N_edges;
  array[N_edges] int<lower=1> node1;
  array[N_edges] int<lower=1> node2;
  array[N] int<lower=0> num_neighbors;
}

parameters {
  vector[P] beta;
  vector[N] u;
  real<lower=0> sigma;
  real<lower=0> tau;

  // rho must be between 0 and 1
  real<lower=0, upper=1> rho;
}

model {

  // Priors
  beta  ~ normal(0, 1);
  sigma ~ normal(0, 1);
  tau   ~ normal(0, 1);

  // ρ ~ Normal(0.5, 0.2) truncated to [0,1]
  rho ~ normal(0.5, 0.2);

  // -------------------------
  // JOINT LEROUX CAR PRIOR
  // -------------------------

  // (1 - rho) * u'u
  target += -0.5 * (1 - rho) / square(tau) * dot_self(u);

  // rho * (D - W) contribution
  for (i in 1:N)
    target += -0.5 * rho / square(tau) * num_neighbors[i] * square(u[i]);

  for (e in 1:N_edges) {
    int i = node1[e];
    int j = node2[e];
    target += rho / square(tau) * u[i] * u[j];
  }

  // Likelihood
  y ~ normal(X * beta + u, sigma);
}
