data {
  int<lower=1> N;                    // number of areas
  int<lower=1> P;                    // number of predictors
  vector[N] y;                       // response
  matrix[N, P] X;                    // predictors

  int<lower=1> N_edges;              // number of adjacency edges
  array[N_edges] int<lower=1> adj;   // adjacency list
  array[N] int<lower=0> num_neighbors;
}

parameters {
  vector[P] beta;                    // regression coefficients
  vector[N] spatial_raw;             // unscaled spatial effects
  real<lower=0> sigma;               // noise SD
  real<lower=0> tau;                 // CAR SD
  real<lower=0, upper=1> phi;        // spatial autocorrelation
}

transformed parameters {
  vector[N] spatial = tau * spatial_raw;
}

model {
  beta  ~ normal(0, 1);
  sigma ~ normal(0, 1);
  tau   ~ normal(0, 1);
  phi   ~ beta(2, 2);

  // Proper CAR (Leroux)
  {
    int pos = 1;
    for (i in 1:N) {

      if (num_neighbors[i] > 0) {
        vector[num_neighbors[i]] neigh_vals;

        for (j in 1:num_neighbors[i]) {
          neigh_vals[j] = spatial_raw[adj[pos]];
          pos += 1;
        }

        // Mean = φ * average(neighbors)
        real mean_i = phi * mean(neigh_vals);

        // Variance = 1.0 / num_neighbors[i]
        spatial_raw[i] ~ normal(mean_i, sqrt(1.0 / num_neighbors[i]));

      } else {
        // isolated region
        spatial_raw[i] ~ normal(0, 1);
      }
    }
  }

  // Likelihood
  y ~ normal(X * beta + spatial, sigma);
}
