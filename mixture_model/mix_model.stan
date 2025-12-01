
data {
  int<lower=1> N;          
  int<lower=1> P;          
  int<lower=1> K;          
  vector[N] y_star;             
  matrix[N, P] X_star;          
  vector[N] offset_star;        
}

parameters {
  simplex[K] pi;           
  vector[K] alpha;         
  matrix[K, P] beta;       
  real<lower=0> sigma;     
}

model {
  // Priors
  pi ~ dirichlet(rep_vector(1.0, K));   // Dirichlet(1,...,1)

  for (k in 1:K) {
    alpha[k] ~ normal(0, 5);
    for (j in 1:P) {
      beta[k, j] ~ normal(0, 5);
    }
  }

  sigma ~ normal(0, 1);   // half-normal(0,1)

  // Likelihood
  for (i in 1:N) {
    vector[K] lps;
    for (k in 1:K) {
      real mu_ik;
      mu_ik = alpha[k] + dot_product(row(X_star, i), beta[k]) + offset_star[i];
      lps[k] = log(pi[k]) + normal_lpdf(y_star[i] | mu_ik, sigma);
    }
    target += log_sum_exp(lps);
  }
}
