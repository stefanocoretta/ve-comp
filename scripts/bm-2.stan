data {
  // Number of observations
  int<lower=1> N;
  // Number of participants
  int<lower=1> P;
  // Number of languages
  int<lower=1> L;
  // Vowel duration, logged
  vector[N] y;
  // Voicing (0 = voiceless, 1 = voiced)
  array[N] int<lower=0,upper=1> voicing;
  // Number of syllables (0 = di, 1 = mono)
  array[N] int<lower=0,upper=1> syllables;
  // Participant ID
  array[N] int<lower=1,upper=P> participant;
  // Participant's language (1 = English, 2 = Italian, 3 = Polish)
  array[N] int<lower=1,upper=L> language;
}

parameters {
  // Intercept by language
  vector[L] intercept;
  // Effect of voiced by language
  vector[L] b_voicing;
  // Effect of mono in English
  real b_syllables_en;
  // Effect of mono:syllables in English
  real b_voicing_syllables_en;

  // Participant-level varying effects (4: intercept, voicing, syllables_en, voicing_syllables_en)
  // Standard normal z
  matrix[P, 4] z_participant;
  // Standard deviations of participants
  vector<lower=0>[4] sigma_participant;
   // Cholesky factor of correlation matrix
  cholesky_factor_corr[4] L_Rho_participant;

  // Residual SD
  real<lower=0> sigma;
}

transformed parameters {
  matrix[P,4] r_participant;
  r_participant = (z_participant
                         * diag_pre_multiply(sigma_participant, L_Rho_participant));

  vector[N] mu;
  for (n in 1:N) {
    int lang = language[n];
    int part = participant[n];

    // 1 = intercept, 2 = voicing, 3 = syllables_en, 4 = voicing_syllables_en
    real r_intercept = r_participant[part,1];
    real r_voicing = r_participant[part,2];
    real r_syllables_en = (lang == 1) ? r_participant[part,3] : 0.0;
    real r_voicing_syllables_en = (lang == 1) ? r_participant[part,4] : 0.0;

    real b_syllables = (lang == 1) ? b_syllables_en * syllables[n] : 0.0;
    real b_voicing_syllables = (lang == 1) ? b_voicing_syllables_en * voicing[n] * syllables[n] : 0.0;

    mu[n] = intercept[lang]
            + r_intercept
            + (b_voicing[lang] + r_voicing) * voicing[n]
            + b_syllables
            + b_voicing_syllables
            + r_syllables_en * syllables[n]
            + r_voicing_syllables_en * voicing[n] * syllables[n];
  }
}

model {
  intercept ~ normal(0, 10);
  b_voicing ~ normal(0, 5);
  b_syllables_en ~ normal(0, 5);
  b_voicing_syllables_en ~ normal(0, 5);

  // Priors for varying effects
  to_vector(z_participant) ~ normal(0, 1);
  sigma_participant ~ cauchy(0, 2);
  L_Rho_participant ~ lkj_corr_cholesky(2);

  // Prior for residual SD
  sigma ~ cauchy(0, 2);

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  corr_matrix[4] corr_participant;
  corr_participant = multiply_lower_tri_self_transpose(L_Rho_participant);
}
