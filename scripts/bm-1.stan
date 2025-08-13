data {
  int<lower=1> N;               // total observations
  int<lower=1> P;               // number of participants
  int<lower=1> L;               // number of languages (3: IT=2, PL=3, EN=1)
  
  vector[N] y;                  // vowel duration
  
  array[N] int<lower=0,upper=1> voicing;      // consonant voicing (0=voiceless,1=voiced)
  array[N] int<lower=0,upper=1> syllables;    // number of syllables (0=di,1=mono)
  
  array[N] int<lower=1,upper=P> participant;  // participant ID
  array[N] int<lower=1,upper=L> language;     // language ID
}

parameters {
  // Intercept by language
  vector[L] intercept;
  // Effect of voiced by language
  vector[L] b_voicing;
  // Effect of mono and voiced:mono only for English (lang = 1)
  real b_syllables_en;
  real b_voicing_syllables_en;
  // Participant varying intercepts
  vector[P] z_participant;
  // Participant varying voicing slopes (pooled across languages)
  vector[P] z_voicing;
  // Participant varying syllables and interaction slopes only for English (lang = 1)
  vector[P] z_syllables_en;
  vector[P] z_voicing_syllables_en;
  // Varying effect standard deviations
  real<lower=0> sigma_participant;
  real<lower=0> sigma_voicing;
  real<lower=0> sigma_syllables_en;
  real<lower=0> sigma_voicing_syllables_en;
  // Residual sd
  real<lower=0> sigma;
}

transformed parameters {
  vector[N] mu;
  
  for (n in 1:N) {
    int lang = language[n];
    int part = participant[n];
    
    // Random syllables and interaction only for English (lang==1)
    real z_syllables_part = 0;
    real z_voicing_syllables_en_part = 0;
    if (lang == 1) {
      z_syllables_part = z_syllables_en[part];
      z_voicing_syllables_en_part = z_voicing_syllables_en[part];
    }
    
    // Effects of mono and mono:voiced for English
    real b_syllables = 0;
    real b_voicing_syllables = 0;
    if (lang == 1) {
      b_syllables = b_syllables_en * syllables[n];
      b_voicing_syllables = b_voicing_syllables_en * voicing[n] * syllables[n];
    }
    
    mu[n] = intercept[lang]
            + sigma_participant * z_participant[part]
            + (b_voicing[lang] + sigma_voicing * z_voicing[part]) * voicing[n]
            + b_syllables
            + b_voicing_syllables
            + sigma_syllables_en * z_syllables_part * syllables[n]
            + sigma_voicing_syllables_en * z_voicing_syllables_en_part * voicing[n] * syllables[n];
  }
}

model {
  // Priors
  intercept ~ normal(0, 10);
  b_voicing ~ normal(0, 5);
  b_syllables_en ~ normal(0, 5);
  b_voicing_syllables_en ~ normal(0, 5);
  
  z_participant ~ normal(0, 1);
  z_voicing ~ normal(0, 1);
  
  z_syllables_en ~ normal(0, 1);
  z_voicing_syllables_en ~ normal(0, 1);
  
  sigma_participant ~ cauchy(0, 2);
  sigma_voicing ~ cauchy(0, 2);
  sigma_syllables_en ~ cauchy(0, 2);
  sigma_voicing_syllables_en ~ cauchy(0, 2);
  sigma ~ cauchy(0, 2);
  
  // Likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[P] r_participant;
  r_participant = sigma_participant * z_participant;
  vector[P] r_voicing;
  r_voicing = sigma_voicing * z_voicing;
  vector[P] r_syllables_en;
  r_syllables_en = sigma_syllables_en * z_syllables_en;
  vector[P] r_voicing_syllables_en;
  r_voicing_syllables_en = sigma_voicing_syllables_en * z_voicing_syllables_en;
}
