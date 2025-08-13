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
    
    // Random intercept for participant
    real re_intercept_z = z_participant[part];
    
    // Random voicing slope for participant (same across languages)
    real re_voicing_z = z_voicing[part];
    
    // Random syllables and interaction only for English (lang==1)
    real re_syllables_z = 0;
    real re_voicing_syllables_z = 0;
    if (lang == 1) {
      re_syllables_z = z_syllables_en[part];
      re_voicing_syllables_z = z_voicing_syllables_en[part];
    }
    
    // Fixed effects syllables and interaction only for English
    real fe_syllables = 0;
    real fe_interaction = 0;
    if (lang == 1) {
      fe_syllables = b_syllables_en * syllables[n];
      fe_interaction = b_voicing_syllables_en * voicing[n] * syllables[n];
    }
    
    mu[n] = intercept[lang]
            + sigma_participant * re_intercept_z
            + (b_voicing[lang] + sigma_voicing * re_voicing_z) * voicing[n]
            + fe_syllables
            + fe_interaction
            + sigma_syllables_en * re_syllables_z * syllables[n]
            + sigma_voicing_syllables_en * re_voicing_syllables_z * voicing[n] * syllables[n];
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
  vector[P] r_intercept;
  r_intercept = sigma_participant * z_participant;
}
