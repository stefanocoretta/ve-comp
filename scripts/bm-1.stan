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
  vector[L] alpha_lang;                     // grand intercept
  
  // Fixed effects for voicing by language (length L)
  vector[L] b_voicing;
  
  // Fixed effects for syllables and interaction only for English (index 1)
  real b_syllables_EN;
  real b_voicing_syllables_EN;
  
  // Participant random intercepts
  vector[P] r_intercept_z;
  
  // Participant random voicing slopes (pooled across languages)
  vector[P] r_voicing_z;
  
  // Participant random syllables and interaction slopes only for English (index 1)
  vector[P] r_syllables_EN_z;
  vector[P] r_voicing_syllables_EN_z;
  
  // Random effect standard deviations
  real<lower=0> sigma_intercept;
  real<lower=0> sigma_voicing;
  
  real<lower=0> sigma_syllables_EN;
  real<lower=0> sigma_voicing_syllables_EN;
  
  // Residual sd
  real<lower=0> sigma;
}

transformed parameters {
  vector[N] mu;
  
  for (n in 1:N) {
    int lang = language[n];
    int part = participant[n];
    
    // Random intercept for participant
    real re_intercept_z = r_intercept_z[part];
    
    // Random voicing slope for participant (same across languages)
    real re_voicing_z = r_voicing_z[part];
    
    // Random syllables and interaction only for English (lang==1)
    real re_syllables_z = 0;
    real re_voicing_syllables_z = 0;
    if (lang == 1) {
      re_syllables_z = r_syllables_EN_z[part];
      re_voicing_syllables_z = r_voicing_syllables_EN_z[part];
    }
    
    // Fixed effects syllables and interaction only for English
    real fe_syllables = 0;
    real fe_interaction = 0;
    if (lang == 1) {
      fe_syllables = b_syllables_EN * syllables[n];
      fe_interaction = b_voicing_syllables_EN * voicing[n] * syllables[n];
    }
    
    mu[n] = alpha_lang[lang]
            + sigma_intercept * re_intercept_z
            + (b_voicing[lang] + sigma_voicing * re_voicing_z) * voicing[n]
            + fe_syllables
            + fe_interaction
            + sigma_syllables_EN * re_syllables_z * syllables[n]
            + sigma_voicing_syllables_EN * re_voicing_syllables_z * voicing[n] * syllables[n];
  }
}

model {
  // Priors
  alpha_lang ~ normal(0, 10);
  b_voicing ~ normal(0, 5);
  b_syllables_EN ~ normal(0, 5);
  b_voicing_syllables_EN ~ normal(0, 5);
  
  r_intercept_z ~ normal(0, 1);
  r_voicing_z ~ normal(0, 1);
  
  r_syllables_EN_z ~ normal(0, 1);
  r_voicing_syllables_EN_z ~ normal(0, 1);
  
  sigma_intercept ~ cauchy(0, 2);
  sigma_voicing ~ cauchy(0, 2);
  sigma_syllables_EN ~ cauchy(0, 2);
  sigma_voicing_syllables_EN ~ cauchy(0, 2);
  sigma ~ cauchy(0, 2);
  
  // Likelihood
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[P] r_intercept;
  r_intercept = sigma_intercept * r_intercept_z;
}
