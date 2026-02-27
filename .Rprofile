set_up <- function() {
  if (!requireNamespace("pak", quietly = TRUE)) install.packages("pak")
  if (!requireNamespace("renv", quietly = TRUE)) pak::pak("renv")
  
  if (!requireNamespace("cmdstanr", quietly = TRUE)) pak::pak("stan-dev/cmdstanr")
  if (!requireNamespace("coretta2018itapol", quietly = TRUE)) pak::pak("stefanocoretta/coretta2018itapol")
  if (!requireNamespace("coretta2019eng", quietly = TRUE)) pak::pak("stefanocoretta/coretta2019eng")
  
  deps <- unique(renv::dependencies()[,2])
  pak::pak(deps)
  
  cmdstanr::check_cmdstan_toolchain()
}
