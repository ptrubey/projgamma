# remotes::install_github("lbelzile/BMAmevt") 
# rm(list = ls())
# libs <- c('BMAmevt', 'coda')
# sapply(libs, require, character.only = TRUE)
# rm(libs)

postpred_pairwise_betas <- function(path, nsim, nburn, nper){
  df <- read.csv(path)
  dfs <- df / apply(df, 1, sum) # project to simplex
  hpar <- list(mean.alpha = 0., sd.alpha = 3., mean.beta = 0., sd.beta = 3.)
  mcpar <- list(sd = 0.1)
  # sink('/dev/null')
  model <- BMAmevt::posteriorMCMC(
    Nsim = nsim,
    Nbin = nburn,
    dat = dfs,
    prior = BMAmevt::prior.pb,
    proposal = BMAmevt::proposal.pb,
    likelihood = BMAmevt::dpairbeta,
    Hpar = hpar,
    MCpar = mcpar
    )
  # sink()
  nCol <- ncol(dfs); nSamp <- nrow(model$stored.vals)
  f <- function(x){BMAmevt::rpairbeta(n = nper, dimData = ncol(dfs), par = x)}
  out <- array(t(apply(model$stored.vals, 1, f)), dim = c(nper * nSamp, nCol))
  return(out)
} 

# path <- "~/git/projgamma/simulated/sphere/data_m5_r5_i0.csv"
# path <- "./simulated/sphere/data_m5_r20_i0.csv"
# ppred <- postpred_pairwise_betas(path, 10000, 5000, 5)
 
# path <- "./simulated/sphere/data_m12_r5_i0.csv"

# d
# ppred = rpairbeta(1, dimData = ncol(dfs), par = model$stored.vals[25001,])

# leeds
