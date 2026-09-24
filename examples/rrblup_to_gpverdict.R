# Complete example: cross-validated rrBLUP models for spring wheat, checked with GPverdict.
#
# Data: Fusarium head blight resistance (100 - disease score) of 384 lines in 109 environments of the
# Uniform Regional Scab Nursery (CC0; Brault et al. 2025, doi:10.5061/dryad.wstqjq2z0), and their
# 2,302 markers coded -1/0/1.  Run from this folder:
#   Rscript rrblup_to_gpverdict.R
# It writes predictions_long.csv; then run  gpverdict predictions_long.csv --min-genotypes 12 --out report.html
# (pip install git+https://github.com/nblvguohao/gpverdict) or upload the CSV at
# https://nblvguohao.github.io/gpverdict/.
suppressMessages(library(rrBLUP))
set.seed(1)

ph <- read.csv("spring_wheat_phenotypes.csv", stringsAsFactors = FALSE)   # environment, genotype, observed
M  <- as.matrix(read.csv("spring_wheat_markers.csv", row.names = 1, check.names = FALSE))
genos <- rownames(M)
ph <- ph[ph$genotype %in% genos, ]
K  <- A.mat(M)                                       # realised relationship matrix
D  <- as.matrix(dist(M)) / sqrt(ncol(M))             # scaled distances for the Gaussian kernel
folds <- setNames(sample(rep(1:5, length.out = length(genos))), genos)   # leave-genotypes-out, 5 folds

predict_fold <- function(train, test, method) {
  # environment means from this fold's training lines only, so no test phenotype enters a prediction
  env_mean <- tapply(train$observed, train$environment, mean)
  base <- env_mean[test$environment]; base[is.na(base)] <- mean(train$observed)
  dev  <- train$observed - env_mean[train$environment]            # within-environment deviations
  g    <- tapply(dev, train$genotype, mean)                        # line means of the deviations
  d    <- data.frame(gid = names(g), y = as.numeric(g))
  u <- switch(method,
    "GBLUP"           = kin.blup(d, geno = "gid", pheno = "y", K = K)$g,
    "Gaussian kernel" = kin.blup(d, geno = "gid", pheno = "y", K = D, GAUSS = TRUE)$g,
    "rrBLUP markers"  = { f <- mixed.solve(d$y, Z = M[d$gid, ]); setNames(as.vector(M %*% f$u), genos) })
  as.numeric(base + u[test$genotype])
}

out <- list()
for (method in c("GBLUP", "rrBLUP markers", "Gaussian kernel")) {
  pred <- rep(NA_real_, nrow(ph))
  for (f in 1:5) {
    te <- ph$genotype %in% names(folds)[folds == f]
    pred[te] <- predict_fold(ph[!te, ], ph[te, ], method)
  }
  out[[method]] <- data.frame(ph, predicted = pred, method = method)
}
long <- do.call(rbind, out)
write.csv(long, "predictions_long.csv", row.names = FALSE)
cat("wrote predictions_long.csv:", nrow(long), "rows,", length(unique(long$method)), "methods\n")
