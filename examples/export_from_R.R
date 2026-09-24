# Export cross-validated predictions from R (rrBLUP, BGLR, sommer, ...) for GPverdict.
# One row per environment and genotype; one column per method's cross-validated prediction.
# Then: gpverdict predictions_long.csv   (or upload it at https://nblvguohao.github.io/gpverdict/)

library(rrBLUP)
# pheno: data.frame(environment, genotype, observed); M: marker matrix (genotypes x markers, coded -1/0/1)
# folds: fold label per genotype (leave-genotypes-out cross-validation)
cv_predict <- function(pheno, M, folds, fit_fun) {
  pred <- rep(NA_real_, nrow(pheno))
  for (f in unique(folds)) {
    test_g <- names(folds)[folds == f]
    tr <- !(pheno$genotype %in% test_g); te <- !tr
    pred[te] <- fit_fun(pheno[tr, ], pheno[te, ], M)
  }
  pred
}
rrblup_fun <- function(train, test, M) {
  # environment means as fixed effects, marker effects shared across environments
  X  <- model.matrix(~ environment, train)
  fit <- mixed.solve(train$observed, Z = M[train$genotype, ], X = X)
  env_mean <- tapply(train$observed, train$environment, mean)
  as.numeric(env_mean[test$environment] + M[test$genotype, ] %*% fit$u)
}
wide <- pheno
wide$rrBLUP <- cv_predict(pheno, M, folds, rrblup_fun)
# add further methods the same way, e.g. wide$BayesB <- cv_predict(pheno, M, folds, bglr_bayesb_fun)

# long format expected by GPverdict
long <- reshape(wide, direction = "long", varying = setdiff(names(wide), c("environment", "genotype", "observed")),
                v.names = "predicted", timevar = "method",
                times = setdiff(names(wide), c("environment", "genotype", "observed")), idvar = c("environment", "genotype"))
write.csv(long[, c("environment", "genotype", "observed", "predicted", "method")], "predictions_long.csv", row.names = FALSE)
