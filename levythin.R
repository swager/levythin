library(glmnet)

# Run cross-validated ridge regression on pseudo-examples generated
# via Levy thinning.
#
# Parameters:
#   X    the original features (n * p matrix)
#   Y    the original labels (length n binary vector)
#   x.distr the generative distribution used to make pseudo-examples
#   reps number of pseudo-examples to make using each original example
#   nfold number of cross-validation folds
#
# Return:
#   coef.calib a learned paremeter vector for linear classification, with intercept

levythin = function(X, Y, alpha, x.distr=c("gauss", "poisson"),
                    reps=floor(max(20, 4/alpha)), nfold=10) {

    if (alpha >= 1 | alpha <= 0) {
        stop("The thinning parameter alpha must be between 0 and 1.")
    }
    x.distr = match.arg(x.distr)
    
    # create n * reps pseudo-examples of the form (\tilde{X}, Y)
    if(x.distr == "gauss") {
        tX = Reduce(rbind, lapply(1:reps, function(rr)
            alpha * X + sqrt(alpha * (1 - alpha)) * matrix(rnorm(nrow(X) * ncol(X)), nrow(X), ncol(X))
        ))
    } else if (x.distr == "poisson") {
        tX = Reduce(rbind, lapply(1:reps, function(rr)
            matrix(rbinom(length(X), as.numeric(X), alpha), nrow(X), ncol(X))
        ))
    }
    tY = rep(Y, reps)
    
    # run cross-validated ridge regression on the pseudo-example, while ensuring that all
    # pseudo-examples generated using the same original example are in the same CV fold
    foldid.single = sample(c(rep(1:nfold, ceiling(n/nfold) - 1), 1 + (0:((n - 1) %% nfold))))
    foldid = rep(foldid.single, reps)
    noised.fit = cv.glmnet(tX, tY, family = "binomial", alpha = 0, foldid=foldid)
    coef.raw = coef(noised.fit)
    
    # calibrate the predictions to the original data
    raw.pred = X %*% coef.raw[-1]
    calib.intercept = coef(glm(Y ~ raw.pred, family = binomial()))
    coef.calib = c(calib.intercept[1], calib.intercept[2] * coef.raw[-1])
    coef.calib
}