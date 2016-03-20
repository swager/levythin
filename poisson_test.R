rm(list = ls())
source("levythin.R")

n.test <- 20000
p <- 500
s0 <- 7
s1 <- 7
lam <- 1000
alphas <- seq(0.05, 0.95, by = 0.05)
alpha.all <- c(0, alphas, 1)

n <- 100

#
# Generate data
#

theta <- matrix(0,n,p,
                dimnames=
                list(rep(c("c0","c1"),c(n/2,n/2)),
                     rep(c("c0","c1","noise"),c(s0,s1,p-s0-s1))))
rnm <- rownames(theta)
cnm <- colnames(theta)

theta[which(rnm=="c0"),which(cnm=="c0")] <- 1
theta[which(rnm=="c1"),which(cnm=="c1")] <- 3*rexp(n/2)
prob <- exp(theta); prob <- prob/rowSums(prob)
X <- t(apply(prob,1,function(x) rpois(p,x*lam)))
Y <- rep(0:1,each=n/2)

theta.test <- matrix(0,n.test,p,
                     dimnames=
                     list(rep(c("c0","c1"),c(n.test/2,n.test/2)),
                          rep(c("c0","c1","noise"),c(s0,s1,p-s0-s1))))
rnm.test <- rownames(theta.test)
cnm.test <- colnames(theta.test)

theta.test[which(rnm.test=="c0"),which(cnm.test=="c0")] <- 1
theta.test[which(rnm.test=="c1"),which(cnm.test=="c1")] <- 3*rexp(n.test/2)

prob.test <- exp(theta.test); prob.test <- prob.test/rowSums(prob.test)
X.test <- t(apply(prob.test,1,function(x) rpois(p,x*lam)))
Y.test <- rep(0:1,c(n.test/2,n.test/2))

#
# Fit classifiers
#

# ridge (i.e., alpha = 1)
ridge = cv.glmnet(X, Y, family = "binomial", alpha = 0, intercept = TRUE)
ridge.coef = coef(ridge)[-1]
ridge.raw = X %*% ridge.coef
ridge.calib = glm(Y ~ ridge.raw, family = binomial())
ridge.coef.calib = c(coef(ridge.calib)[1], coef(ridge.calib)[2] * ridge.coef)
ridge.hat = as.numeric(cbind(1, X.test) %*% ridge.coef.calib > 0)
accuracy.ridge = mean(ridge.hat == Y.test)

# naive Bayes (i.e., alpha = 0)
nb.beta <- log(colSums(X[which(rnm=="c1"),])/colSums(X[which(rnm=="c0"),]))
nb.train.pred <- X %*% nb.beta
calib.coef <- coef(glm(Y ~ nb.train.pred, family=binomial))
nb.hat <- as.numeric((X.test %*% nb.beta * calib.coef[2] + calib.coef[1]) > 0)
accuracy.nb <- mean(nb.hat == Y.test)

# Levy thinning (i.e., 0 < alpha < 1)
coef.alpha = sapply(alphas, function(alpha)levythin(X, Y, alpha, x.distr="poisson"))
Y.margin = matrix(cbind(1, X.test) %*% coef.alpha)
Y.hat = matrix(as.numeric(Y.margin > 0), n.test, length(alphas))
Y.correct = (Y.hat == Y.test)
accuracy = colMeans(Y.correct)

#
# Results
#

accuracy.all = c(accuracy.nb, accuracy, accuracy.ridge)
print(rbind(alpha.all, accuracy.all))
