rm(list = ls())
source("levythin.R")

CC = 5
k = 10
p = 100
p.active = 20
n = 100
n.test = 5000

alphas = seq(0.05, 0.95, by=0.05)
alpha.all = c(0, alphas, 1)

# generate the theta distribution
Y.map = c(rep(0, k), rep(1, k))
centers.0 = CC / sqrt(p.active) * cbind(matrix(rt(k * p.active, df = 4), k, p.active), matrix(0, k, p - p.active))
centers = rbind(centers.0, -centers.0)

# create training data
cluster = sample.int(2*k, n, replace = TRUE)
Y = Y.map[cluster]
X = centers[cluster,] + matrix(rnorm(n * p), n, p)

# create test data
cluster.test = sample.int(2*k, n.test, replace = TRUE)
Y.test = Y.map[cluster.test]
X.test = centers[cluster.test,] + matrix(rnorm(n.test * p), n.test, p)

# run levy thinning
coef.alpha = sapply(alphas, function(alpha) levythin(X, Y, alpha, x.distr="gauss"))
Y.margin = matrix(cbind(1, X.test) %*% coef.alpha)
Y.hat = matrix(as.numeric(Y.margin > 0), n.test, length(alphas))
Y.correct = Y.hat == Y.test
accuracy = colMeans(Y.correct)

# run ridge regression (i.e., Levy thinning with alpha = 1)
ridge = cv.glmnet(X, Y, family = "binomial", alpha = 0, intercept = TRUE)
ridge.coef = coef(ridge)[-1]
ridge.raw = X %*% ridge.coef
ridge.calib = glm(Y ~ ridge.raw, family = binomial())
ridge.coef.calib = c(coef(ridge.calib)[1], coef(ridge.calib)[2] * ridge.coef)
ridge.hat = as.numeric(cbind(1, X.test) %*% ridge.coef.calib > 0)
accuracy.ridge = mean(ridge.hat == Y.test)

# run naive Bayes (i.e., Levy thinning with alpha = 0)
coef.naive = colMeans(X[Y==1,]) - colMeans(X[Y==0,])
pred.naive = X %*% coef.naive
calib.naive = glm(Y ~ pred.naive, family = binomial())
pred.naive.calib = coef(calib.naive)[1] + coef(calib.naive)[2] * X.test %*% coef.naive
accuracy.naive = mean(as.numeric(pred.naive.calib > 0) == Y.test)

#summary
accuracy.all = c(accuracy.naive, accuracy, accuracy.ridge)
print(rbind(alpha.all, accuracy.all))
