function [X_norm,mu,sigma]=featureScaling(X)
% This function returns a normalized version of X where the mean value of
% each feature is 0 and deviation is 1. 

mu=mean(X);
sigma=std(X);

for i=1:size(X,2)
X_norm(:,i)=(X(:,i)-mu(i)) / sigma(i); 
end
