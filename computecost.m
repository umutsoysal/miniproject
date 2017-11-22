function J=computecost(X,Y,theta)
% This function computes the cost function (or error function)
% It is root-mean-squared error
m=length(Y);
J=0;
predictions=X*theta; 
sqrErrors=(predictions-Y).^2;
J=1/(2*m)*sum(sqrErrors);

end
