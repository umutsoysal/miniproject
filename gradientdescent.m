function [theta,J_history,theta_history,iter]=gradientdescent(X,Y,theta,alpha,num_iters,tol)

m=length(Y);
J_history=zeros(num_iters,1);
theta_history=zeros(num_iters,size(X,2));
for iter=1:num_iters
   
    htheta=X*theta;
    for j=1:size(X,2)
    T(j)=theta(j)-alpha/m*sum((htheta-Y).*X(:,j));
    %theta1=theta(2)-alpha/m*sum((htheta-Y).*X(:,2));
    end
    %theta=[theta0;theta1];
    theta=T';
    error=theta_history(end,:)'-theta;
    theta_history(iter,:)=theta;
    J_history(iter)=computecost(X,Y,theta);
     
    if sum(error.^2)<tol && (iter>10)
        break
    end
    
end


end