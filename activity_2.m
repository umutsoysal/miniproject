%% Miniproject for JustinRuths

% initialization of workspace
clc
clear
close all

%% DATASET
Person=[1 2 3 4 5 6 7 8 9 10 11 12 13];
Weight=[69 83 77 75 71 73 67 71 77 69 74 86 84];
Age=[50 20 20 30 30 50 60 50 40 55 40 40 20];
Stress=[55 47 33 65 47 58 46 68 70 42 33 55 48];
BP=[120 141 124 126 117 129 123 125 132 123 132 155 147];

X=[Person' Weight' Age' Stress'];
y=[BP'];


%% Part 2 Plotting DATA

m=length(X); % number of training data points.
n=size(X,2)+1; % number of variables in the linear regression
X_raw=X;
y_raw=y;

% Plot the dataset if it is a single variable dataset
if n<3 
    figure
    plot(X)
    hold on 
    plot (y)
    hold off
    title('Plot of X vs Y')
end

%% Feature Scaling
[X,mu_x,sigma_x]=featureScaling(X);
[y,mu_y,sigma_y]=featureScaling(y);
X=[ones(m,1) X]; % Add a column of ones to x
X_raw=[ones(m,1) X_raw]; % Add a column of ones to x
theta=zeros(n,1); % initialize fitting parameters
iterations=1000; 
alpha=0.01;
tol=0.0001;

%% Gradient Descent
[theta,J_history,theta_history,iter]=gradientdescent(X,y_raw,theta,alpha,iterations,tol);

%% Print theta to screen
%fprintf('Theta parameters found by gradient descent algorithm:');
%fprintf('%f %f \n', theta')

%% Rescaling of the theta_parameters
 theta_raw_1=mu_y-(theta(2)*mu_x*sigma_y)/sigma_x + theta(1);
 %theta_raw(2)=(theta(2)*sigma_y)/sigma_x;
 theta_raw_2=theta(2:end).*sigma_x'+mu_x';
 theta_raw=[theta_raw_1 theta_raw_2'];
 
%% Plot of the linear fit
figure
stem(X(:,2),y_raw,'+','linewidth',2)
hold on
stem(X(:,2),X*theta,'-','linewidth',2)
title('Real BP values vs Estimated values stem plot')
legend('Training Data','Linear regression')
hold off

figure
plot(X(:,2),y_raw,'+','linewidth',2)
hold on
plot(X(:,2),X*theta,'-','linewidth',2)
title('Real BP values vs Estimated values')
legend('Training Data','Linear regression')
grid on
hold off

%% Plot of Cost Function
figure
plot(J_history)
title ('Error function for Normalized Data')
xlabel('iteration number')
xlabel('error')
%% 
% estimated_BP=X*theta;
% real_BP=y_raw;
% error_BP=sum((estimated_BP-real_BP).^2);
% figure
% plot(estimated_BP,'r+')
% hold on
% plot(real_BP,'g+')
% grid on
% hold off

% %% Plot of the linear fit with Rescaling
% figure
% plot(X_raw(:,2),y_raw,'+')
% hold on
% plot(X_raw(:,2),X_raw*theta_raw,'-')
% title ('Raw data vs Linear Fit')
% legend('Training Data','Linear regression')
% hold off

%% Print theta to screen
fprintf('Theta parameters found by gradient descent algorithm is: ');
theta

