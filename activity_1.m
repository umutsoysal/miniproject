%% Miniproject for JustinRuths
% Written by Umut Soysal 

% initialization of workspace
clc
clear
close all

%% DATASET
Midterm=[92 55 100 88 61 75];
Final=[95 70 95 85 75 80];
%% Part 2 Plotting Raw Data
X=Midterm';
y=Final';
m=length(X);   % number of training data points.
n=size(X,2)+1; % number of variables in the linear regression
X_raw=X;
y_raw=y;
figure
stem(X,'filled','linewidth',2)
grid on
hold on 
stem (y,'filled','linewidth',2)
legend('Midterm','Final')
hold off
title('Results of the students in Midterm and Final exams')
xlabel('Student number')
ylabel('Exam Results')
%% Feature Scaling
[X,mu_x,sigma_x]=featureScaling(X);
[y,mu_y,sigma_y]=featureScaling(y);
X=[ones(m,1) X]; % Add a column of ones to x
X_raw=[ones(m,1) X_raw]; % Add a column of ones to x
theta=zeros(2,1); % initialize fitting parameters
iterations=2000; % Maximum iteration number. 
%
alpha=0.01; % step size for the algorithm, it may be constant or 
% can be calculated through each step. In that case we use a constant
% single step.
tol=0.0001;  % Tolerance for convergence
%% Gradient Descent
[theta,J_history,theta_history,iter]=gradientdescent(X,y,theta,alpha,iterations,tol);

%% Print theta to screen
fprintf('Theta parameters found by gradient descent algorithm: ');
fprintf('%f %f \n', theta(1), theta(2)')

%% Rescaling of the theta_parameters
theta_raw(1)=mu_y-(theta(2)*mu_x*sigma_y)/sigma_x + theta(1);
theta_raw(2)=(theta(2)*sigma_y)/sigma_x;
theta_raw=theta_raw';

%% Plot of the linear fit
figure
plot(X(:,2),y,'+','linewidth',2)
hold on
grid on
plot(X(:,2),X*theta,'-','linewidth',2)
title('Linear Regression for Normalized Data')
legend('Normalized Training Data','Linear regression')
hold off
%% Plot of Cost Function
figure
plot(J_history)
title ('Error function for Normalized Data')

%% Plot of the linear fit with Rescaling
figure
plot(X_raw(:,2),y_raw,'+','linewidth',2)
grid on
hold on
plot(X_raw(:,2),X_raw*theta_raw,'-','linewidth',2)
title ('Raw data vs Linear Fit')
legend('Training Data','Linear regression')
hold off

%% Print theta_raw to screen
fprintf('Theta parameters found by gradient descent algorithm for original Dataset: ');
fprintf('%f %f \n', theta_raw(1), theta_raw(2)')
fprintf('The equation of the line is: %fx + %f \n' , theta_raw(2), theta_raw(1)')