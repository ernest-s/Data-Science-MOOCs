function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% ====================== Ernest Code Begins ===================

prediction = X*theta;
squared_errors = (prediction - y).^2;
sum_of_squared_errors = sum(squared_errors);
% cost function
J = 1/(2*m)*sum_of_squared_errors;

% J = (1/(2*m))*sum((X*theta - y).^2);


% ====================== Ernest Code Ends =====================



% =========================================================================

end
