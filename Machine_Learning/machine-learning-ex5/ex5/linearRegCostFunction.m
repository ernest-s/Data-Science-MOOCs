function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% cost function

% unregularized cost
J1 = (1/(2*m))*sum((X*theta - y).^2);

% regularization term
J2 = (lambda/(2*m))*(sum(theta.^2));

% removing regularization for bias term
J = J1 + J2 - ((lambda/(2*m))*theta(1)^2);

% gradient

% unregularized term
grad1 = ((X'*(X*theta -y))).*(1/m);

% regularization term
grad2 = theta.*(lambda/m);

% we don't regularize the bias term
% setting regularization for bias term to zero
grad2(1) = 0;

% calculating final gradient
grad = grad1 + grad2;

% ====================== Ernest Code Begins here ==========================















% ======================== Ernest Code Ends here ==========================


% =========================================================================

grad = grad(:);

end
