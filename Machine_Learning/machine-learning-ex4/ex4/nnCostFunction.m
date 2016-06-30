function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ================= Ernest Code Begins here ===============================

% creating new y values 
len_y = length(unique(y));

new_y = zeros(size(y,1),len_y);

for o = 1:size(y,1);
    p = y(o);
    new_y(o,p) = 1;
end


% finding the predicted values
% predicting hidden layer (A1)

% adding ones for intercept
X = [ones(m, 1) X];


% predicting probabilities
A1 = sigmoid(X*Theta1');
A1 = [ones(size(A1,1),1) A1];

% predicting output layer

p = sigmoid(A1*Theta2');


% Cost Function
J1 = 0;
for o = 1:size(new_y,2);
    new_y_a = new_y(:,o);
    new_p = p(:,o);
    J2 = (-1/m)*((new_y_a'*log(new_p))+((1.-new_y_a)'*(log(1.-new_p))));
    J1 = J1+J2;
end


% Adding regularization term

Theta1a = Theta1(:,2:size(Theta1,2));
Theta2a = Theta2(:,2:size(Theta2,2));

new_theta_for_cost = [Theta1a(:) ; Theta2a(:)];
J3 = sum(new_theta_for_cost.^2);
J3 = (J3*lambda)/(2*m);
% Calculating final cost
J = J1 + J3;


% gradient descent - back propagation
Delta_1 = 0;
Delta_2 = 0;

a_1 = X;
z_2 = a_1*Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2,1), 1) a_2];
z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);
    
delta_3 = a_3 - new_y;
z_2 = [ones(size(z_2,1),1) z_2];
delta_2 = (delta_3*Theta2).*sigmoidGradient(z_2);


delta_2 = delta_2(:,2:size(delta_2,2));

Delta_2 = Delta_2 + delta_3'*a_2;
Delta_1 = Delta_1 + delta_2'*a_1;

% Delta_2 = delta_3'*a_2;
% Delta_1 = delta_2'*a_1;

m=size(X,1);
unreg_grad1 = (1/m).*Delta_1;
unreg_grad2 = (1/m).*Delta_2;

unreg_grad1a = unreg_grad1(:,1);
unreg_grad1b = unreg_grad1 + (lambda/m).*Theta1;
unreg_grad1b = unreg_grad1b(:,2:end);

unreg_grad2a = unreg_grad2(:,1);
unreg_grad2b = unreg_grad2 + (lambda/m).*Theta2;
unreg_grad2b = unreg_grad2b(:,2:end);

Theta1_grad = [unreg_grad1a unreg_grad1b];
Theta2_grad = [unreg_grad2a unreg_grad2b];

% ================= Ernest Code Ends here =================================

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
