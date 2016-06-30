function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% ==================== Ernest Code Begins here ============================

C_values        = [0.01 0.03 0.1 0.3 1 33 10 30];
sigma_values    = [0.01 0.03 0.1 0.3 1 33 10 30];
error_term      = zeros(size(C_values,2),size(sigma_values,2));
for i = 1:size(C_values,2);
    for j = 1:size(sigma_values,2);
        model= svmTrain(X,y,C_values(i),@(x1, x2)...
            gaussianKernel(x1,x2,sigma_values(j)));
        predictions = svmPredict(model,Xval);
        error_term(i,j) = mean(double(predictions ~= yval));
    end
end
[minValue, minIndex]    = min(error_term(:));
[row_no, col_no]        = find(error_term==minValue);
C       = C_values(row_no);
sigma   = sigma_values(col_no);


        
% ==================== Ernest Code Ends here ==============================

% =========================================================================

end
