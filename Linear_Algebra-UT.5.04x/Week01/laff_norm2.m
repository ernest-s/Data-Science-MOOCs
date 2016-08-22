function [ alpha ] = laff_norm2( x )
%NORM OF A VECTOR
%   Function to find the length(norm 2) of a vector x

% check whether x is a vector
if ~isvector(x)
    alpha = 'FAILED';
    disp 'Input should be a vector'
    return
end

% norm of x
alpha = sqrt(laff_dot(x,x));
return

end