function [ x_out ] = laff_scal( alpha, x )

% x = scale( alpha, x ) scales the vector x by a scalar alpha
%   Vector x can be a column or row vector. alpha needs to be a scalar.  

% Extract the row and column sizes of x 
[ m_x, n_x ] = size( x );

% Make sure alpha is a scalar and x is a vector
if ~isscalar(alpha)
    x_out = 'FAILED';
    return
end

if ~isvector(x)
    x_out = 'FAILED';
    return
end

if ( n_x == 1 )     % x is a column vector
    for i=1:m_x
        x(i,1) = alpha * x(i,1);
    end    
else    % x is a row vector
    for i=1:n_x
        x(1,i) = alpha * x(1,i);
    end 
   
end

% Return the updated x in x_out
x_out = x;

return
end