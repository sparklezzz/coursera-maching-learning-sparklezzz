function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%for i = 1:m
    %calculate h(x)
%    theta_x_inner = 0;
%    for k = 1:grad
%        theta_x_inner += theta(k) * X(i,k)
%    endfor
    
%    hx = sigmoid(theta_x_inner)
    
%    tempVal = - y(i) * log(hx) - (1 - y(i)) * log(1 - hx);
    
%    J += tempVal;
    
%    grad += (hx - y(i)) * (X(i, :))' ;
    
%endfor

%J /= m;
%grad /= m;

% X * theta: m * n, n * 1
hx_vec = sigmoid(X * theta);

J = sum(- y .* log(hx_vec) - (1 - y) .* log(ones(m,1) - hx_vec)) / m;

% grad: n * 1, hx_vec - y: m * 1,X: m * n
grad = (X' * (hx_vec - y)) / m;


%fprintf("J: %f\n", J);
%fprintf("grad: %f\n", grad);

% =============================================================

end
