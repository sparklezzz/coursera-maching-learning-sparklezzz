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


% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

% Calculate unregularized J

% expand_X: 5000 * 401
expand_X = [ones(m,1) X];

z2 = expand_X * Theta1';

% hidden_layer: 5000 * 25
hidden_layer = sigmoid(z2);

% expand_hidden_layer: 5000 * 26
expand_hidden_layer = [ones(m,1) hidden_layer];

z3 = expand_hidden_layer * Theta2';

% output_layer: 5000 * 10
output_layer = sigmoid(z3);

% recode the output label to 0-1 vector(only one element is 1 in a row vector)
y_recode = zeros(m, num_labels);

for i = 1:m
    y_recode(i, uint32(y(i))) = 1;
endfor

for i = 1:m
    %fprintf("y_recode: %f\n", size(y_recode(i,:)));
    %fprintf("output_layer: %f\n", size(output_layer(i,:)'));
    tempSum = sum(- y_recode(i,:) .* log(output_layer(i,:)) - ...
              (1 - y_recode(i,:)) .* log(1 - output_layer(i,:)));
    J += tempSum;
endfor

J = (1/m) * J;

%fprintf("Unregularized J: %f\n", J);

% Calculate regularized J

regularization = 0;
Theta1_no_bias = Theta1;
Theta1_no_bias(:,1) = zeros(hidden_layer_size, 1);

Theta2_no_bias = Theta2;
Theta2_no_bias(:,1) = zeros(num_labels, 1);

regularization += sum(Theta1_no_bias(:) .^ 2) + sum(Theta2_no_bias(:) .^ 2);

J += (lambda / (2 * m)) * regularization;

%fprintf("Regularized J: %f\n", J);

% Calulate gradient

% accum_grad1: 25 * 401
accum_grad1 = zeros(hidden_layer_size, input_layer_size+1);
% accum_grad2: 10 * 26
accum_grad2 = zeros(num_labels, hidden_layer_size+1);
for i = 1:m
    % delta3: 1 * 10
    delta3 = output_layer(i,:) - y_recode(i,:); 
    % Theta2: 10 * 26, z2: 5000 * 25
    % delta2: 1 * 25
    delta2 = (delta3 * Theta2)(2:end) .* sigmoidGradient(z2(i,:));

    accum_grad1 += delta2' * expand_X(i,:);
    accum_grad2 += delta3' * expand_hidden_layer(i,:);

endfor

accum_grad1 = (1/m) * accum_grad1;
accum_grad2 = (1/m) * accum_grad2;

Theta1_grad += accum_grad1;
Theta2_grad += accum_grad2;

% regularized grad

reg_part_grad1 = (lambda / m) * Theta1;
reg_part_grad1(:,1) = zeros(hidden_layer_size,1);
reg_part_grad2 = (lambda / m) * Theta2;
reg_part_grad2(:,1) = zeros(num_labels,1);

Theta1_grad += reg_part_grad1;
Theta2_grad += reg_part_grad2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
