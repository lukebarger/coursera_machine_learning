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

% Add ones to the X data matrix (bias)
X = [ones(m, 1) X];

% neural network hypothesis
z2 = Theta1*X';
a2 = [ones(1,m); sigmoid(z2)];   % First hidden layer activation, add bias
z3 = Theta2*a2;
h = sigmoid(z3);                  % hypothesis
     
% unregularized cost function
y = eye(num_labels)(y,:);
for ii = 1:m
  this_J = -1/m*sum((y(ii,:)*log(h(:,ii))+(1-y(ii,:))*log(1-h(:,ii))));
  J = J + this_J;
endfor
% add regularization terms (excluding the bias tersm, which are the first column of each
J = J + lambda/(2*m)*(sum(sum(Theta1(:,2:end).*Theta1(:,2:end))) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end))));

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

%D2 = zeros(num_labels,hidden_layer_size);  % Total gradient, one for the output layer
%D1 = zeros(hidden_layer_size,input_layer_size);   % and one for the hidden layer
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
%(hidden later, output later)
% Here we initialize - we will accumulate this later
for ii=1:m;
%1-4   (number correspond to the ex4.pdf walkthrough of backpropagation 
%1
% Set the input layer's values (a1) to the ii-th training example.
a1 = X(ii,:)';    % the bias was already added
% neural network hypothesis
z2 = Theta1*a1;
a2 = [1; sigmoid(z2)];   % First hidden layer activation, add bias2
z3 = Theta2*a2;
a3 = sigmoid(z3);                  % output activation, also the hypothesis

%2
% first create a logical array to turn the output y (which gives a number)
% into a logical vector that has a 1 corresponding to the correct position:
% 1 goes to [1;0;0;0;0;0;0;0;0;0];  0 goes to [0;0;0;0;0;0;0;0;0;1];
d3 = a3-y(ii,:)';      % error in output vs training example

%3
% backpropogate to see how much of the delta in the output layer is due to
% each element of the last hidden layer
d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(z2);     

%4 
%accumulation
% add the error from the output later d3
D2 = D2 + d3*a2';
D1 = D1 + d2*a1';

endfor
%5 divide by m
Theta1_grad = 1/m*D1;
Theta2_grad = 1/m*D2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Set first column to zero (bias terms aren't regularized)
Theta1_reg = Theta1;
Theta1_reg(:,1)=0;
Theta2_reg = Theta2;
Theta2_reg(:,1)=0;
% Scale by lambda/m
Theta1_reg=(lambda/m)*Theta1_reg;
Theta2_reg=(lambda/m)*Theta2_reg;

% create the regularized gradient matrices
Theta1_grad=Theta1_grad+Theta1_reg;
Theta2_grad=Theta2_grad+Theta2_reg;















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
