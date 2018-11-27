function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

samples = 9;
C_ii = logspace(-2,2,samples);
sigma_jj = logspace(-1,1,samples);
err = zeros(size(samples,samples));

for ii=1:samples
  C = C_ii(ii);
  for jj=1:samples
    sigma = sigma_jj(jj);
    
    % create the model with this C and sigma
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    % get predictions
    predictions = svmPredict(model, Xval);
    % calculate the error of this model
    pred_error = mean(double(predictions ~= yval))
    % store the model error in an array to search for minimum
    err(ii,jj) = pred_error;
  endfor
endfor

% find the min error (more importantly the index) 
err
[x xi] = min(err(:));
% recover the matrix indicies
min_ii = mod(xi-1,samples)+1
min_jj = floor((xi-1)/samples)+1

% assign C and sigma to the values that gave the minimum error
C = C_ii(min_ii)
sigma = sigma_jj(min_jj)


% =========================================================================

end
