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


test_vec = [0.05 0.1 0.5 1 2 10 25]
results = zeros(size([1 2 3 4])(2), 3);

i = 0;
for C_est = test_vec
    for sigma_est = test_vec
        i = i + 1;
        model = svmTrain(X, y, C_est, @(x1, x2) gaussianKernel(x1, x2, sigma_est));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));

        results(i,:) = [C_est, sigma_est, prediction_error];     
    end
end

sorted = sortrows(results, 3);

C = sorted(1,1)
sigma = sorted(1,2)





% =========================================================================

end
