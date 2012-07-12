function [cost,grad] = sparseAutoencoderCostCpu(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% % Cost and gradient variables (your code needs to compute these values). 
% % Here, we initialize them to zeros. 
% cost = 0;
% W1grad = zeros(size(W1)); 
% W2grad = zeros(size(W2));
% b1grad = zeros(size(b1)); 
% b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
datasize = size(data);
numpatches = datasize(2);

% Row-vector to aid in calculation of hidden activations and output values
weightsbuffer = ones(1, numpatches);

% Calculate activations of hidden and output neurons
hiddeninputs = W1 * data + b1 * weightsbuffer; % hiddensize * numpatches
hiddenvalues = sigmoid( hiddeninputs ); % hiddensize * numpatches

finalinputs = W2 * hiddenvalues + b2 * weightsbuffer; %visiblesize * numpatches
outputs = sigmoid( finalinputs ); %visiblesize * numpatches

% Least squares component of cost
errors = outputs - data; %visiblesize * numpatches
%leastsquares = power(norm(errors), 2) / (2 * numpatches); % Average least squares error over numpatches samples

leastsquares = sum( sum((errors .* errors)) ./ (2*numpatches));

% Back-propagation calculation of gradients
delta3 = errors .* outputs .* (1 - outputs); % Matrix of error terms, visiblesize * numpatches
W2grad = delta3 * transpose(hiddenvalues) / numpatches; % visiblesize * hiddensize, averaged over all patches
b2grad = delta3 * transpose(weightsbuffer) / numpatches; % visiblesize * 1, averaged over all patches

% Sparsity stuff
avgactivations = hiddenvalues * transpose(weightsbuffer) / numpatches; % hiddensize * 1
sparsityvec = -sparsityParam ./ avgactivations + (1 - sparsityParam) ./ (1 - avgactivations); % hiddensize * 1
% sparsityvec * weightsbuffer; % Add this to the delta2 parenthesis
kldiv = sparsityParam * log(prod(sparsityParam ./ avgactivations)) + (1 - sparsityParam) * log(prod( (1 - sparsityParam) ./ (1 - avgactivations) )); % Add this to cost

delta2 = (transpose(W2) * delta3 + beta * sparsityvec * weightsbuffer) .* hiddenvalues .* (1 - hiddenvalues); % hiddensize * numpatches
W1grad = delta2 * transpose(data) / numpatches; % hiddensize * visiblesize, averaged over all patches
b1grad = delta2 * transpose(weightsbuffer) / numpatches; % hiddensize * 1, averaged over all patches

cost = leastsquares + beta * kldiv;


% Weight-decay
%%cost = cost + lambda / 2 * ( power(norm(W1), 2) + power(norm(W2), 2) );

cost = cost + lambda / 2 * ( sum(sum(W1 .* W1)) +sum( sum(W2 .* W2)) );

W1grad = W1grad + lambda * W1;
W2grad = W2grad + lambda * W2;





%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

% % Sigmoid function derivative using identity from lecture notes
% function sigmD = sigmoidDerivative(x)
%   
%     sigm = sigmoid(x);
%     sigmD = sigm .* (1 - sigm);
% end

