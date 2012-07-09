%% sparseAutoencoderCost


function [cost,grad] = sparseAutoencoderCostGpu(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)
                                         
import parallel.gpu.GPUArray

gtheta = gpuArray(theta);
                                         
W1 = reshape(gtheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(gtheta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = gtheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = gtheta(2*hiddenSize*visibleSize+hiddenSize+1:end);


datasize = size(data);
numpatches = datasize(2);

% Row-vector to aid in calculation of hidden activations and output values
weightsbuffer = GPUArray.ones(1, numpatches);

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

cost = gather(cost);
grad = gather([W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)]);

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end                                         