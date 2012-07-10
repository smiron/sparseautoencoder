function [opttheta, cost] = ...
    trainCase(theta, patches, visibleSize, hiddenSize, ...
        lambda, sparsityParam, beta)
%% prepare execution
import parallel.gpu.GPUArray

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 200;    % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

%%======================================================================
%% minimize
try
    [opttheta, cost] = minFunc( @(p) sparseAutoencoderCostGpu(p, ...
                                       visibleSize, hiddenSize, ...
                                       lambda, sparsityParam, ...
                                       beta, patches), ...
                                  theta, options);
                              
catch exception
    disp(exception)
end
%%======================================================================