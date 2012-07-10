function [opttheta, cost] = trainCase(theta, patches)

import parallel.gpu.GPUArray

gtheta = gpuArray(theta);
gpatches =  gpuArray(patches);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;    % Maximum number of iterations of L-BFGS to run 
options.display = 'off';

try
    
    [opttheta, cost] = minFunc( @(p) sparseAutoencoderCostGpu(p, ...
                                       visibleSize, hiddenSize, ...
                                       lambda, sparsityParam, ...
                                       beta, gpatches), ...
                                  gtheta, options);
                              
catch exception
    disp(exception)
end
    