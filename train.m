%% main application
fprintf('\n\n\n------------------------------------------------\n')
patchsize = 12;
numpatches = 10000;

visibleSize = patchsize * patchsize;
hiddenSize = ceil(visibleSize / 2.56);

sparsityParam = 0.01;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term      




startTime = cputime;
patches = sampleIMAGES(patchsize, numpatches);
fprintf('sampleIMAGES TIME: %g\n',cputime-startTime);


startTime = cputime;
theta = initializeParameters(hiddenSize, visibleSize);
fprintf('initializeParameters TIME: %g\n',cputime-startTime);

trainGpu

fprintf('minFunc TIME: %g\n',cputime-startTime);

theta = initializeParameters(hiddenSize, visibleSize);
fprintf('initializeParameters TIME: %g\n',cputime-startTime);

trainCpu

fprintf('minFunc TIME: %g\n',cputime-startTime);

                          
                          
%%======================================================================
%% STEP 5: Visualization 
W1 = gather(reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize));
display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 