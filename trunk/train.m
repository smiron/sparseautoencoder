%% main application
fprintf('\n\n\n------------------------------------------------\n')
% load images from disk
load IMAGES;    

patchsize = 12;
numpatches = 10000;

visibleSize = patchsize * patchsize;
hiddenSize = ceil(visibleSize / 2.56);

sparsityParam = 0.01;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term      


%startTime = cputime;
% DO WORK
%fprintf('sampleIMAGES TIME: %g\n',cputime-startTime);

patches = sampleImages(IMAGES, patchsize, numpatches);

theta = initializeParameters(hiddenSize, visibleSize);

[opttheta, cost] = ...
    trainCase(theta, patches, visibleSize, hiddenSize, ...
        lambda, sparsityParam, beta);

                          
%%======================================================================
%% STEP 5: Visualization 
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 