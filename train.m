function theta = train(patchsize, hiddenSize, numpatches, sparsityParam, lambda , beta)


%% main application
fprintf('\n\n\n------------------------------------------------\n')
% load images from disk
load IMAGES;    

visibleSize = patchsize * patchsize;

sparsityParam = 0.01;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term      


%startTime = cputime;
% DO WORK
%fprintf('sampleIMAGES TIME: %g\n',cputime-startTime);

patches = sampleImages(IMAGES, patchsize, numpatches);

gpatches = gpuArray(patches);

theta = initializeParameters(hiddenSize, visibleSize);

[opttheta, cost] = ...
    trainCase(theta, gpatches, visibleSize, hiddenSize, ...
        lambda, sparsityParam, beta);

                          
%%======================================================================
%% STEP 5: Visualization 
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 



name = strcat(   int2str(visibleSize) ,'-' , int2str(hiddenSize) , '-' , int2str(visibleSize) ,  ...
        '-N' , int2str(numpatches) , '-S' , sprintf('%f', sparsityParam) , '-L' , sprintf('%f', lambda) , '-B' , sprintf('%f',beta));

mkdir(  strcat('Results\',name));
    
path =  strcat('Results\',name,'\weights.jpg');

print('-djpeg',path);   % save the visualization to a file 

thetaPath =  strcat('Results\',name,'\theta.txt');
dataSizePath =  strcat('Results\',name,'\dataSize.txt');

dlmwrite(thetaPath,opttheta);
dlmwrite(dataSizePath,[visibleSize,hiddenSize,cost]);

