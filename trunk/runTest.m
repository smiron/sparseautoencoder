
load test.txt;

for i=1:size(test,1)
   
    train(test(i,1),test(i,2),test(i,3),test(i,4),test(i,5),test(i,6));
        
end