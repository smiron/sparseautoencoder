
load test2.txt;

for i=1:size(test2,1)
   
    try
        
        train(test2(i,1),test2(i,2),test2(i,3),test2(i,4),test2(i,5),test2(i,6));
        
    catch exception
        disp(exception);
    end
    
    
        
end