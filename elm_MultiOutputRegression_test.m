function [output] = elm_MultiOutputRegression_test(TestingData_File,parameters)


InputWeight=parameters.InputWeight;
 BiasofHiddenNeurons=parameters.BiasofHiddenNeurons;
 OutputWeight=parameters.OutputWeight;
 ActivationFunction=parameters.ActivationFunction;
 No_of_Output=parameters.No_of_Output;


%load elm_model.mat;
%%%%%%%%%%% Load testing dataset
test_data=TestingData_File;
TV.P=test_data';
%TV.P=test_data(:,1:(size(test_data,2)-No_of_Output))';
%TV.T=test_data(:,(size(test_data,2)-No_of_Output+1):size(test_data,2))';
clear test_data;  


NumberofTestingData=size(TV.P,2);




%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
ot=(H_test' * OutputWeight)'; %   TY: the actual output of the testing data
output=ot';
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

%TestingAccuracy=sqrt(mse(TV.T - ot))            %   Calculate testing accuracy (RMSE) for regression case
end