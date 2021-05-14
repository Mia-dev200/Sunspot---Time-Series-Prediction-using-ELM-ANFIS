function [parameters] = elm_MultiOutputRegression_train(TrainingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction)
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% No_of_Output          - Number of outputs for regression
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression


%%%%%%%%%%% Load training dataset
train_data=TrainingData_File;
P=train_data(:,1:(size(train_data,2)-No_of_Output))';
T=train_data(:,(size(train_data,2)-No_of_Output+1):size(train_data,2))';
clear train_data;                                   %   Release raw training data array


NumberofTrainingData=size(P,2);
NumberofInputNeurons=size(P,1);

start_time_train=cputime;


%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = hardlim(tempH);            
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;       %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
TrainingAccuracy=sqrt(mse(T - Y));             %   Calculate training accuracy (RMSE) for regression case
clear H;

%save('elm_model', 'InputWeight', 'BiasofHiddenNeurons', 'OutputWeight', 'ActivationFunction','No_of_Output');  
parameters.InputWeight=InputWeight;
parameters.BiasofHiddenNeurons=BiasofHiddenNeurons;
parameters.OutputWeight=OutputWeight;
parameters.ActivationFunction=ActivationFunction;
parameters.No_of_Output=No_of_Output;