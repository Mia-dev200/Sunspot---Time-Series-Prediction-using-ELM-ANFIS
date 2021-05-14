% Load the dataset
solar_data = solar_dataset;

% Define parameters for preprocessing
data = cell2mat(solar_data(1:end));
window = 12;
train_size = 0.8;

% Prepare the data for training
[X_train, y_train, X_test, y_test] = prepare_data(data, window, train_size);

% Define parameters for passing into ELM function
NumberofHiddenNeurons = 100000;  % Try increasing number of hidden neurons
ActivationFunction = 'hardlim'; % Try changing activation to 'sin' or 'sig'
No_of_Output = 1;

% Train the time series data using ELM
[parameters] = elm_MultiOutputRegression_train([X_train y_train], No_of_Output, NumberofHiddenNeurons, ActivationFunction);

% Test the model on train and test set
[train_output] = elm_MultiOutputRegression_test(X_train, parameters);
[test_output] = elm_MultiOutputRegression_test(X_test, parameters);

test_size = size(X_test,1);
data_size = size(train_output,1) + size(test_output,1);

% Plot the test predictions
figure(1);
plot(1:test_size, test_output, 'b');
hold on;
plot(1:test_size, y_test,'r');
legend('Prediction', 'Actual', 'Location', 'best');
title('ANFIS Test Prediction Graph');
xlabel('No. of epochs');
ylabel('No. of Sunspots');

% Plot the data predictions
figure(2);
plot(1:data_size, [train_output' test_output'], 'b');
hold on;
plot(1:data_size, transpose(data(1, 1+window:data_size+window)), 'r')
legend('Prediction', 'Actual', 'Location', 'best');
title('ANFIS Data Prediction Graph');
xlabel('No. of epochs');
ylabel('No. of Sunspots');