% Load the dataset
solar_data = solar_dataset;

% Define parameters for preprocessing
data = cell2mat(solar_data(1:end));
window = 12;
train_size = 0.8;

% Prepare the data for training
[X_train, y_train, X_test, y_test] = prepare_data(data, window, train_size);
train_data = [X_train y_train];
test_data = [X_test y_test];

options = genfisOptions('SubtractiveClustering');
fismat = genfis(X_train, y_train, options);

figure(1)
plotmf(fismat, 'input', 1);
title('initial membership function of anfis')

% Determine number of iterations
numepochs = 100;

out_fis = anfis(train_data, fismat, numepochs, [], test_data);

train_output = evalfis(out_fis, X_train);
test_output = evalfis(out_fis, X_test);

test_size = size(X_test,1);
data_size = size(X_train,1) + size(X_test,1);

% Plot the test predictions
figure(2);
plot(1:test_size, test_output, 'b');
hold on;
plot(1:test_size, y_test, 'r');
legend('anfis ouput', 'actual output','Location', 'best');
title('ANFIS Test Prediction Graph');
xlabel('No. of epochs');
ylabel('No. of Sunspots');

% Plot the data predictions
figure(3);
plot(1:data_size, [train_output' test_output'], 'b');
hold on;
plot(1:data_size, transpose(data(1, 1+window:data_size+window)), 'r');
legend('Prediction', 'Actual', 'Location', 'best');
title('ANFIS Data Prediction Graph');
xlabel('No. of epochs');
ylabel('No. of Sunspots');