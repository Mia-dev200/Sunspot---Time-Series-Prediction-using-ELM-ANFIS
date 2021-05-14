function [X_train, y_train, X_test, y_test] = prepare_data(data, window, train_size)
% prepare_dataset function takes in the sequence data and converts it into a format
% that can be fed into the ELM model.

% Input:
% data       - Sequence data that needs to be formatted
% window     - Number of previous data points to be considered for
%              prediction
% train_size - Proportion of data to be taken as training set

% Output
% X_train  - The training feature matrix
% y_train  - The training output vector
% X_test   - The testing feature matrix
% y_test   - The testing output vector

N = size(data,2); % Size of input dataset

for i = 1:1:N
    % Break out of the loop if target datapoint out of range
    if((i + window + 1) > N)
        break;
    end
    c = 1;
    for j = i:1:(i + window -1)
        % Append the next "window" [12] datapoints into input feature matrix
        X(i, c) = data(j);
        c = c + 1;
    end
    % Append the "(window+1)th" [13th] datapoint to target vector
    y(i, 1) = data(i + window);
end

% Split the transformed data into train and test
split = ceil(size(X, 1) * train_size);
X_train = X(1:split,:);
y_train = y(1:split,:);
X_test = X(split+1:end,:);
y_test = y(split+1:end,:);

end

