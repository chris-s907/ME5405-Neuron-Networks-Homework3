clc 
clear
set(0,'defaultfigurecolor','w')

load MNIST_database.mat
% matric number: A0263252L
% ignore classes 5, 2 
% process train set
trainIdx = find(train_classlabel~=5 & train_classlabel~=2);
Train_Data = train_data(:, trainIdx);
Train_ClassLabel = train_classlabel(trainIdx);
% process test set
testIdx = find(test_classlabel~=5 & test_classlabel~=2);
Test_Data = test_data(:, testIdx);
Test_ClassLabel = test_classlabel(testIdx);
train_num = size(Train_Data, 2);
test_num = size(Test_Data, 2);
% SOM parameter
iter = 0;
N = 5000;
record = [1, 10, 20, 50, 100:100:N]; 
r = 1;
N_weights = 100;
weights = rand(784, N_weights);
sigma0 = N_weights / 2;
tau = N / log(sigma0);
TeAcc = zeros(1, size(record, 2)); TrAcc = zeros(1, size(record, 2));
while iter <= N
    chosen = randi(train_num);
    now = Train_Data(:, chosen);
    % find the index of minimum
    distance = zeros(1,100);
    for j = 1:100
        diff = weights(:,j) - now;
        distance(j) = sum(diff.^2);  
    end
    [val, idx] = min(distance);
    sigma = 2 * (sigma0 * exp(-iter / tau)) ^ 2 ;
    learning_rate = 0.1 * exp(iter / N);
    for i = 1 : N_weights
        d = (fix((i - 1) / 10) - fix((idx - 1) / 10)) ^ 2 + (mod(i - 1, 10) - mod(idx - 1, 10)) ^ 2;
        h = exp(-d / sigma);
        weights(:, i) = weights(:, i) + learning_rate * h * (now - weights(:, i));
    end
    if iter == record(r)
        vote = zeros(10, N_weights);
        for i = 1 : train_num
            now = Train_Data(:, i);
            distance = zeros(1,100);
            for j = 1:100
                diff = weights(:,j) - now;
                distance(j) = sum(diff.^2);  
            end
            [val, idx] = min(distance);
            vote(Train_ClassLabel(i) + 1, idx) = vote(Train_ClassLabel(i) + 1, idx) + 1;
        end
        % calculate each neuron's label
        neurons_label = zeros(1, N_weights);
        neurons_val = zeros(1, N_weights);
        for i = 1 : N_weights
            [val, idx] = max(vote(:, i));
            neurons_label(i) = idx - 1;
            neurons_val(i) = val;
        end
        % calculate test accuracy
        for i = 1 : test_num
            now = Test_Data(:, i);
            for j = 1:100
                diff = weights(:,j) - now;
                distance(j) = sum(diff.^2);  
            end
            [val, idx] = min(distance);
            TeAcc(r) = TeAcc(r) + (neurons_label(idx) == Test_ClassLabel(i));
        end
        for i = 1 : train_num
            now = Train_Data(:, i);
             for j = 1:100
                diff = weights(:,j) - now;
                distance(j) = sum(diff.^2);  
            end
            [val, idx] = min(distance);
            TrAcc(r) = TrAcc(r) + (neurons_label(idx) == Train_ClassLabel(i));
        end
        TeAcc(r) = TeAcc(r) / test_num;
        TrAcc(r) = TrAcc(r) / train_num;
        r = r + 1;
    end
    iter = iter + 1;
end

% visualize weights
trained_weights = [];
for i = 0 : 9
    weights_row = [];
    for j = 1 : 10
        weights_row = [weights_row, reshape(weights(:, i*10+j), 28, 28)];
    end
    trained_weights = [trained_weights; weights_row];
end
figure
imshow(imresize(trained_weights, 4))

% show the conceptual map
neurons_label = reshape(neurons_label, 10, 10)';
figure
img = imshow(neurons_label);

for i = [0, 1, 3, 4, 6, 7, 8 ,9]
    neurons_label(neurons_label == i) = num2str(i);
end
label = num2str(neurons_label, '%s');       
[x, y] = meshgrid(1:10);  
hStrings = text(x(:), y(:), label(:), 'HorizontalAlignment', 'center');


