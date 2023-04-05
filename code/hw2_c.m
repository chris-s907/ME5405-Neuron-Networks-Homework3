clc 
clear
set(0,'defaultfigurecolor','w')
%%
load MNIST_database.mat
% matric number: A0263252L
% find the training and testing of classes 5, 2 
trainIdx = find(train_classlabel==5 | train_classlabel==2); 
Train_ClassLabel = train_classlabel(trainIdx); 
Train_Data = train_data(:,trainIdx);
for i = 1:length(trainIdx)
    if Train_ClassLabel(i) == 5
        Train_ClassLabel(i) = 0;
    else
        Train_ClassLabel(i) = 1;
    end
end

testIdx = find(test_classlabel==5 | test_classlabel==2);
Test_ClassLabel = test_classlabel(testIdx); 
Test_Data = test_data(:,testIdx);
for i = 1:length(testIdx)
    if Test_ClassLabel(i) == 5
        Test_ClassLabel(i) = 0;
    else
        Test_ClassLabel(i) = 1;
    end
end

%% K-means
center = rand(size(Train_Data,1),2);
cluster_label = zeros(1,length(Train_ClassLabel));
time = 0;
while true
    time = time + 1;
    %assigenment
    distance = zeros(1,2);
    pre_cluster = cluster_label;
    for i = 1:length(Train_ClassLabel)
        for j = 1:2
            diff = Train_Data(:,i) - center(:,j);
            distance(j) = sum(diff.^2);
        end
        [min_value, min_index] = min(distance);
        cluster_label(i) = min_index;
    end
    
    if isequal(pre_cluster,cluster_label)
        break;
    end
    
    %update
    for i = 1:2
        cl_index = find(cluster_label == i);
        center(:,i) = mean(Train_Data(:,cl_index),2);
    end
end

%% RBFN
sigma = 10;
for i = 1:length(trainIdx)
    for j = 1:2
        diff = Train_Data(:,i) - center(:,j);
        fai_train(i,j) = exp(-1/(2*sigma^2)* sum(diff.^2));
    end
end
w = pinv(fai_train)*Train_ClassLabel';
TrPred = fai_train * w;
TrPred = TrPred';

for i = 1:length(testIdx)
    for j = 1: 2
        diff = Test_Data(:,i) - center(:,j);
        fai_test(i,j) = exp(-1/(2*sigma^2)* sum(diff.^2));
    end
end
TePred = fai_test * w;
TePred = TePred';

% evaluation
TrAcc = zeros(1,1000); 
TeAcc = zeros(1,1000); 
thr = zeros(1,1000);
TrLabel = Train_ClassLabel;
TeLabel = Test_ClassLabel;
TrN = length(TrLabel); 
TeN = length(TeLabel); 
for i = 1:1000 
    t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred); 
    thr(i) = t; 
    TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN; 
    TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN; 
end 
figure
plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');
legend('tr','te');
xlabel('Threshold');
ylabel('Accuracy');
fprintf('accuracy_train = %f accuracy_test = %f\n', max(TrAcc), max(TeAcc));

%% center visualization
for i = 1:2
    figure
    tmp = reshape(center(:,i),28,28);
    imshow(double(tmp))
end

%% mean of training images visualization
tr_index0 = find(Train_ClassLabel == 0 );
tr_mean0 = mean(Train_Data(:,tr_index0),2);
tr_0 = reshape(tr_mean0,28,28);
figure
imshow(double(tr_0))

tr_index1 = find(Train_ClassLabel == 1 );
tr_mean1 = mean(Train_Data(:,tr_index1),2);
tr_1 = reshape(tr_mean1,28,28);
figure
imshow(double(tr_1))




