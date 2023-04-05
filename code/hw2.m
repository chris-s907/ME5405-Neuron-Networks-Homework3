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



%% RBFN with and without rugularization
% sigma = 100;
% for lamda = [0 0.01 0.1 1 10 100]
%     for i = 1:length(trainIdx)
%         for j = 1:length(trainIdx)
%             diff = Train_Data(:,j)-Train_Data(:,i);
%             fai_train(i,j) = exp(-1/(2*sigma^2)* sum(diff.^2));
%         end
%     end
%     w = inv(fai_train'*fai_train + lamda * eye(length(Train_ClassLabel)))* fai_train'*Train_ClassLabel';
%     TrPred = fai_train * w;
%     TrPred = TrPred';
% 
%     for i = 1:length(testIdx)
%         for j = 1: length(trainIdx)
%             diff = Test_Data(:,i)-Train_Data(:,j);
%             fai_test(i,j) = exp(-1/(2*sigma^2)* sum(diff.^2));
%         end
%     end
%     TePred = fai_test * w;
%     TePred = TePred';
%     
%     % evaluation
%     TrAcc = zeros(1,1000); 
%     TeAcc = zeros(1,1000); 
%     thr = zeros(1,1000);
%     TrLabel = Train_ClassLabel;
%     TeLabel = Test_ClassLabel;
%     TrN = length(TrLabel); 
%     TeN = length(TeLabel); 
%     for i = 1:1000 
%         t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred); 
%         thr(i) = t; 
%         TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN; 
%         TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN; 
%     end 
%     figure
%     plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');
%     legend('tr','te');
%     xlabel('Threshold');
%     ylabel('Accuracy')
%     fprintf('lamda = %g accuracy_train = %f accuracy_test = %f\n',lamda, max(TrAcc), max(TeAcc));
% end

%% select 100 centers
for sigma = [0.1 1 10 100 1000 10000]
    idx = randperm(197);
    for i = 1:100
        centers(:,i) = Train_Data(:,idx(i));
    end

    for i = 1:length(trainIdx)
        for j = 1:100
            diff = Train_Data(:,i) - centers(:,j);
            fai_train(i,j) = exp(-1/(2*sigma^2)* sum(diff.^2));
        end
    end
    w = pinv(fai_train)*Train_ClassLabel';
    TrPred = fai_train * w;
    TrPred = TrPred';

    for i = 1:length(testIdx)
        for j = 1: 100
            diff = Test_Data(:,i) - centers(:,j);
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
    ylabel('Accuracy')
    fprintf('sigma = %g accuracy_train = %f accuracy_test = %f\n', sigma, max(TrAcc), max(TeAcc));
end

