clc 
clear
set(0,'defaultfigurecolor','w')
%% Generate training and testing data
train_x = -1:0.05:1;
num_train = length(train_x);
train_y = 1.2*sin(pi*train_x) - cos(2.4*pi*train_x) + 0.3*rand(1,num_train);
test_x = -1:0.01:1;
test_y = 1.2*sin(pi*test_x) - cos(2.4*pi*test_x);


%% a)RBFN
% sigma = 0.1;
% for i = 1:41
%     for j = 1:41
%         fai_train(i,j) = exp(-1/(2*sigma^2)*(train_x(i)-train_x(j))^2);
%     end
% end
% 
% w = inv(fai_train)*train_y';
% 
% for i = 1:201
%     for j = 1:41
%         fai_test(i,j) = exp(-1/(2*sigma^2)*(test_x(i)-train_x(j))^2);
%     end
% end
% 
% pred_y = fai_test*w;

%% b)Random select 15 points
idx = randperm(41);
for i = 1:15
    centers(i) = train_x(idx(i));
end
sigma = (max(centers)-min(centers))/sqrt(2*15);
for i = 1:41
    for j = 1:15
        fai_train(i,j) = exp((-1/(2*sigma^2))*(train_x(i)-centers(j))^2);
    end
end
w = pinv(fai_train)*train_y';

for i = 1:201
    for j = 1:15
        fai_test(i,j) = exp(-1/(2*sigma^2)*(test_x(i)-centers(j))^2);
    end
end

pred_y = fai_test*w;


%% c)regularization methods
% sigma = 0.1;
% for lamda = [0 0.1 0.3 0.5 2.5 5]
%     for i = 1:41
%         for j = 1:41
%             fai_train(i,j) = exp(-1/(2*sigma^2)*(train_x(i)-train_x(j))^2);
%         end
%     end
% 
%     w = inv(fai_train'*fai_train + lamda * eye(41))*fai_train'*train_y';
% 
%     for i = 1:201
%         for j = 1:41
%             fai_test(i,j) = exp(-1/(2*sigma^2)*(test_x(i)-train_x(j))^2);
%         end
%     end
%     pred_y = fai_test*w;
%     
%     figure
%     plot(train_x, train_y, 'r*','color',[238 121 66]/255);
%     hold on
%     plot(test_x, pred_y','color',[107 142 35]/255);
%     hold on 
%     plot(test_x, test_y,'color',[24 116 205]/255);
%     legend('train sample','test result','ideal function')
%    
%     error = abs(pred_y'-test_y);
%     error = mean(error)
% end


%% plot a) b)
plot(train_x, train_y, 'r*','color',[238 121 66]/255);
hold on
plot(test_x, pred_y','color',[107 142 35]/255);
hold on 
plot(test_x, test_y,'color',[24 116 205]/255);
legend('train sample','test result','ideal function')

error = abs(pred_y'-test_y);
error = mean(error)



