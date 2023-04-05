clc 
clear
set(0,'defaultfigurecolor','w')

trainX = rands(2,500); %2x500 matrix, column-wise points 
N = 500;
num_train = size(trainX,2);
weight = rand(2,25);
lr0 = 0.1;
tao = N;
sigma0 = sqrt(25+25)/2;
time_constant = N/log(sigma0);
record = [1, 20, 50, 100, 300, N];
r = 1;

%iteration
for i = 1:N
    samp = trainX(:,randi(num_train));
    distance = zeros(1,25);
    for j = 1:25
        diff = weight(:,j) - samp;
        distance(j) = sum(diff.^2);  
    end
    [min_value, min_index] = min(distance);
    sigma = sigma0 * exp(-i/time_constant);
    lr = lr0 * exp(-i/tao);
    for k = 1:25
        codi_k = zeros(1,2);
        codi_i = zeros(1,2);
        if mod(k,5) == 0
            codi_k(1) = floor(k/5);
            codi_k(2) = 5;
        else
            codi_k(1) = floor(k/5)+1;
            codi_k(2) = mod(k,5);
        end
        
        if mod(min_index,5) == 0
            codi_i(1) = floor(min_index/5);
            codi_i(2) = 5;
        else
            codi_i(1) = floor(min_index/5)+1;
            codi_i(2) = mod(min_index,5);
        end
        diff = codi_k - codi_i;
        d = sum(diff.^2);
        h = exp(-d/(2*sigma^2));
        weight(:,k) = weight(:,k) + lr*h*(samp - weight(:,k));
    end
    
    if i == record(r)
        figure
        hold on
        plot(trainX(1,:), trainX(2,:), '+r');
        for j = 1 : 5
            plot(weight(1, j*5-4:j*5), weight(2, j*5-4:j*5), '+b-');
            plot(weight(1, j:5:end), weight(2, j:5:end), '+k-');
        end
        legend('training points','trained weights');
        r = r + 1;
    end
end
