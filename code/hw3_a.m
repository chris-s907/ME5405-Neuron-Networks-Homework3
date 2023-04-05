clc 
clear
set(0,'defaultfigurecolor','w')

% initialization
t = linspace(-pi,pi,200); 
trainX = [t.*sin(pi*sin(t)./t); 1-abs(t).*cos(pi*sin(t)./t)];  % 2x200 matrix, column-wise points 
num_train = size(trainX,2);
weight = rand(2,25);
iter = 500;
lr0 = 0.1;
tao = iter;
sigma0 = sqrt(1+25^2)/2;
time_constant = iter/log(sigma0);
record = [1, 20, 50, 100, 300, iter];
r = 1;

%iteration
for i = 1:iter
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
        d = k - min_index;
        h = exp(-d^2/(2*sigma^2));
        weight(:,k) = weight(:,k) + lr*h*(samp - weight(:,k));
    end
    
    if i == record(r)
        figure
        plot(weight(1,:), weight(2,:), '+b-');
        hold on
        plot(trainX(1,:), trainX(2,:), '+r');
        legend('trained weights','training points');
        r = r + 1;
    end
end
