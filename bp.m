clear all;close all;clc;
% load data
% 178*14，数据分为三类，第一列为类别，以此往后13列分别为：
%  	1) Alcohol
%  	2) Malic acid
%  	3) Ash
% 	4) Alcalinity of ash
%  	5) Magnesium
% 	6) Total phenols
%  	7) Flavanoids
%  	8) Nonflavanoid phenols
%  	9) Proanthocyanins
% 	10)Color intensity
%  	11)Hue
%  	12)OD280/OD315 of diluted wines
%  	13)Proline
%  每个类别的数据数目：
%   class 1 59 (1-59)
% 	class 2 71 (60-130)
% 	class 3 48  (131-178)
load chp_wineclass.mat;
% 将第一类1-30，第二类的60-95，第三类的131-153作为训练集
train_wine = [wine(1:30,2:end);wine(60:95,2:end);wine(131:153,2:end)];
train_wine_labels = [wine(1:30,1);wine(60:95,1);wine(131:153,1)];
% 将第一类31-59，第二类的96-130，第三类的154-178作为测试集
test_wine = [wine(31:59,2:end);wine(96:130,2:end);wine(154:178,2:end)];
test_wine_labels = [wine(31:59,1);wine(96:130,1);wine(154:178,1)];

%% normalize the input data
train_std  = (std(train_wine'))';
train_std1 = repmat(train_std,1,13);
train_mean = (mean(train_wine'))';
train_mean1 = repmat(train_mean,1,13); %矩阵扩展，方便后面矩阵减法和除法
test_std = (std(test_wine'))';
test_std1 = repmat(test_std,1,13);
test_mean = (mean(test_wine'))';
test_mean1 = repmat(test_mean,1,13);%矩阵扩展，方便后面矩阵减法和除法

train_wine = (train_wine - train_mean1)./train_std1;
test_wine = (test_wine - test_mean1)./test_std1;


%% 处理标签，三类可以表示为[1 0 0],[0 1 0],[0 0 1];
[t,r] = size(train_wine);
for i = 1:t
    if train_wine_labels(i,1) == 1
        train_labels(i,:) = [1;0;0];
    elseif train_wine_labels(i,1) == 2
        train_labels(i,:) = [0;1;0];
    else
        train_labels(i,:) = [0;0;1];
    end
end
%对测试数据集输出类别重新设置
[t1,r1] = size(test_wine);
for j = 1:t1
    if test_wine_labels(j,1) == 1
        test_labels(j,:) = [1;0;0];
    elseif train_wine_labels(i,1) == 2
        test_labels(j,:) = [0;1;0];
    else
        test_labels(j,:) = [0;0;1];
    end
end

%% 建立BP神经网络
%层数选择神经元设计
insize = 13;%输入层神经元数目
hidesize = 20;%隐含层神经元数目
outsize = 3;%输出层神经元数目

yita1 = 0.01;%输入层到隐含层之间的学习率
yita2 = 0.01;%隐含层到输出层之间的学习率
alpha = 0.2;%动量因子

W1 = rand(hidesize,insize);%输入层到隐含层之间的权重
W2 = rand(outsize,hidesize);%隐含层到输出层之间的权重
B1 = rand(hidesize,1);%隐含层神经元的阈值
B2 = rand(outsize,1);%输出层神经元的阈值

loop = 1000;
E = zeros(1,loop);
for loopi = 1:loop
    
    for i = 1:89
        x = train_wine(i,:);
        
        hidein = W1*x'+B1;%隐含层输入值
        hideout = zeros(hidesize,1);%计算隐含层输出值
        for j = 1:hidesize
            hideout(j) = sigmod(hidein(j));
        end
        
        yin(:,i) = W2*hideout+B2;%输入层输入值
        yout(:,i) = zeros(outsize,1);%输出层输出值
        for j = 1:outsize
            yout(j,i) = sigmod(yin(j,i));
%             yout(j,i) = purelin(yin(j,i));% output layer activation function is linear
        end
        
        e = yout(:,i)-train_labels(i,:)';%输出层计算结果误差
        E(loopi) = 0.5*sum(e.^2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %后向反馈
        dB2 = zeros(outsize,1);%误差对输出层阈值求偏导，计算阈值变化量
        for j = 1:outsize
            dB2(j) = sigmod(yin(j,i))*(1-sigmod(yin(j,i)))*e(j)*yita2; %源代码此处为dB2，错误，dB2为3*1向量不是1*1向量
%             dB2 = e(j)*yita2;% output layer activation function is linear
        end
        
        %隐含层与输出层之间的权重的变化量
        dW2 = zeros(outsize,hidesize);
        for j = 1:outsize
            for k = 1:hidesize
                dW2(j,k) = sigmod(yin(j,i))*(1-sigmod(yin(j,i)))*hideout(k)*e(j)*yita2;
%                 dW2(j,k) = hideout(k)*e(j)*yita2;% output layer activation function is linear
            end
        end
        
        %隐含层阈值变化量
        dB1 = zeros(hidesize,1);
        for j = 1:hidesize
            tempsum = 0;
            for k = 1:outsize
                tempsum = tempsum + sigmod(yin(k,i))*(1-sigmod(yin(k,i)))*W2(k,j)*sigmod(hidein(j))*(1-sigmod(hidein(j)))*e(k)*yita1;
%                 tempsum = tempsum + W2(k,j)*sigmod(hidein(j))*(1-sigmod(hidein(j)))*e(k)*yita1;% output layer activation function is linear
            end
            dB1(j) = tempsum;
        end
        
        %输入层到隐含层的权重变化量
        dW1 = zeros(hidesize,insize);
        for j = 1:hidesize
            for k = 1:insize
                tempsum = 0;
                for m = 1:outsize
                    tempsum = tempsum + sigmod(yin(m,i))*(1-sigmod(yin(m,i)))*W2(m,j)*sigmod(hidein(j))*(1-sigmod(hidein(j)))*x(k)*e(m)*yita1;
                end
                
                dW1(j,k) = tempsum;
                
            end
            
        end
        
        W1 = W1-dW1;
        W2 = W2-dW2;
        B1 = B1-dB1;
        B2 = B2-dB2;
        
    end
    
    if mod(loopi,100)==0
        loopi
    end
    
end

plot(E);

%% test data
for j =1:89
    x_test = test_wine(j,:);
    
    hidein_test= W1*x_test'+B1;%隐含层输入值
    hideout_test = zeros(hidesize,1);%计算隐含层输出值
    for j1 = 1:hidesize
        hideout_test(j1) = sigmod(hidein_test(j1));
    end
    
    yin_test(:,j) = W2*hideout_test+B2;%输入层输入值
    yout_test(:,j) = zeros(outsize,1);%输出层输出值
    for j2 = 1:outsize
        yout_test(j2,j) = sigmod(yin_test(j2,j));
    end
    
end