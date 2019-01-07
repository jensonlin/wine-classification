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
    elseif test_wine_labels(j,1) == 2
        test_labels(j,:) = [0;1;0];
    else
        test_labels(j,:) = [0;0;1];
    end
end

%% normalize
[train_data,ps] = mapminmax(train_wine',0,1);
[test_data,ps1] = mapminmax(test_wine',0,1);

%% network
net = newff(train_data,train_labels', [20], { 'logsig' 'purelin' } , 'traingdx' , 'learngdm') ;%输入数据为 特征数*数据个数，输出为 类别向量*数据个数
net.trainparam.show = 50 ;%每间隔50步显示一次训练结果
net.trainparam.epochs = 1000 ;%允许最大训练步数500步
net.trainparam.goal = 0.01 ;%训练目标最小误差0.01
net.trainParam.lr = 0.001 ;%学习速率0.05

tic;
%% 开始训练
net = train( net, train_data , train_labels' );

toc;
%% 仿真测试
Y = sim( net,test_data ) ;

%统计识别正确率
[s1,s2] = size( Y ) ;
hitNum = 0 ;
count = 0;
for i = 1:s2
    [m,Index] = max(Y(:,i ));
    [m_c,Index_c] = max(test_labels(i,:)');
    if( Index == Index_c)
        hitNum = hitNum + 1 ;
    else
        count = count + 1;
    end
end
sprintf('识别率是 %3.3f%%',100 * hitNum / s2 )


