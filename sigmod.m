function [ y ] = sigmod( x )
% 激活函数Sigmod，用于神经网络

y = 1/(1+exp(-x));
end