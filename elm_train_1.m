function [perfomance]=elm_train_1(X,Y,number_neurons)
% elm_train: this function allows to train a single hidden layer
% feedforward network  for regression with Moore-Penrose pseudoinverse of matrix.
% Inputs:- number_neurons: number of neurons in the hidden layer
%        - X:  N instances by Q atrebutes matrix of  training inputs;
%        - Y:  N raws and 1 atrebutes matrix of training targets
% outputs:- prefomance: RMSE of regression

%%%% 1st step: generate a random input weights
input_weights=rand(number_neurons,size(X,2))*2-1;
%%%% 2nd step: calculate the hidden layer
H=radbas(input_weights*X');
%%%% 3rd step: calculate the output weights beta
B=pinv(H') * Y ; %Moore-Penrose pseudoinverse of matrix
%%%% calculate the actual output 
output=(H' * B)' ;
%%%%%%%%%%%%%%%%calculate the prefomance%%%
perfomance=sqrt(mse(Y'-output));
end