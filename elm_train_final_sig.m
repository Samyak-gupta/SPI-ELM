function [p2]=elm_train_final_sig(Train,Tar,Lmax)
% elm_train: this function allows to train a single hidden layer
% feedforward network  for regression with Moore-Penrose pseudoinverse of matrix.
% Inputs:- Lmax: maximum number of input neurons in the hidden layer
%        - Train:  N instances by Q attributes matrix of  training inputs;
%        - Tar:  N raws and 1 atrebutes matrix of training targets
% outputs:- perfomance: RMSE of regression

% generate a random input weights and bias of input neurons
input_weights=rand(Lmax,size(Train,2))*2-1;
% calculate the hidden layer
H = 1 ./ (1 + exp(-input_weights*Train'));
% calculate the output weights beta
B=geninv(H') * Tar ; % Using Moore-Penrose pseudoinverse of matrix from paper 1
% calculate the actual output 
output=(H' * B)' ;
% calculate the perfomance
p2=sqrt(mse(Tar'-output));
end