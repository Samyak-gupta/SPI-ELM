clear all;clc
tic
data=load('iris_dataset');
X=data.irisInputs';
Y=data.irisTargets';
Lmax=input("Enter maximum number of hidden neurons ");
for i=1:Lmax
[perfomance(i)]=elm_train_1(X,Y,Lmax);
end
plot(perfomance,'LineWidth',2);
xlabel('hidden nodes')
ylabel('RMSE')
legend('iris dataset');
grid
timeelapsed=toc
