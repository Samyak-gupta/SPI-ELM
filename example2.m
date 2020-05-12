clear all;clc
tic
data=load('iris_dataset');
X=data.irisInputs';
Y=data.irisTargets';
Lmax=input("Enter maximum number of hidden neurons ");
for i=1:Lmax
[p2(i)]=elm_train_final_sig(X,Y,i);
end
plot(p2,'LineWidth',2);
xlabel('hidden nodes')
ylabel('RMSE')
legend('iris dataset');
grid
timeelapsed=toc