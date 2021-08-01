% 计时
%% 清空环境导入数据
clear
clc
close all
format long
load X.mat
%% GWO-SVR
% 训练/测试数据准备（用前3天预测后一天）,用前100天做测试数据
train_input(1,:)=X(1:7005);
train_input(2,:)=X(2:7006);
train_input(3,:)=X(3:7007);
train_output=[X(4:7008)]';
test_input(1,:)=X(7009:end-3);
test_input(2,:)=X(7010:end-2);
test_input(3,:)=X(7011:end-1);
test_output=[X(7012:end)]';

[input_train,rule1]=mapminmax(train_input);
[output_train,rule2]=mapminmax(train_output);
input_test=mapminmax('apply',test_input,rule1);
output_test=mapminmax('apply',test_output,rule2);
%% 利用灰狼算法选择最佳的SVR参数
tic
SearchAgents_no=30; % 狼群数量
Max_iteration=100; % 最大迭代次数
dim=2; % 此例需要优化两个参数c和g
lb=[0.01,0.01]; % 参数取值下界
ub=[100,100]; % 参数取值上界

Alpha_pos=zeros(1,dim); % 初始化Alpha狼的位置
Alpha_score=inf; % 初始化Alpha狼的目标函数值，change this to -inf for maximization problems

Beta_pos=zeros(1,dim); % 初始化Beta狼的位置
Beta_score=inf; % 初始化Beta狼的目标函数值，change this to -inf for maximization problems

Delta_pos=zeros(1,dim); % 初始化Delta狼的位置
Delta_score=inf; % 初始化Delta狼的目标函数值，change this to -inf for maximization problems

Positions=initialization(SearchAgents_no,dim,ub,lb);

Convergence_curve=zeros(1,Max_iteration);

l=0; % 循环计数器

while l<Max_iteration  % 对迭代次数循环
    for i=1:size(Positions,1)  % 遍历每个狼

       % 若搜索位置超过了搜索空间，需要重新回到搜索空间
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        % 若狼的位置在最大值和最小值之间，则位置不需要调整，若超出最大值，最回到最大值边界；
        % 若超出最小值，最回答最小值边界
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; % ~表示取反           
     
        % 计算适应度函数值
        cmd = ['-s 3 -t 2',' -c ',num2str(Positions(i,1)),' -g ',num2str(Positions(i,2))];
        model=svmtrain(output_train',input_train',cmd); % SVM模型训练
        [~,fitness]=svmpredict(output_test',input_test',model); % SVM模型预测及其精度
        fitness=fitness(2); % 以平均均方误差MSE作为优化的目标函数值

        if fitness<Alpha_score % 如果目标函数值小于Alpha狼的目标函数值
            Alpha_score=fitness; % 则将Alpha狼的目标函数值更新为最优目标函数值
            Alpha_pos=Positions(i,:); % 同时将Alpha狼的位置更新为最优位置
        end
        
        if fitness>Alpha_score && fitness<Beta_score % 如果目标函数值介于于Alpha狼和Beta狼的目标函数值之间
            Beta_score=fitness; % 则将Beta狼的目标函数值更新为最优目标函数值
            Beta_pos=Positions(i,:); % 同时更新Beta狼的位置
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score  % 如果目标函数值介于于Beta狼和Delta狼的目标函数值之间
            Delta_score=fitness; % 则将Delta狼的目标函数值更新为最优目标函数值
            Delta_pos=Positions(i,:); % 同时更新Delta狼的位置
        end
    end
    
    a=2-l*((2)/Max_iteration); % 对每一次迭代，计算相应的a值，a decreases linearly fron 2 to 0

    for i=1:size(Positions,1) % 遍历每个狼
        for j=1:size(Positions,2) % 遍历每个维度
            
            % 包围猎物，位置更新
            
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % 计算系数A，Equation (3.3)
            C1=2*r2; % 计算系数C，Equation (3.4)
            
            % Alpha狼位置更新
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % 计算系数A，Equation (3.3)
            C2=2*r2; % 计算系数C，Equation (3.4)
            
            % Beta狼位置更新
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % 计算系数A，Equation (3.3)
            C3=2*r2; % 计算系数C，Equation (3.4)
            
            % Delta狼位置更新
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            % 位置更新
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    l=l+1;    
    Convergence_curve(l)=Alpha_score;
end
bestc=Alpha_pos(1,1);
bestg=Alpha_pos(1,2);
bestGWOaccuarcy=Alpha_score;
%% 打印参数选择结果
disp('打印选择结果');
str=sprintf('Best Cross Validation Accuracy = %g%%，Best bestc = %g，Best bestg = %g',bestGWOaccuarcy*100,bestc,bestg);
disp(str)
%% 利用回归预测分析最佳的参数进行SVM网络训练
cmd_gwo_svr=['-s 3 -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
model_gwo_svr=svmtrain(output_train',input_train',cmd_gwo_svr); % SVM模型训练
toc
%% SVM网络回归预测
tic
[output_test_pre,acc]=svmpredict(output_test',input_test',model_gwo_svr); % SVM模型预测及其精度
test_pre=mapminmax('reverse',output_test_pre',rule2);
test_pre = test_pre';
toc
err_pre=X(7012:end)-test_pre;
figure('Name','测试数据残差图')
set(gcf,'unit','centimeters','position',[0.5,5,30,5])
plot(err_pre,'*-');
figure('Name','原始-预测图')
plot(test_pre,'*r-');hold on;plot(X(104:end),'bo-');
legend('预测','原始')
set(gcf,'unit','centimeters','position',[0.5,13,30,5])

result=[X(7012:end),test_pre]

MAE=mymae(X(7012:end),test_pre)
MSE=mymse(X(7012:end),test_pre)
MAPE=mymape(X(7012:end),test_pre)
%% 显示程序运行时间
