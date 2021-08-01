% ��ʱ
%% ��ջ�����������
clear
clc
close all
format long
load X.mat
%% GWO-SVR
% ѵ��/��������׼������ǰ3��Ԥ���һ�죩,��ǰ100������������
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
%% ���û����㷨ѡ����ѵ�SVR����
tic
SearchAgents_no=30; % ��Ⱥ����
Max_iteration=100; % ����������
dim=2; % ������Ҫ�Ż���������c��g
lb=[0.01,0.01]; % ����ȡֵ�½�
ub=[100,100]; % ����ȡֵ�Ͻ�

Alpha_pos=zeros(1,dim); % ��ʼ��Alpha�ǵ�λ��
Alpha_score=inf; % ��ʼ��Alpha�ǵ�Ŀ�꺯��ֵ��change this to -inf for maximization problems

Beta_pos=zeros(1,dim); % ��ʼ��Beta�ǵ�λ��
Beta_score=inf; % ��ʼ��Beta�ǵ�Ŀ�꺯��ֵ��change this to -inf for maximization problems

Delta_pos=zeros(1,dim); % ��ʼ��Delta�ǵ�λ��
Delta_score=inf; % ��ʼ��Delta�ǵ�Ŀ�꺯��ֵ��change this to -inf for maximization problems

Positions=initialization(SearchAgents_no,dim,ub,lb);

Convergence_curve=zeros(1,Max_iteration);

l=0; % ѭ��������

while l<Max_iteration  % �Ե�������ѭ��
    for i=1:size(Positions,1)  % ����ÿ����

       % ������λ�ó����������ռ䣬��Ҫ���»ص������ռ�
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        % ���ǵ�λ�������ֵ����Сֵ֮�䣬��λ�ò���Ҫ���������������ֵ����ص����ֵ�߽磻
        % ��������Сֵ����ش���Сֵ�߽�
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; % ~��ʾȡ��           
     
        % ������Ӧ�Ⱥ���ֵ
        cmd = ['-s 3 -t 2',' -c ',num2str(Positions(i,1)),' -g ',num2str(Positions(i,2))];
        model=svmtrain(output_train',input_train',cmd); % SVMģ��ѵ��
        [~,fitness]=svmpredict(output_test',input_test',model); % SVMģ��Ԥ�⼰�侫��
        fitness=fitness(2); % ��ƽ���������MSE��Ϊ�Ż���Ŀ�꺯��ֵ

        if fitness<Alpha_score % ���Ŀ�꺯��ֵС��Alpha�ǵ�Ŀ�꺯��ֵ
            Alpha_score=fitness; % ��Alpha�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ
            Alpha_pos=Positions(i,:); % ͬʱ��Alpha�ǵ�λ�ø���Ϊ����λ��
        end
        
        if fitness>Alpha_score && fitness<Beta_score % ���Ŀ�꺯��ֵ������Alpha�Ǻ�Beta�ǵ�Ŀ�꺯��ֵ֮��
            Beta_score=fitness; % ��Beta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ
            Beta_pos=Positions(i,:); % ͬʱ����Beta�ǵ�λ��
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score  % ���Ŀ�꺯��ֵ������Beta�Ǻ�Delta�ǵ�Ŀ�꺯��ֵ֮��
            Delta_score=fitness; % ��Delta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ
            Delta_pos=Positions(i,:); % ͬʱ����Delta�ǵ�λ��
        end
    end
    
    a=2-l*((2)/Max_iteration); % ��ÿһ�ε�����������Ӧ��aֵ��a decreases linearly fron 2 to 0

    for i=1:size(Positions,1) % ����ÿ����
        for j=1:size(Positions,2) % ����ÿ��ά��
            
            % ��Χ���λ�ø���
            
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % ����ϵ��A��Equation (3.3)
            C1=2*r2; % ����ϵ��C��Equation (3.4)
            
            % Alpha��λ�ø���
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % ����ϵ��A��Equation (3.3)
            C2=2*r2; % ����ϵ��C��Equation (3.4)
            
            % Beta��λ�ø���
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % ����ϵ��A��Equation (3.3)
            C3=2*r2; % ����ϵ��C��Equation (3.4)
            
            % Delta��λ�ø���
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            % λ�ø���
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    l=l+1;    
    Convergence_curve(l)=Alpha_score;
end
bestc=Alpha_pos(1,1);
bestg=Alpha_pos(1,2);
bestGWOaccuarcy=Alpha_score;
%% ��ӡ����ѡ����
disp('��ӡѡ����');
str=sprintf('Best Cross Validation Accuracy = %g%%��Best bestc = %g��Best bestg = %g',bestGWOaccuarcy*100,bestc,bestg);
disp(str)
%% ���ûع�Ԥ�������ѵĲ�������SVM����ѵ��
cmd_gwo_svr=['-s 3 -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
model_gwo_svr=svmtrain(output_train',input_train',cmd_gwo_svr); % SVMģ��ѵ��
toc
%% SVM����ع�Ԥ��
tic
[output_test_pre,acc]=svmpredict(output_test',input_test',model_gwo_svr); % SVMģ��Ԥ�⼰�侫��
test_pre=mapminmax('reverse',output_test_pre',rule2);
test_pre = test_pre';
toc
err_pre=X(7012:end)-test_pre;
figure('Name','�������ݲв�ͼ')
set(gcf,'unit','centimeters','position',[0.5,5,30,5])
plot(err_pre,'*-');
figure('Name','ԭʼ-Ԥ��ͼ')
plot(test_pre,'*r-');hold on;plot(X(104:end),'bo-');
legend('Ԥ��','ԭʼ')
set(gcf,'unit','centimeters','position',[0.5,13,30,5])

result=[X(7012:end),test_pre]

MAE=mymae(X(7012:end),test_pre)
MSE=mymse(X(7012:end),test_pre)
MAPE=mymape(X(7012:end),test_pre)
%% ��ʾ��������ʱ��
