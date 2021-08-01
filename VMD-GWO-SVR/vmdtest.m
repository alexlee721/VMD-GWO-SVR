tic
clc
clear all
fs=8760;%����Ƶ��
Ts=1/fs;%��������
L=8760;%��������
t=(0:L-1)*Ts;%ʱ������
STA=1; %������ʼλ��
%----------------������Ȧ���ϵ�����-----------------------------------------
load X
% %----------------�����������ϵ�����---------------------------------------
% load('X121_DE_time.mat')
%--------- some sample parameters forVMD������VMD��Ʒ������������---------------
alpha = 2500;       % moderate bandwidth constraint���ʶȵĴ���Լ��/�ͷ�����
tau = 0;          % noise-tolerance (no strict fidelity enforcement)���������ޣ�û���ϸ�ı����ִ�У�
K = 4;              % modes���ֽ��ģ̬��
DC = 0;             % no DC part imposed����ֱ������
init = 1;           % initialize omegas uniformly  ��omegas�ľ��ȳ�ʼ��
tol = 1e-7         
%--------------- Run actual VMD code:���ݽ���vmd�ֽ�---------------------------
[u, u_hat, omega] = VMD(X, alpha, tau, K, DC, init, tol);
figure(1);
imfn=u;
n=size(imfn,1); %size(X,1),���ؾ���X��������size(X,2),���ؾ���X��������N=size(X,2)�����ǰѾ���X��������ֵ��N
subplot(n+1,1,1);  % m�����У�n�����У�p��������ͼ�λ��ڵڼ��С��ڼ��С�����subplot(2,2,[1,2])
plot(t,X); %�����ź�
ylabel('ԭʼ�ź�','fontsize',12,'fontname','����');

for n1=1:n
    subplot(n+1,1,n1+1);
    plot(t,u(n1,:));%���IMF������a(:,n)���ʾ����a�ĵ�n��Ԫ�أ�u(n1,:)��ʾ����u��n1��Ԫ��
    ylabel(['IMF' int2str(n1)]);%int2str(i)�ǽ���ֵi���������ת����ַ���y������
end
 xlabel('ʱ��\itt/s','fontsize',12,'fontname','����');
 toc;
 %----------------------��������Ƶ��ȷ���ֽ����K-----------------------------
average=mean(omega);%������е�ƽ��ֵ
