function h=PSONEW(hid,hmax,hmin)
%% �����趨  
N = 10;  
d = hid;  
ger = 10;  
wmax = 0.9;  wmin=0.5;
cmax = 0.9;  cmin=0.5;
for i=1:hid
    xlim(i,1:2)=[hmin,hmax];
    vlim(i,1:2)=[-1,1];
end
xlimit = xlim(:,1:2);  
vlimit =vlim(:,1:2);  
%% ��Ⱥ��ʼ��  
x = repmat(xlimit(:,1)',N,1)+repmat(diff(xlimit'),N,1).*rand(N,d);  
v = repmat(vlimit(:,1)',N,1)+repmat(diff(vlimit'),N,1).*rand(N,d);  
xm = x;  
fxm = -inf*ones(N,1);  
ym = xlimit(:,1)'+diff(xlimit').*rand(1,d);  
fym = -inf;  
%% ��ʼ����  
for i = 1 : ger  
    t=i
   x1=round(x);
    for j = 1 : N  
        % ��Ӧ�Ⱥ��� 
       y(j)=fitnessnew(x1(j,:));
        if y(j)>fxm(j)  
       fxm(j)=y(j);  
       xm(j,:) = x(j,:);     %���弫ֵ����λ��
            if y(j)>fym  
                fym = y(j);  
                ym = x(j,:); %Ⱥ�弫ֵ����λ��
            end  
        end  
    end  
    w=wmax-(wmax-wmin)*i./ger;c1=cmax-(cmax-cmin)*i./ger;c2=cmin+(cmax-cmin)*i./ger;
    v = w*v+c1*rand*(xm-x)+c2*rand*(repmat(ym,N,1)-x);  
    x = x+v;  
    x = min(x,repmat(xlimit(:,2)',N,1));  
    x = max(x,repmat(xlimit(:,1)',N,1));  
    v = min(v,repmat(vlimit(:,2)',N,1));  
    v = max(v,repmat(vlimit(:,1)',N,1));  
    trace(i)=fym;
end  
toc  
ym=round(ym);
disp(['���Ž�Ϊ:',num2str(ym)]);  
disp(['����ֵΪ:',num2str(fym)]);
h=ym;
figure
plot(trace)
end