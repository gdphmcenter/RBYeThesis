function [EstimatedLabels1,timee]=S3OFIS(datatra,labeltra,datates,granlevel,chunksize)
data1=datatra;%训练样本数据
label1=labeltra;%训练样本数据的标签
lambda0=1;
lambda1=2;
[L1,W]=size(data1);%训练样本数据的行数返回L1，列数返回W；L1为数据样本量，w为维数
numbatch1=ceil(L1/chunksize);%朝正无穷方向对（数据样本量/块数）取整；要迭代多少次
%%
%有标签数据的训练过程
unilabel=unique(label1);%去掉已标记的样本数据标签的重复值；unilabel=[1;2;3;4;5;6;7;8;9;10]；样本有哪几个类
numlabel=length(unilabel);%去掉重复已标记样本数据的长度；有多少个不同的类
seq11=[1:chunksize:L1,L1];%seq11初值为1，步长为块数，终值为数据样本数，再加上数据样本数；seq11=[1;20]
averdist=0;%平均值为0
Ldata=0;
timee=0;
tic

for ii=1:1:numbatch1 %初值为1，步长为1，终值为1
    data10=data1(seq11(ii):1:seq11(ii+1)-1,:);%seq11(ii)为在seq11里的数按照ii的规则；19X512；比data1训练样本数据少了最底下一行
    label10=label1(seq11(ii):1:seq11(ii+1)-1);%19X1；比label少了一行
    dist00=pdist(data10,'euclidean').^2;%计算data10里面各个向量对象之间的距离;dist00=1x171
    
    for tt=1:granlevel  %从1到参数G
        dist00(dist00>mean(dist00))=[];%若dist00>平均值则在dist00里删除该值；
    end %从1循环到G，共循环（G-1）次，直到删到只剩下一个最佳的距离
    
    averdist=(averdist*Ldata+mean(dist00)*length(label10))/(Ldata+length(label10));%重新计算平均值
    Ldata=Ldata+length(label10);%0+19=19
    if ii==1%
        %调用training子函数
        TrainedClassifier=training([],data10,label10,granlevel,lambda0,lambda1,unilabel);
    end
    if ii>1%无运行
        TrainedClassifier=training(TrainedClassifier,data10,label10,granlevel,lambda0,lambda1,unilabel);
    end
end
%输出了:TrainedClassifier


%记录完成的时间
timee=timee+toc;
%%
%对无标签的数据进行预测
data0=datates;%未标记的数据
[L0,W]=size(data0);%未标记数据的样本数返回L0，维数返回w
numbatch0=ceil(L0/chunksize);%numbatch0=1
seq00=[1:chunksize:L0,L0];%seq00=[1,1305] 
tic

for ii=1:1:numbatch0
    tempseq=seq00(ii):1:seq00(ii+1)-1;%tempseq=1X1304
    data00=data0(tempseq,:);%data00=1304X512
    LC=length(tempseq);%LC=1304
    pseduolabel=zeros(LC,1);%伪标签是一个行为1304，列为1的零矩阵
    dist00=pdist(data00,'euclidean').^2;%dist00=1X849556
    for tt=1:granlevel
        dist00(dist00>mean(dist00))=[];
    end
    averdist=(averdist*Ldata+mean(dist00)*length(tempseq))/(Ldata+length(tempseq));
    Ldata=Ldata+length(tempseq);%Ldata=19+1304=1323
    %%
    %调用了testing子函数
    [label_est,dist2]=testing(TrainedClassifier,data00,unilabel,numlabel);%第⑤个函数
    C=exp(-1*(dist2)./(2*averdist));%exp(x)返回指数 e^x;./点运算;C=1304x10x10
    C1=[];
    C2=[];
    for tt=1:1:numlabel %1到已标记样本类别数
        C1=[C1,C(:,:,tt)];%C1=1304xtt*10
        C2=[C2,ones(1,numlabel)*tt];%ones(n,m)是一个n行m列的一矩阵；C2=1Xtt*10的一矩阵
    end
    Idx=[];
    for tt=1:1:LC %LC=1034
        [~,seq]=sort(C1(tt,:),'descend'); %[~,m]表示忽略行输出，只输出列；sort(a,'descend')降序；seq为1X100，从1-100，但是有些顺序乱了？
        C3=C2(seq);%C2为1X100，从1-10；C3为1-10，但是最后十列的数据没有顺序
        C3=C3(1:1:numlabel);%读取C3前十行
        [UD,x1,x2]=unique(C3);%UD为1X2=1，2的矩阵；X1为2X1=1，10的矩阵；X2为10X1=1.....2的矩阵
        F = histc(x2,1:numel(x1));%hitc()定义数值的边界;numel()返回数组中元素个数;F=[9;1]-9和1分别为X2中不同的两个类别的个数，行为x1个数.
        [x1,x2]=max(F);%max()求数组的最大元素;x1为F里最大的元素，x2最大元素的行号
        if x1>=ceil((numlabel+0.1)/2)%如果F里最大的元素大于等于(类别数+0.1)/2的向正无穷取整
            Idx=[Idx;tt];
        end
    end%一直循环，最后输出Idx=1-1304
    pseduolabel(Idx)=label_est(Idx);%label_est为1304X1的2矩阵；pseduolabel为之前定义的1304X1的零矩阵
    pseudolabel10=pseduolabel(Idx);%pseduolabel0为1304X1的2矩阵
    data10=data00(Idx,:);%data10变为1304X512
    %%
    if isempty(pseudolabel10)~=1%如果pseudolabel10是空矩阵
        %调用training子函数
        [TrainedClassifier]=training(TrainedClassifier,data10,pseudolabel10,granlevel,lambda0,lambda1,unilabel);
    end
end



timee=timee+toc;
data0=datates;%未标记的数据样本重新赋值，为1305X512

Output0=ones(size(data0,1),numlabel);%size(A,1)为返回矩阵A的行，size(A,2)为返回矩阵A的列；output为1305X10的1矩阵

[label_est,dist2,dist]=testing(TrainedClassifier,data0,unilabel,1);%第五个函数

[x1,x2,x3]=size(dist2);%x1为dist的行1305，x2为dist的列1，x3为dist第三维10
dist3=reshape(mean(dist2,2),[x1,x3]);%mean(A,2)输出Ａ在行的平均值；reshape(A,[m,n])是将A按列优先的顺序，返回与[m,n]维度一致的矩阵

Output0=Output0.*exp(-1*dist3./(2*averdist));

[~,EstimatedLabels1]=max(Output0,[],2);%只输出列EstimatedLabels1；max(A,[],dim)是输出A矩阵每一行（dim=2）最大值的列向量

end




function [TrainedClassifier]=training(TrainedClassifier,allbutIR014_s,allbutIR014_l,GranLevel,lambda0,lambda1,seq)
data_train=allbutIR014_s;%带标签的数据样本的数据
label_train=allbutIR014_l;%带标签的数据样本的标签
N=length(seq);%有多少个类别
if isempty(TrainedClassifier)==1 %isempty（）判断是否为空，空返回1；
    CN=zeros(N,1);%CN为行为N，列为1的零矩阵
    averdist=zeros(N,1);%平均值为行为N，列为1的零矩阵
    centre={};
    data_train1={};
    for ii=1:1:N  %从ii初值为1到类别数N，步长为1
        centre{ii}=[]; %将数组赋予空列表
        data_train1{ii}=data_train(label_train==seq(ii),:);  %？；data_train1为一个数组，
        if isempty(data_train1{ii})~=1  %~=为不等于；如果data_train1不为空，则isempty不返回1
            [CN0,W]=size(data_train1{ii});  %data_train1的行输出为CNO，列为w
            dist00=pdist(data_train1{ii},'euclidean').^2;  %计算data_train1里各个向量的欧几里得距离
            for tt=1:GranLevel
                dist00(dist00>mean(dist00))=[];  %若dist00>平均值则在dist00里删除该值；
            end
            averdist(ii)=mean(dist00);  %将dist00平均值赋给averdist(ii)
            if isnan(averdist(ii))==1  %isnan()判断A的的元素是否为NaN，如果是则返回1，否则返回0
                averdist(ii)=0;
            end
            CN(ii)=CN(ii)+CN0;  %CN慢慢变成10x1全是2
            [centre{ii}]=online_training_Euclidean(data_train1{ii},averdist(ii));  %第④个函数
        end
    end
    centre0=centre;  %centre最后慢慢累加为10列的[1x512 double]
end
if isempty(TrainedClassifier)==0 %以下不会被运行
    centre0=TrainedClassifier.centre;
    averdist=TrainedClassifier.averdist;
    CN=TrainedClassifier.CN;
    centre={};
    data_train1={};
    for ii=1:1:N
        centre{ii}=[];
        data_train1{ii}=data_train(label_train==seq(ii),:);%
        if isempty(data_train1{ii})~=1
            [CN0,W]=size(data_train1{ii});
            dist00=pdist(data_train1{ii},'euclidean').^2;
            for tt=1:GranLevel
                dist00(dist00>mean(dist00))=[];
            end
            averdist(ii)=(CN(ii)*averdist(ii)+CN0*mean(dist00))/(CN(ii)+CN0);
            if isnan(averdist(ii))==1
                averdist(ii)=0;
            end
            CN(ii)=CN(ii)+CN0;
            %调用了online_training_Euclidean子函数
            [centre{ii}]=online_training_Euclidean(data_train1{ii},averdist(ii));
        end
    end
    %调用CombiningCentres子函数
    [centre0]=CombiningCentres(centre0,centre,averdist,N,lambda0,lambda1);
end
TrainedClassifier.centre=centre0;
TrainedClassifier.averdist=averdist;
TrainedClassifier.CN=CN;
end



function [centre0]=CombiningCentres(centre0,centre,thresholddistance,N,lambda0,lambda1)
La1=[];
La2=[];
CC1=[];
CC2=[];
for ii=1:1:N
    CC1=[CC1;centre0{ii}];
    CC2=[CC2;centre{ii}];
    La1=[La1;ones(size(centre0{ii},1),1)*ii];
    La2=[La2;ones(size(centre{ii},1),1)*ii];
end
if isempty(CC1)~=1 && isempty(CC2)~=1
    dist11=pdist2(CC1,CC2).^2;
    for ii=1:1:N
        seq11=find(La1==ii);
        seq22=find(La1~=ii);
        seq33=find(La2==ii);
        if isempty(seq11)~=1 && isempty(seq11)~=1
            %%
            dist1=dist11(seq11,seq33);
            seq1=min(dist1,[],1);
            seq2=find(seq1>=thresholddistance(ii)*lambda0);
            dist2=dist11(seq22,seq33);
            dist3=repmat(thresholddistance(La1(seq22))*lambda1,1,length(seq33));
            dist4=dist2-dist3;
            seq44=min(dist4,[],1);
            seq4=find(seq44<=0);
            centre0{ii}=[centre0{ii};centre{ii}(unique([seq2,seq4]),:)];
        else
            centre0{ii}=[centre0{ii};centre{ii}];
        end
    end
elseif isempty(CC1)==1
    centre0=centre;
end
end

function [centre]=online_training_Euclidean(data,averdist)
[L,W]=size(data);
centre=data(1,:);
member=1;
for ii=2:1:L
    [dist3,pos3]=min(pdist2(data(ii,:),centre,'euclidean').^2);
    if dist3>averdist
        centre(end+1,:)=data(ii,:);
        member(end+1,1)=1;
    else
        centre(pos3,:)=(member(pos3,1)*centre(pos3,:)+data(ii,:))/(member(pos3,1)+1);
        member(pos3,1)=member(pos3,1)+1;
    end
end
end

function [label_est,dist2,dist]=testing(TrainedClassifier,data_test,seq,K)
centre=TrainedClassifier.centre;%centre的长度
N=length(centre);
L=size(data_test,1);%输出data_test的行
dist=zeros(L,N);
dist2=zeros(L,K,N);
for i=1:1:N
    if isempty(centre{i})~=1
        tempseq=pdist2(data_test,centre{i},'euclidean').^2;%pdist2(a,b,distance)输出ab中每一对观测之间的距离
        dist(:,i)=min(tempseq,[],2);%取temseq2维内的最小值
        tempseq=sort(tempseq,2,'ascend');%2为对x每一行进行升序排列
        K1=min([length(centre{i}(:,1)),K]);
        dist2(:,1:1:K1,i)=tempseq(:,1:1:K1);
    end
end
[~,label_est]=min(dist,[],2);%[Y,U]=min(A)：返回行向量Y和U，Y向量记录A的每列的最小值，U向量记录每列最小值的行号
label_est=seq(label_est);
end