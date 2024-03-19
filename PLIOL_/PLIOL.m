function [EstimatedLabels1,timee]=S3OFIS(datatra,labeltra,datates,granlevel,chunksize)
data1=datatra;%ѵ����������
label1=labeltra;%ѵ���������ݵı�ǩ
lambda0=1;
lambda1=2;
[L1,W]=size(data1);%ѵ���������ݵ���������L1����������W��L1Ϊ������������wΪά��
numbatch1=ceil(L1/chunksize);%���������ԣ�����������/������ȡ����Ҫ�������ٴ�
%%
%�б�ǩ���ݵ�ѵ������
unilabel=unique(label1);%ȥ���ѱ�ǵ��������ݱ�ǩ���ظ�ֵ��unilabel=[1;2;3;4;5;6;7;8;9;10]���������ļ�����
numlabel=length(unilabel);%ȥ���ظ��ѱ���������ݵĳ��ȣ��ж��ٸ���ͬ����
seq11=[1:chunksize:L1,L1];%seq11��ֵΪ1������Ϊ��������ֵΪ�������������ټ���������������seq11=[1;20]
averdist=0;%ƽ��ֵΪ0
Ldata=0;
timee=0;
tic

for ii=1:1:numbatch1 %��ֵΪ1������Ϊ1����ֵΪ1
    data10=data1(seq11(ii):1:seq11(ii+1)-1,:);%seq11(ii)Ϊ��seq11���������ii�Ĺ���19X512����data1ѵ�������������������һ��
    label10=label1(seq11(ii):1:seq11(ii+1)-1);%19X1����label����һ��
    dist00=pdist(data10,'euclidean').^2;%����data10���������������֮��ľ���;dist00=1x171
    
    for tt=1:granlevel  %��1������G
        dist00(dist00>mean(dist00))=[];%��dist00>ƽ��ֵ����dist00��ɾ����ֵ��
    end %��1ѭ����G����ѭ����G-1���Σ�ֱ��ɾ��ֻʣ��һ����ѵľ���
    
    averdist=(averdist*Ldata+mean(dist00)*length(label10))/(Ldata+length(label10));%���¼���ƽ��ֵ
    Ldata=Ldata+length(label10);%0+19=19
    if ii==1%
        %����training�Ӻ���
        TrainedClassifier=training([],data10,label10,granlevel,lambda0,lambda1,unilabel);
    end
    if ii>1%������
        TrainedClassifier=training(TrainedClassifier,data10,label10,granlevel,lambda0,lambda1,unilabel);
    end
end
%�����:TrainedClassifier


%��¼��ɵ�ʱ��
timee=timee+toc;
%%
%���ޱ�ǩ�����ݽ���Ԥ��
data0=datates;%δ��ǵ�����
[L0,W]=size(data0);%δ������ݵ�����������L0��ά������w
numbatch0=ceil(L0/chunksize);%numbatch0=1
seq00=[1:chunksize:L0,L0];%seq00=[1,1305] 
tic

for ii=1:1:numbatch0
    tempseq=seq00(ii):1:seq00(ii+1)-1;%tempseq=1X1304
    data00=data0(tempseq,:);%data00=1304X512
    LC=length(tempseq);%LC=1304
    pseduolabel=zeros(LC,1);%α��ǩ��һ����Ϊ1304����Ϊ1�������
    dist00=pdist(data00,'euclidean').^2;%dist00=1X849556
    for tt=1:granlevel
        dist00(dist00>mean(dist00))=[];
    end
    averdist=(averdist*Ldata+mean(dist00)*length(tempseq))/(Ldata+length(tempseq));
    Ldata=Ldata+length(tempseq);%Ldata=19+1304=1323
    %%
    %������testing�Ӻ���
    [label_est,dist2]=testing(TrainedClassifier,data00,unilabel,numlabel);%�ڢݸ�����
    C=exp(-1*(dist2)./(2*averdist));%exp(x)����ָ�� e^x;./������;C=1304x10x10
    C1=[];
    C2=[];
    for tt=1:1:numlabel %1���ѱ�����������
        C1=[C1,C(:,:,tt)];%C1=1304xtt*10
        C2=[C2,ones(1,numlabel)*tt];%ones(n,m)��һ��n��m�е�һ����C2=1Xtt*10��һ����
    end
    Idx=[];
    for tt=1:1:LC %LC=1034
        [~,seq]=sort(C1(tt,:),'descend'); %[~,m]��ʾ�����������ֻ����У�sort(a,'descend')����seqΪ1X100����1-100��������Щ˳�����ˣ�
        C3=C2(seq);%C2Ϊ1X100����1-10��C3Ϊ1-10���������ʮ�е�����û��˳��
        C3=C3(1:1:numlabel);%��ȡC3ǰʮ��
        [UD,x1,x2]=unique(C3);%UDΪ1X2=1��2�ľ���X1Ϊ2X1=1��10�ľ���X2Ϊ10X1=1.....2�ľ���
        F = histc(x2,1:numel(x1));%hitc()������ֵ�ı߽�;numel()����������Ԫ�ظ���;F=[9;1]-9��1�ֱ�ΪX2�в�ͬ���������ĸ�������Ϊx1����.
        [x1,x2]=max(F);%max()����������Ԫ��;x1ΪF������Ԫ�أ�x2���Ԫ�ص��к�
        if x1>=ceil((numlabel+0.1)/2)%���F������Ԫ�ش��ڵ���(�����+0.1)/2����������ȡ��
            Idx=[Idx;tt];
        end
    end%һֱѭ����������Idx=1-1304
    pseduolabel(Idx)=label_est(Idx);%label_estΪ1304X1��2����pseduolabelΪ֮ǰ�����1304X1�������
    pseudolabel10=pseduolabel(Idx);%pseduolabel0Ϊ1304X1��2����
    data10=data00(Idx,:);%data10��Ϊ1304X512
    %%
    if isempty(pseudolabel10)~=1%���pseudolabel10�ǿվ���
        %����training�Ӻ���
        [TrainedClassifier]=training(TrainedClassifier,data10,pseudolabel10,granlevel,lambda0,lambda1,unilabel);
    end
end



timee=timee+toc;
data0=datates;%δ��ǵ������������¸�ֵ��Ϊ1305X512

Output0=ones(size(data0,1),numlabel);%size(A,1)Ϊ���ؾ���A���У�size(A,2)Ϊ���ؾ���A���У�outputΪ1305X10��1����

[label_est,dist2,dist]=testing(TrainedClassifier,data0,unilabel,1);%���������

[x1,x2,x3]=size(dist2);%x1Ϊdist����1305��x2Ϊdist����1��x3Ϊdist����ά10
dist3=reshape(mean(dist2,2),[x1,x3]);%mean(A,2)��������е�ƽ��ֵ��reshape(A,[m,n])�ǽ�A�������ȵ�˳�򣬷�����[m,n]ά��һ�µľ���

Output0=Output0.*exp(-1*dist3./(2*averdist));

[~,EstimatedLabels1]=max(Output0,[],2);%ֻ�����EstimatedLabels1��max(A,[],dim)�����A����ÿһ�У�dim=2�����ֵ��������

end




function [TrainedClassifier]=training(TrainedClassifier,allbutIR014_s,allbutIR014_l,GranLevel,lambda0,lambda1,seq)
data_train=allbutIR014_s;%����ǩ����������������
label_train=allbutIR014_l;%����ǩ�����������ı�ǩ
N=length(seq);%�ж��ٸ����
if isempty(TrainedClassifier)==1 %isempty�����ж��Ƿ�Ϊ�գ��շ���1��
    CN=zeros(N,1);%CNΪ��ΪN����Ϊ1�������
    averdist=zeros(N,1);%ƽ��ֵΪ��ΪN����Ϊ1�������
    centre={};
    data_train1={};
    for ii=1:1:N  %��ii��ֵΪ1�������N������Ϊ1
        centre{ii}=[]; %�����鸳����б�
        data_train1{ii}=data_train(label_train==seq(ii),:);  %����data_train1Ϊһ�����飬
        if isempty(data_train1{ii})~=1  %~=Ϊ�����ڣ����data_train1��Ϊ�գ���isempty������1
            [CN0,W]=size(data_train1{ii});  %data_train1�������ΪCNO����Ϊw
            dist00=pdist(data_train1{ii},'euclidean').^2;  %����data_train1�����������ŷ����þ���
            for tt=1:GranLevel
                dist00(dist00>mean(dist00))=[];  %��dist00>ƽ��ֵ����dist00��ɾ����ֵ��
            end
            averdist(ii)=mean(dist00);  %��dist00ƽ��ֵ����averdist(ii)
            if isnan(averdist(ii))==1  %isnan()�ж�A�ĵ�Ԫ���Ƿ�ΪNaN��������򷵻�1�����򷵻�0
                averdist(ii)=0;
            end
            CN(ii)=CN(ii)+CN0;  %CN�������10x1ȫ��2
            [centre{ii}]=online_training_Euclidean(data_train1{ii},averdist(ii));  %�ڢܸ�����
        end
    end
    centre0=centre;  %centre��������ۼ�Ϊ10�е�[1x512 double]
end
if isempty(TrainedClassifier)==0 %���²��ᱻ����
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
            %������online_training_Euclidean�Ӻ���
            [centre{ii}]=online_training_Euclidean(data_train1{ii},averdist(ii));
        end
    end
    %����CombiningCentres�Ӻ���
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
centre=TrainedClassifier.centre;%centre�ĳ���
N=length(centre);
L=size(data_test,1);%���data_test����
dist=zeros(L,N);
dist2=zeros(L,K,N);
for i=1:1:N
    if isempty(centre{i})~=1
        tempseq=pdist2(data_test,centre{i},'euclidean').^2;%pdist2(a,b,distance)���ab��ÿһ�Թ۲�֮��ľ���
        dist(:,i)=min(tempseq,[],2);%ȡtemseq2ά�ڵ���Сֵ
        tempseq=sort(tempseq,2,'ascend');%2Ϊ��xÿһ�н�����������
        K1=min([length(centre{i}(:,1)),K]);
        dist2(:,1:1:K1,i)=tempseq(:,1:1:K1);
    end
end
[~,label_est]=min(dist,[],2);%[Y,U]=min(A)������������Y��U��Y������¼A��ÿ�е���Сֵ��U������¼ÿ����Сֵ���к�
label_est=seq(label_est);
end