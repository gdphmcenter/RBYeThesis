dist2=importdata('dist.mat');
label_est2=importdata('label_est2.mat');

%�ӷ�����ȷ��dist��ȡ��ֵ
threshold = []
for x=1:9 %�ӵ�һ�е��ھ���
    ncc = []%nccΪ��ȷ�������һ������
    locationStart = 1+20*(x-1);%���ü�����ֵѡȡ����һ���������½�
    locationEnd = 20+20*(x-1);
    for a=1:length(label_est2)%����label_estһ����
        if a>locationEnd & a<locationStart%���a�ڷ�Χ���򲻲���
            continue
        elseif label_est2(a) == x%���a�ڷ�Χ�����ж����a��Ӧlabel_est���е����ݵ���x��ֵ
            ncc(end +1)=dist2(a, x);%����ncc�����a��x��dist��Ӧ������
        end
    end
    threshold(end + 1) = mean(ncc) + 1*std(ncc);%������һ�������һ����ֵ
end
threshold=threshold'

%��������Щ������ֵ��
temp=zeros([9,1]);
tacc=zeros([9,1]);
for i=1:9%����1-9��
    locationStart = 1+20*(i-1);%���ò�ȡֵ�ķ�Χ
    locationEnd = 20+20*(i-1);
    for ii=1:180%����dist��1-180��
        if ii<=locationEnd & ii>=locationStart%���ii���ڲ�ȡֵ�ķ�Χ���򲻲���
           continue;
        elseif dist2(ii,i)>=threshold(i,1)%���ii���ڲ�ȥֵ�÷�Χ�ڣ���dist��ii�е�i�е���ֵ����threshold�ĵ�i�е�һ��
             temp(i) = temp(i) + 1;%���temp�ĵ�i�е����ּ�1
        end 
    tacc(i)=temp(i)/160;
    end    
end
