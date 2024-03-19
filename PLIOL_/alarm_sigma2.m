dist2=importdata('dist.mat');
label_est2=importdata('label_est2.mat');

%从分类正确的dist里取阈值
threshold = []
for x=1:9 %从第一列到第九列
    ncc = []%ncc为正确分类的那一列数据
    locationStart = 1+20*(x-1);%设置计算阈值选取的那一段数据上下界
    locationEnd = 20+20*(x-1);
    for a=1:length(label_est2)%遍历label_est一整列
        if a>locationEnd & a<locationStart%如果a在范围外则不操作
            continue
        elseif label_est2(a) == x%如果a在范围内则判断如果a对应label_est的行的数据等于x的值
            ncc(end +1)=dist2(a, x);%则在ncc里添加a和x在dist对应的数据
        end
    end
    threshold(end + 1) = mean(ncc) + 1*std(ncc);%遍历完一列则计算一次阈值
end
threshold=threshold'

%看看有哪些符合阈值的
temp=zeros([9,1]);
tacc=zeros([9,1]);
for i=1:9%遍历1-9列
    locationStart = 1+20*(i-1);%设置不取值的范围
    locationEnd = 20+20*(i-1);
    for ii=1:180%遍历dist的1-180行
        if ii<=locationEnd & ii>=locationStart%如果ii是在不取值的范围内则不操作
           continue;
        elseif dist2(ii,i)>=threshold(i,1)%如果ii不在不去值得范围内，且dist第ii行第i列的数值大于threshold的第i行第一列
             temp(i) = temp(i) + 1;%则给temp的第i行的数字加1
        end 
    tacc(i)=temp(i)/160;
    end    
end
