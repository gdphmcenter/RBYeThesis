clear all
clc
close all

load data.mat
load label.mat

label=label'

labelknow1=label(1:2,:);
labelknow2=label(1024:1025,:);
labelknow3=label(2047:2048,:);
labelknow4=label(3070:3071,:);
labelknow5=label(4093:4094,:);
allbutIR014_l=[labelknow1;labelknow2;labelknow3;labelknow4;labelknow5]

dataknow1=data(1:2,:);
dataknow2=data(1024:1025,:);
dataknow3=data(2047:2048,:);
dataknow4=data(3070:3071,:);
dataknow5=data(4093:4094,:);
allbutIR014_s =[dataknow1;dataknow2;dataknow3;dataknow4;dataknow5]

labelunknow1=label(3:1023,:);
labelunknow2=label(1026:2046,:);
labelunknow3=label(2049:3069,:);
labelunknow4=label(3072:4092,:);
labelunknow5=label(4095:5115,:);
newIR014_l=[labelunknow1;labelunknow2;labelunknow3;labelunknow4;labelunknow5]

dataunknow1=data(3:1023,:);
dataunknow2=data(1026:2046,:);
dataunknow3=data(2049:3069,:);
dataunknow4=data(3072:4092,:);
dataunknow5=data(4095:5115,:);
newIR014_s=[dataunknow1;dataunknow2;dataunknow3;dataunknow4;dataunknow5]


granlevel=2000;   % level of granularity
chunksize=100000; % chunk size

datatra=allbutIR014_s;  % Labelled training data  DTra1--allbutIR014_s
labeltra=allbutIR014_l; % Labels of the labelled training data  LTra1--allbutIR014_l
datates=newIR014_s;  % Unlabelled training data  DTes1--newIR014_s
labeltes=newIR014_l; % Labels of the unlabelled training data LTes1--
[EstimatedLabels1,timme]=S3OFIS(datatra,labeltra,datates,granlevel,chunksize); % Run S3OFIS+
CM1=confusionmat(labeltes,EstimatedLabels1); % Confusion matrix
%Acc1=sum(sum(confusionmat(labeltes,EstimatedLabels1).*(eye(length(unique(labeltes))))))/length(labeltes); % Classification accuracy on unlabelled training data

temp1=confusionmat(labeltes,EstimatedLabels1);
temp2=eye(length(unique(labeltes)));
Acc1=sum(sum(temp1.*(temp2)))/length(labeltes);