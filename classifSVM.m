function [acc,C] = myfun(i_layer,labels,paramsSVM)
    %%Input
        %i_layer: The index of layer that performed the feature extraction
        %labels: True class labels, size equal to the number of intances
        %paramsSVM: String with svm parameters according to libSVM (matlab)
     %Output
        %acc: the SVM accuracy, 3/4 of data for test, rest for train
        %C: The corresponding confusion matrix    
    %% CODE
    my_features = readNPY(['imgfeatures_layer' num2str(i_layer) '.npy']);
    my_features=my_features';%#treat features as row vectors
    my_features(17,:)=[];
    
    data = zscore(my_features); %#scale features
    numInst = size(data,1);
    numLabels = max(labels);
    
    %# split training/testing
    idx = randperm(numInst);
    numTrain = round(3*numInst/4); numTest = numInst - numTrain;
    trainData = data(idx(1:numTrain),:);  testData = data(idx(numTrain+1:end),:);
    trainLabel = labels(idx(1:numTrain)); testLabel = labels(idx(numTrain+1:end));
    
    %# train one-against-all models
    model = cell(numLabels,1);
    for k=1:numLabels
        model{k} = svmtrain(double(trainLabel==k), trainData, paramsSVM);%'-t 0 -b 1'
    end
    
    %# get probability estimates of test instances using each model
    prob = zeros(numTest,numLabels);
    for k=1:numLabels
        [~,~,p] = svmpredict(double(testLabel==k), testData, model{k}, '-b 1');
        prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
    end
    
    %# predict the class with the highest probability
    [~,pred] = max(prob,[],2);
    acc = sum(pred == testLabel) ./ numel(testLabel)    %# accuracy
    C = confusionmat(testLabel, pred)
end
