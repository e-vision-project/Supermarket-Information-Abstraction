 clear all
 %reading features from the .npy file
 my_features = readNPY('intermediate_layers\resnet_features.npy');
 
 %loading labels  - essential for the cross validation step
 load('labels.mat')
 
 %setting an SVM template
 t = templateSVM('Standardize',false,'KernelFunction','linear')
 %setting several SVMs for multi-label task 
 Mdl = fitcecoc(my_features,labels,'Learners',t);

 %setting a cross validator
 CVMdl = crossval(Mdl,'Leaveout','on');
 %Calculating the generalization error
 for i=length(labels)
 genError(i) = kfoldLoss(CVMdl,'Folds',i)
 end
 mean(genError)
