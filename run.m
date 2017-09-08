clear;
clc;

dataSetName = 'Data.mat';
data_dir = 'dim_234\';
dirctory = ['..\0_Data\', data_dir] ;
fileName = 'XinShuai_AllView_cls_';

load([dirctory, dataSetName]);
data = Data.data;
label = Data.label;
clear Data;

doNorm = true;
totalCycle = 5;
if doNorm
    type = 2;
    Data = Normalization(data, type);
end
ind_pos = label == 1;
ind_neg = label == 0;
tmpData = {[Data(ind_pos, :), label(ind_pos)], [Data(ind_neg, :), label(ind_neg)]};
Segment = samples2Pieces(tmpData , totalCycle) ;

DataSet = [];
for i = 1 : totalCycle
    DataSet{i, 2} =  [Segment{i, 1}; Segment{i, 2}];
    for j = 1 : totalCycle
        if i == j
            continue;
        else
            DataSet{i, 1} = [DataSet{i, 1}; Segment{j, 1}; Segment{j, 2}];
        end
    end
end

cls_num_set = [1:2];
for wId = 1 : length(cls_num_set)
    cls_num = cls_num_set(wId);
    postName = fileName;
    saveMatName =  ['.\report\', data_dir, postName, num2str(cls_num), '.mat']; % '.\report\iris_v.mat' ;
    saveFileName = ['.\report\', data_dir, postName, num2str(cls_num), '.txt'] ;
    
    warning off ;
    fid=fopen(saveFileName,'w');
    fclose(fid);
    
    conf.nb_epoch = 10000;
    conf.w_decay = 0.01;
    conf.lr = 0.05;
    conf.cls_num = cls_num;
    
    inPutInf.conf = conf;
    inPutInf.kernelized = false;
    
    saveRes = cell(totalCycle, 1);
    val_res = [];
    tst_res = [];
    for index_cycle = 1:totalCycle;
        trnData = DataSet{index_cycle, 1} ;
        tstData = DataSet{index_cycle, 2} ;
        [trnRes, t_train] = bagging_dyml(trnData, tstData , inPutInf) ;
        saveRes{index_cycle} = trnRes;
        tst_res = [tst_res ; [trnRes.tst.acc, trnRes.tst.AUC, trnRes.tst.AP, trnRes.tst.TP, trnRes.tst.TN, t_train]];
        fid=fopen(saveFileName,'a');
        fprintf('The %d cycle--- Acc: %.4f -- AUC: %.4f -- AP: %.4f -- TP: %.4f -- TN: %.4f ----;\n' , ...
            index_cycle , trnRes.tst.acc, trnRes.tst.AUC, trnRes.tst.AP, trnRes.tst.TP, trnRes.tst.TN) ;
        fprintf(fid,'The %d cycle--- Acc: %.4f -- AUC: %.4f -- AP: %.4f -- TP: %.4f -- TN: %.4f ----;\n' , ...
            index_cycle , trnRes.tst.acc, trnRes.tst.AUC, trnRes.tst.AP, trnRes.tst.TP, trnRes.tst.TN) ;
        fclose(fid);
    end;
    tst_res(totalCycle+1 , :) = mean(tst_res) ;
    tst_res(totalCycle+2 , :) = std(tst_res(1:totalCycle , :)) ; 
    
    fid=fopen(saveFileName,'a');
    fprintf('Validation MEAN:\n -- ACC: %.4f(%.4f) -- AUC: %.4f(%.4f) -- AP: %.4f(%.4f) -- TP: %.4f(%.4f) -- TN: %.4f(%.4f) -- ' ,...
        tst_res(totalCycle+1, 1), tst_res(totalCycle+2, 1), ...
        tst_res(totalCycle+1, 2), tst_res(totalCycle+1, 2), ...
        tst_res(totalCycle+1, 3), tst_res(totalCycle+1, 3), ...
        tst_res(totalCycle+1, 4), tst_res(totalCycle+1, 4), ...
        tst_res(totalCycle+1, 5), tst_res(totalCycle+1, 5)) ;
    fprintf(fid, 'Validation MEAN:\n -- ACC: %.4f(%.4f) -- AUC: %.4f(%.4f) -- AP: %.4f(%.4f) -- TP: %.4f(%.4f) -- TN: %.4f(%.4f) -- ' ,...
        tst_res(totalCycle+1, 1), tst_res(totalCycle+2, 1), ...
        tst_res(totalCycle+1, 2), tst_res(totalCycle+1, 2), ...
        tst_res(totalCycle+1, 3), tst_res(totalCycle+1, 3), ...
        tst_res(totalCycle+1, 4), tst_res(totalCycle+1, 4), ...
        tst_res(totalCycle+1, 5), tst_res(totalCycle+1, 5)) ;
    fprintf(' meanTime: %.3f(%.3f) .......\n' , tst_res(totalCycle+1 , 6) , tst_res(totalCycle+2 , 6)) ;
    fprintf(fid,' meanTime: %.3f(%.3f) .......\n' , tst_res(totalCycle+1 , 6) , tst_res(totalCycle+2 , 6)) ;
    fclose(fid);
    
    savedObj.tst_res = tst_res;
    save(saveMatName, 'savedObj');
end

