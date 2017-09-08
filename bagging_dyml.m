function [Res, t_train] = bagging_dyml(trnData, tstData, inPutInf)
    %
    % trainSet = [data, label], label \in {0, 1}
    %
    trnLabel = trnData(:, end);
    trnData(:, end) = [];   
    
    tic;
    if inPutInf.kernelized == true
        trainData = trnData;
        testData = tstData(:, 1:end-1);     testLabel = tstData(:, end);
        [empTrn, empTst] = generateEmpiricalData(trainData , testData , inPutInf);
        trnData = empTrn;
        tstData = [empTst, testLabel];
    end
    
    Res = bagging_train(trnData, trnLabel, tstData, inPutInf.conf);
    t_train = toc;
end

function res = bagging_train(trn, T, tst, conf)
    
    tstLabel = int32((tst(:, end)-0.5)*2); % map target from {0, 1} to {-1, 1}
    tst(:, end) = [];
    
    pos_ind = T == 1;
    neg_ind = T == 0;
    X_pos = trn(pos_ind, :);      y_pos =  ones(sum(pos_ind), 1);
    X_neg = trn(neg_ind, :);      y_neg = -ones(sum(neg_ind), 1);
    
    total_times = conf.cls_num;
    
    y_tst = 0;
    for t = 1 : total_times
        fprintf('-------------------- Starting Classifier %d/%d------------------\n', t, total_times)
        ind = randperm(size(X_neg, 1));
        len_pos_selected = round(size(X_neg, 1)/total_times);
        if len_pos_selected < length(y_pos)
            len_pos_selected = length(y_pos);
        end
        index = ind(1: len_pos_selected);
        
        [W_t, b_t] = base_classifier([X_pos; X_neg(index, :)], [y_pos; y_neg(index)] , conf);
        y_tst = y_tst + tst*W_t + b_t;
    end

    y_tst = y_tst / total_times;
    pred_tst = sign(y_tst); 

    [tst_auc, ~] = ROC_fuc(y_tst, tstLabel, 1, -1);
    
    tst_acc = (1.0*sum(pred_tst == tstLabel))/length(pred_tst);
    tst_TP = sum(pred_tst(tstLabel == 1) == 1)/sum(tstLabel == 1);
    tst_TN = sum(pred_tst(tstLabel == -1) == -1)/sum(tstLabel == -1);
    
    res.tst.acc = tst_acc;
    res.tst.AUC = tst_auc(1);
    res.tst.AP = (tst_TP + tst_TN)/2.0;
    res.tst.TP = tst_TP;
    res.tst.TN = tst_TN;
    res.tst.y_tst = y_tst;
end

function [W, b] = base_classifier(train, T, conf)
    dim = size(train, 2);
    
    pos_ind = T == 1;
    neg_ind = T == -1;
    X_pos = train(pos_ind, :);      y_pos =  ones(sum(pos_ind), 1);
    X_neg = train(neg_ind, :);      y_neg = -ones(sum(neg_ind), 1);
    lenNeg = size(X_neg, 1);
    lenPos = size(X_pos, 1);
    
    nb_epoch = conf.nb_epoch;
    w_decay = conf.w_decay;
    lr = conf.lr;
    
    W = rand(dim, 1);
    b = 0;
    L_old = 0;
    
    neg_y = abs(X_neg*W + b);
    tmp_neg_y = [neg_y, [1:lenNeg]'];
    tmp_neg_y = sortrows(tmp_neg_y, 1);
    ind_neg_s = tmp_neg_y(1:lenPos, 2);
    count_end = 0;
    for epoch = 1 : nb_epoch        
        X = [X_pos; X_neg(ind_neg_s, :)];
        t = [y_pos; y_neg(ind_neg_s, :)];        
        y = X*W + b;
        L = sum((y - t).^2) + w_decay*sum(W.^2);
        if abs(L - L_old) < 1e-6
            count_end = count_end + 1;
            if count_end > 2
                break;
            end
        else
            count_end = 0;
        end
        if mod(epoch, 100) == 0
            tmp_X = [X_pos; X_neg];
            tmp_t = [y_pos; y_neg];
            y_trn = tmp_X*W + b;
            pred_trn = sign(y_trn);
            acc = (1.0*sum(pred_trn == tmp_t))/length(pred_trn);
            TP = sum(pred_trn(tmp_t == 1) == 1)/sum(tmp_t == 1);
            TN = sum(pred_trn(tmp_t == -1) == -1)/sum(tmp_t == -1);
            fprintf('Cls_Ns+P %d/%d --- Loss = %f: Acc = %.2f%% -- TP = %.2f%% -- TN = %.2f%% ===\n', epoch, nb_epoch, L, acc*100, TP*100, TN*100);
        end
        L_old = L;
        W = W - lr*(mean(repmat((y - t), 1, size(X, 2)).*X, 1)' + w_decay*W);
        b = b - lr*mean(y - t);
        
        % Re-select Nearest Negative samples
        neg_y = abs(X_neg*W + b);
        tmp_neg_y = [neg_y, [1:lenNeg]'];
        tmp_neg_y = sortrows(tmp_neg_y, 1);
        ind_neg_s = tmp_neg_y(1:lenPos, 2);
    end
end

