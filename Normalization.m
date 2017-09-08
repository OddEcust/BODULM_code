function res = Normalization(B, type)
% type = 1 : min_max --> (x-min)/(max-min) ==> [0, 1]
% type = 2 : mean_std --> (x-mean)/std  ==> N(0, 1)
%
    if type == 1
        res = min_max(B);
    elseif type == 2
        res = mean_std(B);
    else
        res = min_max(B);
    end
end

function res = min_max(B)
    dim = length(B(1,:)) ;
    res = [] ;
    for i = 1 : dim
        minX = min(B(:,i)) ;
        maxX = max(B(:,i)) ;
        res(:,i) = (B(:,i) - minX)./(maxX - minX+ 0.000001) ;
    end
end

function res = mean_std(B)
    dim = length(B(1,:)) ;
    res = [] ;
    for i = 1 : dim
        B_mean = mean(B(:,i)) ;
        B_std = std(B(:,i)) ;
        res(:,i) = (B(:,i) - B_mean)./(B_std + 0.00001);
    end
end