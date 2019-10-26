function [split_w, split_net_dims] = split_weights(weights, net_dims, split_amount)
    % 

    num_weights = size(weights, 2);
    counter = 1;
    split_w = {}; split_net_dims = {};
    
    for k = 1:split_amount:num_weights
        
        % get ending index of split
        next_max_idx = k + split_amount - 1;
        
        % if we exceed the total number of weights, cut this one short
        if next_max_idx > num_weights
            next_max_idx = num_weights;
        end
        
        % add split section of weights to cell
        split_w{counter} = weights(k : next_max_idx);
        split_net_dims{counter} = net_dims(k : next_max_idx + 1);
        counter = counter + 1;
    end

end