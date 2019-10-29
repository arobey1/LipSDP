function [split_w, split_net_dims] = split_weights(weights, net_dims, split_amount)
    % Splits neural network into subnetworks of size {split_amount}
    %
    % params:
    %   * weights: cell             - weights of neural network
    %   * net_dims: list of ints    - dimensions of layers in network
    %   * split_amount: int         - size of each subnetwork
    %
    % returns:
    %   * split_w: cell         - weights of each subnetwork
    %   * split_net_dims: cell  - dimensions of each subnetwork
    % ---------------------------------------------------------------------

    % number of weights in neural network
    num_weights = size(weights, 2);
    
    split_w = {}; 
    split_net_dims = {};
    
    counter = 1;
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