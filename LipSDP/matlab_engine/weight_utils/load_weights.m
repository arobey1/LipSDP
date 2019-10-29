function [weights, net_dims] = load_weights(path)
    % Load weights from given path and extract network dimensions
    % 
    % params:
    %   * path: str - path of saved neural network weights
    %
    % returns:
    %   * weights: cell          - loaded weights of neural network
    %   * net_dims: list of ints - dimensions of each layer in network
    % ---------------------------------------------------------------------

    % load weights from path
    weight_dict = load(path);
    weights = weight_dict.weights;
    
    % extract network dimensions from weights
    net_dims = size(weights{1}, 2);
    for i = 1:length(weights)
        net_dims = [net_dims, size(weights{i}, 1)];
    end

end