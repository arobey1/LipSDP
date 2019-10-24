function W = create_weights(net_dims, weight_type)
    % Create cell of weights for neural network with given dimensions
    %
    % params:
    %   * net_dims: list of ints    - dimensions of neural network
    %   * weights_type: str         - type of weights - in ['rand', 'ones']
    %
    % returns:
    %   W: cell - weights of neural network

    num_layers = length(net_dims)-1;

    for i = 1:num_layers
        if strcmp(weight_type, 'ones')
            
            W{i} = ones(net_dims(i+1), net_dims(i));
            
        elseif strcmp(weight_type, 'rand')
            
            W{i} = (1 / sqrt(num_layers)) * randn(net_dims(i+1), net_dims(i));
            
        else
            error('[ERROR]: Please use weight_type in ["ones", "rand"]\n');
        end
       
    end

end