function lip_prod = split_and_solve(split_W, mode, verbose, parallel, ...
    split_net_dims, network)
    % Compute Lipschitz constant as product of constants from subnetworks
    %
    % params:
    %   weights: cell           - weights of trained neural network
    %   net_dims: list of ints  - neural network layer dimensions
    %   split_amount: int       - number of weights in each subnetwork
    %   num_workers: int        - number of workers for parallelization
    %   alpha: float            - lower sector bound
    %   mode: str               - which Mahyar formulation to use
    %
    % returns:
    %   lip_prod: float     - Lipschitz constant of neural network

    num_splits = size(split_W, 2);

    lip_prod = 1;
    parfor (k = 1:num_splits, num_workers)
        curr_weights = split_W{k};
        curr_net_dims = split_net_dims{k};

        if size(curr_weights, 2) > 1
            Lf_reduced_piece = mahyar_lip_multi(curr_weights, curr_net_dims, ...
                verbose, -1, -1, alpha, mode, 0);

        % if there is only one matrix in this layer, just
        % multiply by the norm of that matrix
        else
            Lf_reduced_piece = norm(curr_weights{1}, 2);
        end

        % update product
        lip_prod = lip_prod * Lf_reduced_piece;
    end

end