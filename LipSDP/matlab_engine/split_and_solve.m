function lip_prod = split_and_solve(split_W, split_net_dims, lip_params, network)
    % Compute Lipschitz constant as product of constants from subnetworks
    % Can be run in parallel by specifying parallel and num_workers flags
    % in lip_params struct
    %
    % params:
    %   * split_W: cell         - weights for each subnetwork
    %   * split_net_dims: cell  - dimensions of layers in each subnetwork
    %   * network: struct       - data describing neural network
    %       - fields:
    %           (1) alpha: float            - slope-restricted lower bound
    %           (3) beta: float             - slope-restricted upper bound
    %           (3) net_dims: list of ints  - dimensions of NN
    %           (4) weight_path: str        - path of saved weights of NN
    %   * lip_params: struct    - parameters for LipSDP
    %       - fields:
    %           (1) formulation: str    - LipSDP formulation to use
    %           (2) split: logical      - if true, use splitting 
    %           (3) parallel: logical   - if true, parallelize splitting
    %           (4) verbose: logical    - if true, print CVX output
    %           (5) split_size: int     - size of subnetwork for splitting
    %           (6) num_neurons: int    - number of neurons to couple in
    %                                     LipSDP-Neuron-rand mode
    %           (7) num_workers: int    - number of workers for parallel-
    %                                     ization of splitting formulations
    %
    % returns:
    %   * lip_prod: float - Lipschitz constant found by splitting network
    %                       into subnetworks and solving LipSDP for each
    %                       piece
    % ---------------------------------------------------------------------

    % number of subnetworks after splitting
    num_splits = size(split_W, 2);
    
    % unpack variables from lip_params
    mode = lip_params.formulation;
    verbose = lip_params.verbose;
    rand_num_neurons = lip_params.num_neurons;

    % initialize Lipschitz constant of network
    lip_prod = 1;
    
    % for loop is parallelizable 
    if lip_params.parallel
        parpool('local', lip_params.num_workers);
    end
    
    parfor (k = 1:num_splits, lip_params.num_workers)
        curr_weights = split_W{k};
        curr_net_dims = split_net_dims{k};

        if size(curr_weights, 2) > 1
            Lf_reduced_piece = lipschitz_multi_layer(curr_weights, mode, ...
                verbose, rand_num_neurons, curr_net_dims, network);

        % if there is only one matrix in this layer, just
        % multiply by the norm of that matrix
        else
            Lf_reduced_piece = norm(curr_weights{1}, 2);
        end

        % update product
        lip_prod = lip_prod * Lf_reduced_piece;
    end

end