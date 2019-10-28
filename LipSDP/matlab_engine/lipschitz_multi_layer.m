function L = lipschitz_multi_layer(weights, mode, verbose, rand_num_neurons, ...
    net_dims, network)
    % Computes Lipschitz constant of NN using LipSDP formulation
    % mode parameter is used to select which formulation of LipSDP to use
    %
    % params:
    %   * weights: cell          - weights of neural network in cell array
    %   * mode: str              - LipSDP formulation in ['network',
    %                             'neuron','layer','network-rand']
    %   * verbose: logical       - if true, prints CVX output from solve
    %   * rand_num_neurons: int  - num of neurons to couple in LipSDP-Neuron-rand
    %   * net_dims: list of ints - dimensions of layers in neural net
    %   * network: struct        - data describing neural network
    %       - fields:
    %           (1) alpha: float            - slope-restricted lower bound
    %           (2) beta: float             - slope-restricted upper bound
    %           (3) weight_path: str        - path of saved weights of NN
    %                                         
    % returns:
    %   * L: float - Lipschitz constant of neural network
    % ---------------------------------------------------------------------

    warning('off', 'all');

    % set verbosity of CVX using verbose flag
    if verbose == true
        cvx_begin sdp
    else
        cvx_begin sdp quiet
    end

    cvx_solver mosek
    variable L_sq nonnegative   % optval will be square of Lipschitz const

    % extract neural network parameters
    alpha = network.alpha;
    beta = network.beta;
    N = sum(net_dims(2:end-1));     % total number of hidden neurons

    % LipSDP-Network - one variable for each of the (N choose 2) neurons in
    % the network to parameterize T matrix.  This mode has complexity O(N^2)
    if strcmp(mode, 'network')
        
        variable D(N, 1) nonnegative
        variable zeta(N * (N-1) / 2, 1) nonnegative

        id = eye(N);
        T = diag(D);
        C = nchoosek(1:N, 2);
        E = id(:, C(:, 1)) - id(:, C(:, 2));
        T = T + E * diag(zeta) * E';
        Q = [-2 * alpha * beta * T, (alpha+beta) * T;
             (alpha + beta) * T, -2 * T];

    % Repeated-rand mode: uses repeated nonlinearities with a random subset
    % of coupled neurons from the entire set of N choose 2 total neurons
    elseif strcmp(mode, 'network-rand')
        
        % cap number of random neurons
        if rand_num_neurons > nchoosek(N, 2)
            fprintf('[INFO]: Capping number of randomly chosen neurons to %d.\n', nchoosek(N, 2))
            fprintf('You specified %d neurons.\n', rand_num_neurons);
            rand_num_neurons = nchoosek(N, 2); 
        end

        variable D(N, 1) nonnegative
        variable zeta(rand_num_neurons, 1) nonnegative

        id = eye(N);
        T = diag(D);
        C = nchoosek(1:N, 2);

        % take a random subset of neurons to couple
        k = randperm(size(C, 1));
        C = C(k(1:rand_num_neurons), :);

        % form T matrix using these randomly chosen neurons
        E = id(:, C(:, 1)) - id(:, C(:, 2));
        T = T + E * diag(zeta) * E';
        Q = [-2 * alpha * beta * T, (alpha + beta) * T; 
             (alpha + beta) * T, -2 * T];

    % LipSDP-Neuron - one CVX variable per hidden neuron in the network to 
    % parameterize T matrix.  This mode has complexity O(N).
    elseif strcmp(mode, 'neuron')

        variable D(N, 1) nonnegative
        Q = [-2 * alpha * beta * diag(D), (alpha + beta) * diag(D);
             (alpha + beta) * diag(D), -2 * diag(D)];

    % LipSDP-Layer - one CVX variable per hidden hidden layer in the
    % network to parameterize T matrix.  This mode has complexity O(m)
    % where m is the number of hidden layers
    elseif strcmp(mode, 'layer')

        n_hid = length(net_dims) - 2;
        variable D(n_hid, 1) nonnegative

        for i = 1:n_hid
            identities{i} = D(i) * eye(net_dims(i+1));
        end

        D_mat = blkdiag(identities{:});
        Q = [-2 * alpha * beta * D_mat, (alpha + beta) * D_mat;
             (alpha + beta) * D_mat, -2 * D_mat];

    % If mode is not valid, raise error
    else
        error_msg = '[ERROR]: formulation must be in ["neuron", "network", "layer", "network-rand"]\n%s';
        error_info = sprintf(' --> You supplied formulation == %s', mode);
        error(error_msg, error_info);
       
    end

    % Create A term in Lipschitz formulation
    first_weights = blkdiag(weights{1:end-1});
    zeros_col = zeros(size(first_weights, 1), size(weights{end}, 2));
    A = horzcat(first_weights, zeros_col);

    % Create B term in Lipschitz formulation
    eyes = eye(size(A, 1));
    init_col = zeros(size(eyes, 1), net_dims(1));
    B = horzcat(init_col, eyes);

    % Stack A and B matrices
    A_on_B = vertcat(A, B);

    % Create M matrix encoding Lipschitz constant
    weight_term = -1 * weights{end}' * weights{end};
    middle_zeros = zeros(sum(net_dims(2 : end - 2)), sum(net_dims(2 : end - 2)));
    lower_right = blkdiag(middle_zeros, weight_term);
    upper_left = L_sq * eye(net_dims(1));

    % M = blkdiag(upper_left, lower_right);
    M = cvx(zeros(size(upper_left, 1) + size(lower_right, 1), size(upper_left, 2) + size(lower_right, 2)));
    M(1:size(upper_left, 1), 1:size(upper_left, 1)) = upper_left;
    M(size(upper_left, 1) + 1:end, size(upper_left, 2) + 1:end) = lower_right;

    % Solve optimizaiton problem - minimize squared Lipschitz constant
    minimize L_sq

    % LMI for minimization problem
    subject to
        (A_on_B' * Q * A_on_B) - M <= 0;
    cvx_end

    L = sqrt(L_sq);

end
