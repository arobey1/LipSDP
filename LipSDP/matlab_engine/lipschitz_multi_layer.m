function L = lipschitz_multi_layer(weights, mode, verbose, num_rand_neurons, ...
    num_dec_vars, net_dims, network)
    % Computes Lipschitz constant of NN using LipSDP formulation
    % mode parameter is used to select which formulation of LipSDP to use
    %
    % params:
    %   * weights: cell          - weights of neural network in cell array
    %   * mode: str              - LipSDP formulation in ['network',
    %                             'neuron','layer','network-rand', 
    %                              'network-dec-vars']
    %   * verbose: logical       - if true, prints CVX output from solve
    %   * num_rand_neurons: int  - num of neurons to couple in 
    %                              LipSDP-Network-rand
    %   * num_dec_vars: int      - num of decision variables for
    %                              LipSDP-Network-Dec-Vars
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
    id = eye(N);

    % LipSDP-Network - one variable for each of the (N choose 2) neurons in
    % the network to parameterize T matrix.  This mode has complexity O(N^2)
    if strcmp(mode, 'network')
        
        variable D(N, 1) nonnegative
        variable zeta(nchoosek(N, 2), 1) nonnegative

        T = diag(D);
        C = nchoosek(1:N, 2);
        E = id(:, C(:, 1)) - id(:, C(:, 2));
        T = T + E * diag(zeta) * E';

    % LipSDP-Network-Rand uses repeated nonlinearities with a random subset
    % of coupled neurons from the entire set of N choose 2 total neurons
    elseif strcmp(mode, 'network-rand')
        
        % cap number of random neurons
        num_rand_neurons = cap_input(num_rand_neurons, N, 'randomly chosen neurons');

        variable D(N, 1) nonnegative
        variable zeta(num_rand_neurons, 1) nonnegative

        T = diag(D);
        C = nchoosek(1:N, 2);

        % take a random subset of neurons to couple
        k = randperm(size(C, 1));
        C = C(k(1:num_rand_neurons), :);

        % form T matrix using these randomly chosen neurons
        E = id(:, C(:, 1)) - id(:, C(:, 2));
        T = T + E * diag(zeta) * E';
        
    % LipSDP-Network-Dec-Vars - uses repeated nonlinearities with a
    % spcified number of decision variables spaced out equally
    elseif strcmp(mode, 'network-dec-vars')
        
        % cap number of decision variables
        num_dec_vars = cap_input(num_dec_vars, N, 'decision variables');
        
        variable D(N, 1) nonnegative
        
        T = diag(D);
        C = nchoosek(1:N, 2);
        
        % space out decision variables in couplings
        spacing = ceil(nchoosek(N, 2) / num_dec_vars);
        C = C(1:spacing:end, :);

        variable zeta(size(C, 1), 1) nonnegative

        % form T matrix using these randomly chosen neurons
        E = id(:, C(:, 1)) - id(:, C(:, 2));
        T = T + E * diag(zeta) * E';

    % LipSDP-Neuron - one CVX variable per hidden neuron in the network to 
    % parameterize T matrix.  This mode has complexity O(N).
    elseif strcmp(mode, 'neuron')

        variable D(N, 1) nonnegative
        T = diag(D);

    % LipSDP-Layer - one CVX variable per hidden hidden layer in the
    % network to parameterize T matrix.  This mode has complexity O(m)
    % where m is the number of hidden layers
    elseif strcmp(mode, 'layer')

        n_hid = length(net_dims) - 2;
        variable D(n_hid, 1) nonnegative

        for i = 1:n_hid
            identities{i} = D(i) * eye(net_dims(i+1));
        end

        T = blkdiag(identities{:});

    % If mode is not valid, raise error
    else
        invalid_mode(mode);
       
    end
    
    % Create Q matrix, which is parameterized by T, which in turn depends
    % on the chosen LipSDP formulation 
    Q = [-2 * alpha * beta * T, (alpha + beta) * T; 
             (alpha + beta) * T, -2 * T];

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
