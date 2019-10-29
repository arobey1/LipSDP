function input_num = cap_input(input_num, N, type)
    % Caps number of a quantity (rand_num_neurons or num_dec_vars) to 
    % N choose 2 and prints information to user
    %
    % params:
    %   * input_num: int - input quantity: rand_num_neurons or num_dec_vars
    %   * N: int         - total number of hideen neurons in neural network
    %   * type: str      - name of input_num to print to user
    %
    % returns:
    %   * input_num: int - capped input number if quantity is over limit
    %                      otherwise, the original quantity is returned
    % ---------------------------------------------------------------------

    if input_num > nchoosek(N, 2)
        fprintf('[INFO]: Capping number of %s to %d.\n', type, nchoosek(N, 2))
        fprintf('[INFO]: Your network has %d hidden neurons and this\n', N);
        fprintf('[INFO]: only allows for (%d choose 2) = %d %s.\n', ...
            N, nchoosek(N, 2), type); 
        input_num = nchoosek(N, 2); 
    end

end