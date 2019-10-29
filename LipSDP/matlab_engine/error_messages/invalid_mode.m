function [] = invalid_mode(mode)
    % Error message for invalid mode - should already be caught in Python
    %
    % params:
    %   * mode: str - formulation for LipSDP supplied by user
    % ---------------------------------------------------------------------

    error_msg = '[ERROR]: formulation must be in ["neuron", "network", "layer", "network-rand", "network-dec-vars"]\n%s';
    error_info = sprintf('[ERROR]: You supplied formulation = %s', mode);
    error(error_msg, error_info);
    
end