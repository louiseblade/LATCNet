import torch

def group_parameters(model, verbose):
    """Groups model parameters for different weight decay settings."""

    conv_params = []
    dense_params = []
    no_decay_params = []
    A_params = []
    # Iterate through all named parameters
    for name, param in model.named_parameters():

        if not param.requires_grad :
            continue  # Skip non-trainable parameters

        # Prioritize BatchNorm parameters and biases
        if 'bn' in name or 'batchnorm' in name.lower():
            if verbose: print("No decay in batchnorm", name)
            no_decay_params.append(param)
            continue

        if '.bias' in name:
            if verbose: print("No decay in bias", name)
            no_decay_params.append(param)
            continue

        has_mlp = 'mlp.' in name or 'feedforward.' in name or 'fc_layer' in name


        if 'recurrent_block.linear_out.weight' in name or 'mlp_block.ffw_down.weight' in name:
            if verbose: print("decay in recurrent block", name)
            dense_params.append(param)
            continue

        if 'conv_block' in name:
            if verbose: print("decay in convolution", name)
            conv_params.append(param)
            continue

        if 'tcn_list' in name or "tcn_list_2" in name :
            if verbose: print("decay in temporal convolution", name)
            conv_params.append(param)
            continue

        # Assign Dense layers
        if 'fc_list' in name or 'slide_out_list' in name or 'final_layer' in name or has_mlp:
            if verbose: print("decay in dense/fully connected", name)
            dense_params.append(param)
            continue



        # Any remaining parameters
        if verbose: print("No decay in others", name)
        no_decay_params.append(param)

    # Combine all grouped parameters
    all_grouped_params = (
            conv_params +
            dense_params +
            no_decay_params +
            A_params
    )

    # Assert that all trainable parameters are grouped
    all_grouped_param_ids = set(id(p) for p in all_grouped_params)
    total_trainable_params = len([p for p in model.parameters() if p.requires_grad])
    assert len(all_grouped_param_ids) == total_trainable_params, \
        "Some parameters are assigned to multiple groups or not assigned."

    print("All parameters are correctly assigned to a single group.")
    return conv_params, dense_params, no_decay_params, A_params

def get_optimizer(model, lr=0.001, weight_decay=False, verbose=False):
    if not weight_decay:
        if verbose:
            print("No weight decay was applied")
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        if verbose:
            print("Sets up the optimizer with grouped parameters for different weight decays.")
        conv_params, dense_params, no_decay_params, a_param = \
            group_parameters(model, verbose)

        optimizer = torch.optim.Adam([
            {'params': conv_params, 'weight_decay': 0.009},
            {'params': dense_params, 'weight_decay': 0.5},
            {'params': no_decay_params, 'weight_decay': 0.0},

        ], lr=lr)

        return optimizer


