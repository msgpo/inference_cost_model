hbm_efficiency = 0.5
comp_efficiency = 0.5
comm_efficiency = 0.5


def compute_model_memory_limit(num_layers, h_dim, b_type):
    return h_dim * h_dim * 12 * b_type * num_layers


def compute_prompt_time_stage(seq_in, batch_size, memory_bandwidth, gpu_flops, num_layers, tp_degree=1, h_dim=12288,
                              b_type=2) -> float:
    layer_scan_time = 12 * h_dim * h_dim * b_type / tp_degree / memory_bandwidth / hbm_efficiency
    layer_compute_time = 24 * batch_size * seq_in * h_dim * h_dim / tp_degree / gpu_flops / comp_efficiency
    return (layer_scan_time + layer_compute_time) * num_layers


def compute_token_step_time_stage(batch_size, memory_bandwidth, gpu_flops, num_layers, tp_degree=1, h_dim=12288,
                                  b_type=2) -> float:
    layer_scan_time = 12 * h_dim * h_dim * b_type / tp_degree / memory_bandwidth / hbm_efficiency
    layer_compute_time = 24 * batch_size * h_dim * h_dim / tp_degree / gpu_flops / comp_efficiency
    return (layer_scan_time + layer_compute_time) * num_layers


def communicate_prompt_time_stage(seq_in, batch_size, num_layers, tp_degree, delay, bandwidth,
                                  h_dim=12288, b_type=2) -> float:
    step_time = 0
    for i in range(tp_degree):
        current_step = 0
        for j in range(tp_degree):
            if i != j:
                current_step += (delay+batch_size*seq_in*h_dim*b_type/tp_degree/bandwidth)
        step_time = max(step_time, current_step)
    result = step_time * 4 * num_layers / comm_efficiency
    return result


def communicate_token_step_time_stage(batch_size, num_layers, tp_degree, delay, bandwidth,
                                      h_dim=12288, b_type=2) -> float:
    step_time = 0
    for i in range(tp_degree):
        current_step = 0
        for j in range(tp_degree):
            if i != j:
                current_step += (delay + batch_size*h_dim*b_type/tp_degree/bandwidth)
        step_time = max(step_time, current_step)
    result = step_time * 4 * num_layers / comm_efficiency
    return result


def end_to_end_time(batch_size, seq_in, seq_out, memory_bandwidth, gpu_flops, delay, bandwidth, num_layers, tp_degree,
                    h_dim=12288, b_type=2) -> float:
    prompt_comp_time = compute_prompt_time_stage(seq_in, batch_size, memory_bandwidth, gpu_flops, num_layers, tp_degree,
                                                 h_dim, b_type)
    prompt_comm_time = communicate_prompt_time_stage(seq_in, batch_size, num_layers, tp_degree, delay, bandwidth,
                                                     h_dim, b_type)

    print(f"Prompt phase time: {prompt_comp_time + prompt_comm_time}s (compute: {prompt_comp_time}s, "
          f"communication: {prompt_comm_time}s)")

    token_comp_time = compute_token_step_time_stage(batch_size, memory_bandwidth, gpu_flops, num_layers, tp_degree,
                                                    h_dim, b_type)
    token_comm_time = communicate_token_step_time_stage(batch_size, num_layers, tp_degree, delay, bandwidth,
                                                        h_dim, b_type)
    print(f"Token phase per-token time: {token_comp_time + token_comm_time}s (compute: {token_comp_time}s, "
          f"communication: {token_comm_time}s)")

    total_time = prompt_comp_time + prompt_comm_time + seq_out * (token_comp_time + token_comm_time)
    print(f"Total time: {total_time}s")
    print(f"Throughput: {batch_size*seq_out/total_time}")
    return total_time

