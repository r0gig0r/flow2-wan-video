def find_step_index_percent(sigmas, timestep):
    matched_index = (sigmas == timestep[0]).nonzero()
    
    if len(matched_index) > 0:
        index = matched_index.item()
    else:
        index = 0
        for i in range(len(sigmas) - 1):
            if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                index = i
                break
            
    percent = index / (len(sigmas) - 1)

    return (index, percent)