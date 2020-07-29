import numpy as np

def reweight(inputs, weight_dict):
    '''

    Example: Cluster weather data (beam radiation and temperature) and
    electricity demand
    Inputs are 1-dimensional arrays with 8760 entries (or no_days_input*24)


    :param inputs: dataframe,
        Dataframe contains the input data

    :param weight_dict: dicts,
        Dicts with following structure:
        {'input_name':
            'f_weight': 0.2}}
        sum(f_weight) = 1
    :return:
        inputsScaled: array,
        Reweighted input data

    '''

    cum_weights = 0  # Cumulated weighting factors
    for key in weight_dict:
        #input_dict[key].update(values=np.loadtxt(input_dict[key]['file_path']))
        cum_weights = cum_weights + weight_dict[key]['f_weight']
    assert cum_weights == 1.0, "Sum of weighting factors must equal 1."

    #length_input = len(input_dict[list(input_dict.keys())[0]]['values'])
    #no_days_input = int(length_input / 24)  # Number of days in input files
    #isinstance(no_days_input, int)

    #inputs = []
    weights = []
    for key in weight_dict:
        #inputs.append(input_dict[key]['values'])
        weights.append(weight_dict[key]['f_weight'])

    # Manipulate inputs
    inputsTransformed = []
    inputsScaled = []
    inputsScaledTransformed = []

    for j in range(len(weights)):
        i = inputs[:,j]
        inputsScaled.append(weights[j] * (i - np.min(i)) / (np.max(i) - np.min(i)))
        #inputsTransformed.append(i.reshape((24, no_days_input), order="F"))
    #for i in inputsScaled:
        #inputsScaledTransformed.append(i.reshape((24, no_days_input), order="F"))

    #L = np.concatenate(tuple(inputsScaledTransformed))
    inputsScaled =np.transpose(inputsScaled)
    return inputsScaled
