import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

def membership_values(x, K):
    membership_values_output = []
    n_features = len(K)
    for sample in range(len(x)):
        membership_i = []
        # for every feature in the sample
        i = 0
        for f in range(n_features):
          tmp=[]
          # for every subspace of that feature
          for j in range(K[i]):
            if n_features == 1:
                a = (j)/(K[i]-1)
                b = 1/(K[i]-1)
                value = 1 - abs(x[sample]-a)/b
                tmp.append(max(0,value))
            else:
                a = (j)/(K[i]-1)
                b = 1/(K[i]-1)
                value = 1 - abs(x[sample][f]-a)/b
                tmp.append(max(0,value))
          # check for the next k_i
          i+=1
          membership_i.append(tmp)
        # when you finish to append you will get the membership of every characteristic of that sample  
        membership_values_output.append(membership_i)
    return membership_values_output

# Generate all posible combinations for all the feature K_i
def combinations_for_k(k:np.ndarray):
    # List to store combinations
    output = [] 
    num_features = len(k)
    # Calculate total number of combinations
    num_combinations = np.prod(k)

    # Iterate from 0 to num_combinations - 1
    for i in range(num_combinations):
        # Store current combination
        combination = []
        temp_i = i

        # For each K_i
        for j in range(num_features):
            comb_value = temp_i % k[j]
            combination.append(comb_value)
            # Update i for next K_i rounding it to nearest whole number
            temp_i = temp_i // k[j]
    
        output.append(combination)

    return output

#Compute every degree of compatibility with every combination. There are k_1*k2*...k_n different combinations
def degree_of_compatibility(membership_values, combinations):
    compatibilities = []

    for i in range(len(membership_values)):
        combination = []
        for c in combinations:
            mask = []
            for z in range(len(combinations[0])):
                if len(membership_values[i])==1:
                    mask.append(membership_values[i][0][c[z]])
                else:
                    mask.append(membership_values[i][z][c[z]])
            mask=np.array(mask)
            combination.append(np.prod(mask))
        compatibilities.append(combination)
    return compatibilities

"""
b = Real number for every fuzzy if-then rule
"""
def heuristic_method_of_b(weight_matrix, target_vals):
    b_values_list = [] # Results list
    transposed_matrix = np.transpose(weight_matrix) # Easier to iterate with columns

    for column in transposed_matrix:
        # Numerator operation: EW_j1..jn(x_p) * Y_p
        num_sum = np.dot(column, target_vals)

        # Denominator operation: EW_j1..jn(x_p)
        denom_sum = np.sum(column)

        # Calculate b
        if denom_sum != 0:
            b_values_list.append(num_sum/denom_sum)
        else:
            b_values_list.append(0)
    return np.array(b_values_list)

# Predict function
def predict(input_data, input_results, coeff_k, B, alpha):

    memb_values_input = membership_values(input_data, coeff_k)
    # Generate all possible combinations and calculate each degree of compatibility 
    possible_combinations = combinations_for_k(coeff_k)
    dc_values = degree_of_compatibility(memb_values_input, possible_combinations)
    
    # Calculate weight for  every degree of compatibility 
    w = np.power(dc_values,alpha)
    # Heuristic method
    b = heuristic_method_of_b(w, input_results)
    # Compute each membership value
    memb_values_b = membership_values(b, B)

    predict_results = []
    primary_memb_table = np.zeros((coeff_k[0], coeff_k[1])) # coeff_k[0] = 3, coeff_k[1] = 3
    secondary_memb_table = np.zeros((coeff_k[0], coeff_k[1]))

    # Iterate every possible combination of characteristics
    for i, combination in enumerate(possible_combinations):
        primary_idx = np.argmax(memb_values_b[i]) # Search index with bigger membership value
        primary_memb_table[combination[0]][combination[1]] = primary_idx 
        
        secondary_values = memb_values_b[i][0]
        # Override bigger membership value to find the second one
        secondary_values[np.argmax(memb_values_b[i])] = -np.inf 
        # Search for the second one
        second_max_idx = np.argmax(secondary_values) 
        secondary_memb_table[combination[0]][combination[1]] = second_max_idx
            
    """ 
    Realize the prediction using the tables and
    degree of compatibility, iterating each sample
    """
    for sample in dc_values: 
        # Sample = Vector with degrees of compatibility of that sample for every rule
        pred = sum(sample * primary_memb_table.flatten()) / sum(sample)
        predict_results.append(pred)
    return predict_results, primary_memb_table, secondary_memb_table

def rules(matrix, r):
    t = r #['S','MS','M','ML','L']
    
    a=0
    for i in range(len(matrix)):

        for j in range(len(matrix[0])):
            print("Rule_",a," IF x1 = ",t[i]," AND x2 =", t[j]," THEN ", t[int(matrix[i][j])])
            a+=1

def example_2():
   
    # Example 2
    data = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    output = []
    for x in data:
        # Equation 12 in the paper
        output.append(0.2 * math.sin(2 * math.pi * x + math.pi / 4) + 0.5)

    # Add an outlier
    data.append(0.75)
    output.append(0.2)

    # Parameters
    alpha = 10
    K = [5]
    
    """ 
    Call the functions to know:
    membership values
    all possible combinations
    each degree of compatibility with each weight
    and b value for each rule
    """
    membership_val = membership_values(data, K)
    c = combinations_for_k(K)
    dc = degree_of_compatibility(membership_val, c)
    w = np.power(dc, alpha)
    b = heuristic_method_of_b(w, output) 

    prediction = []
    n_samples = len(w)

    # Make predictions
    for i in range(n_samples):
        numerator = [] 
        denominator = []

        for j in range(len(w[0])):
            numerator.append(dc[i][j] * b[j])
            denominator.append(dc[i][j])

        numerator = np.array(numerator)
        denominator = np.array(denominator)
        value = np.sum(numerator) / np.sum(denominator)
        prediction.append(value)
    
    # Visualización de los resultados
    plt.figure()
    plt.plot(data, prediction, label='Predicted')
    plt.scatter(data, output, color='red', label='Real')
    plt.title("Example 2: Comparison between Predicted and Real Data")
    plt.xlabel("Input Data")
    plt.ylabel("Output Data")
    plt.legend()
    plt.show()  
   

def example4_2():
    
    # Obtain data from paper "A Fuzzy-Logic-Based Approach to Qualitative Modeling"
    x1 = [1.40, 4.28, 1.18, 1.96, 1.85, 3.66, 3.64, 4.51, 3.77, 4.84, 1.05, 4.51, 1.84, 1.67, 2.03, 3.62, 1.67, 3.38, 2.83, 1.48, 3.37, 2.84, 1.19, 4.10, 1.65, 2.00, 2.71, 1.78, 3.61, 2.24, 1.81, 4.85, 3.41, 1.38, 2.46, 2.66, 4.44, 3.11, 4.47, 1.35, 1.24, 2.81, 1.92, 4.61, 3.04, 4.82, 2.58, 4.14, 4.35, 2.22]
    x2 = [1.80, 4.96, 4.29, 1.90, 1.43, 1.60, 2.14, 1.52, 1.45, 4.32, 2.55, 1.37, 4.43, 2.81, 1.88, 1.95, 2.23, 3.70, 1.77, 4.44, 2.13, 1.24, 1.53, 1.71, 1.38, 2.06, 4.13, 1.11, 2.27, 3.74, 3.18, 4.66, 3.88, 2.55, 2.12, 4.42, 4.71, 1.06, 3.66, 1.76, 1.41, 1.35, 4.25, 2.68, 4.97, 3.80, 1.97, 4.76, 3.90, 1.35]
    x1 = np.array(x1)
    x2 = np.array(x2)

    
    input4_2 = np.power((1 + (x1**-2) + (x2**-1.5)),2)
    
    # Normalize data 
    data = np.column_stack((x1,x2,input4_2))
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
                 
    # Separate columns for inputs and outputs
    x4_2 = data_scaled[:, :-1]
    y4_2 = data_scaled[:, -1]
    
    #Define system parameters
    K = [5,5]
    B = [5]
    alpha = 5
    
    values_K= [[2,2],[3,3],[4,4],[5,5]]
    values_alpha = [0.1,0.5,1,5,10,50,100]
    
    tabla = []
    
    for alpha4_2 in values_alpha: 
        row = [alpha4_2]
        
        for k4_2 in values_K:
            
            membership_val4_2 = membership_values(x4_2, k4_2)
            c4_2 = combinations_for_k(k4_2)
            dc4_2 = degree_of_compatibility(membership_val4_2, c4_2)
            w4_2 = np.power(dc4_2, alpha4_2)
            b4_2 = heuristic_method_of_b(w4_2, y4_2) 

            prediction4_2 = []
            n_samples = len(w4_2)
                
            # Predictions
            for i in range(n_samples):
                numerator = [] 
                denominator = []

                for j in range(len(w4_2[0])):
                    numerator.append(dc4_2[i][j] * b4_2[j])
                    denominator.append(dc4_2[i][j])

                numerator = np.array(numerator)
                denominator = np.array(denominator)
                val = np.sum(numerator) / np.sum(denominator)
                prediction4_2.append(val)
                
            valueP = []
            for l in range(len(y4_2)):
                error = (prediction4_2[l] - y4_2[l])**2
                valueP.append(error/2)
            valueProm = np.array(valueP)
            valueProm = np.sum(valueProm)    
            row.append(valueProm)
        tabla.append(row)
        
    performanceTable = pd.DataFrame(tabla, columns=['alpha', 'K = 2','K = 3','K = 4','K = 5'])
        
    print(performanceTable)
    
    predictions, first, second = predict(x4_2, y4_2, K, B, alpha)
    
    #Assing the labels
    labels = ['S','MS','M','ML','L']
    first = first.transpose() #Just to sort the graph
    second = second.transpose() #Just to sort the graph
    
    #Present the Main Rule Table
    mainMatrix = first.astype(int)
    labelMain = np.array(labels)[mainMatrix]
    MainTable = sns.heatmap(first, annot=labelMain, cbar= None, fmt='', cmap='viridis')
    MainTable.set_xticklabels(labels)
    MainTable.set_yticklabels(labels)
    MainTable.invert_yaxis()
    plt.title('Example 4.2: Main Rule Table')
    plt.show()
    
    #Present the Secondary Rule Table
    secondaryMatrix = second.astype(int)
    labelSecond = np.array(labels)[secondaryMatrix]
    SecondaryTable = sns.heatmap(second, annot=labelSecond, cbar= None, fmt='', cmap='viridis')
    # Change the axis labels to show the original labels.
    SecondaryTable.set_xticklabels(labels)
    SecondaryTable.set_yticklabels(labels)
    SecondaryTable.invert_yaxis()
    plt.title('Example 4.2 Secondary Rule Table')
    plt.show()
            

def Proposed_problem(path_data):
    # Inputs of our problem
    data = pd.read_csv(path_data).values
    
    # Normalize data 
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
                 
    # Separate input and output columns
    x = data_scaled[:, :-1]
    y = data_scaled[:, -1]
    
    #Define the system parameters
    K = [5,5]
    B = [5]
    alpha = 100
    
    predictions, first, second = predict(x, y, K, B, alpha)
    print("Predicted Results:", predictions)
    rules(first, ['Very Low','Low','Medium', 'High', 'Very High'])

    #Assing the labels
    labels = ['Very Low','Low','Medium', 'High', 'Very High']
    first = first.transpose() #Just to sort the graph
    second = second.transpose() #Just to sort the graph
    
    #Present the Main Rule Table
    mainMatrix = first.astype(int)
    labelMain = np.array(labels)[mainMatrix]
    MainTable = sns.heatmap(first, annot=labelMain, cbar= None, fmt='', cmap='viridis')
    MainTable.set_xticklabels(labels)
    MainTable.set_yticklabels(labels)
    MainTable.invert_yaxis()
    plt.title('Heart attack: Main Rule Table')
    plt.show()
    
    #Present the Secondary Rule Table
    secondaryMatrix = second.astype(int)
    labelSecond = np.array(labels)[secondaryMatrix]
    SecondaryTable = sns.heatmap(second, annot=labelSecond, cbar= None, fmt='', cmap='viridis')
    SecondaryTable.set_xticklabels(labels)
    SecondaryTable.set_yticklabels(labels)
    SecondaryTable.invert_yaxis()
    plt.title('Heart attack: Secondary Rule Table')
    plt.show()
    
    
    membership_val = membership_values(x, K)
    c = combinations_for_k(K)
    dc = degree_of_compatibility(membership_val, c)
    w = np.power(dc, alpha)
    b = heuristic_method_of_b(w, y) 

    prediction = []
    n_samples = len(w)
    
    for i in range(n_samples):
        numerator = [] 
        denominator = []

        for j in range(len(w[0])):
            numerator.append(dc[i][j] * b[j])
            denominator.append(dc[i][j])

        numerator = np.array(numerator)
        denominator = np.array(denominator)
        value = np.sum(numerator) / np.sum(denominator)
        prediction.append(value)

    # Visualize the results
    plt.figure()
    plt.plot(range(len(prediction)), prediction, label='Predicted')
    plt.scatter(range(len(y)), y, color='red', label='Real')
    plt.title("Heart attack: Comparison between Predicted and Real Data")
    plt.xlabel("Sample index")
    plt.ylabel("Heart Risk")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    print("Papers examples")
    example_2()
    example4_2()
    
    print("Our proposed problem")
    csv_file_path = os.getenv('CSV_FILE_PATH', 'data.csv') #Change data here
    Proposed_problem(csv_file_path)
