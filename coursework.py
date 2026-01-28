import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cleaning import remove_outliers
import seaborn as sns
from scipy.interpolate import PchipInterpolator as pchip_interpolate
from scipy.stats import pearsonr


# Load the data from the Excel file
df = pd.read_excel('Mean water.xlsx', header=1)

# Rename columns correctly
df.columns = ["Date", "Crakehill", "Skip Bridge", "Westwick", "Skelton", 
              "Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme", 
              "Unused_1", "Unused_2", "Notes"]

# Drop unnecessary columns and convert date to datetime
df_water = df[["Date", "Crakehill", "Skip Bridge", "Westwick", "Skelton"]].copy()
df_water["Date"] = pd.to_datetime(df_water["Date"], errors='coerce')

# Convert numeric columns to numeric types, forcing errors to NaN
for col in ["Crakehill", "Skip Bridge", "Westwick", "Skelton"]:
    df_water[col] = pd.to_numeric(df_water[col], errors='coerce')

#Plot original data individually against time
for col in ["Crakehill", "Skip Bridge", "Westwick", "Skelton"]:
   plt.figure(figsize=(10, 4))
   plt.scatter(df_water["Date"], df_water[col], label=f"{col} (Original)", alpha=0.6)
   plt.xlabel("Date")
   plt.ylabel("Flow (Cumecs)")
   plt.title(f"Mean Daily Flow Before Cleaning - {col}")
   plt.legend()
   plt.grid(True)
   plt.show()



'''---Cleaning river flow: Outlier removal---'''
for col in ["Crakehill", "Skip Bridge", "Westwick", "Skelton"]:
    mean_value = df_water[col].mean()
    std_dev = df_water[col].std()
    df_water[col] = remove_outliers(df_water[col].values, mean_value, std_dev)





# Plot cleaned data individually against time
for col in ["Crakehill", "Skip Bridge", "Westwick", "Skelton"]:
    plt.figure(figsize=(10, 4))
    plt.scatter(df_water["Date"], df_water[col], label=f"{col} (Cleaned)", alpha=0.6)
    plt.xlabel("Date")
    plt.ylabel("Flow (Cumecs)")
    plt.title(f"Mean Daily Flow After Cleaning - {col}")
    plt.legend()
    plt.grid(True)
    plt.show()


'''---------Interpolate missing values----------'''
for col in ["Crakehill", "Skip Bridge", "Westwick", "Skelton"]:
    mask = df_water[col].notna()
    
    # Ensure Date column has no NaN values by filling them with previous values
    df_water["Date"] = df_water["Date"].ffill()  # Forward fill missing dates
    
    # Convert Date to ordinal (integer format) for interpolation
    x = df_water["Date"][mask].map(pd.Timestamp.toordinal)
    y = df_water[col][mask]
    x_interp = df_water["Date"].map(pd.Timestamp.toordinal)
    
    # Perform PCHIP interpolation
    interpolator = pchip_interpolate(x, y)
    df_water[col] = interpolator(x_interp)




# Apply moving averages
window_size = 4  # Define window size for moving average
df_water.loc[:, ["Crakehill", "Skip Bridge", "Westwick"]] = df_water.loc[:, ["Crakehill", "Skip Bridge", "Westwick"]].rolling(window=window_size, min_periods=1).mean()

#Lag skelton values by 1 day
df_water["Skelton tommorow"] = df_water["Skelton"].shift(-1)
df_water["Predictor_1"] = df_water["Crakehill"] + df_water["Skip Bridge"] + df_water["Westwick"]
#df_water = df_water[["Date","Predictor_1", "Skelton"]]

# Round values to 3 decimal places
df_water = df_water.round(3)




# Save cleaned data to a new file
df_water.to_csv('Cleaned_Water_Data.txt', sep='\t', index=False)
print("Data cleaning complete. Cleaned file saved as 'Cleaned_Water_Data.csv'.")









#---- Process rainfall data------
df_rainfall = df[["Date", "Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme"]].copy()
df_rainfall["Date"] = pd.to_datetime(df_rainfall["Date"], errors='coerce')
for col in ["Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme"]:
    df_rainfall[col] = pd.to_numeric(df_rainfall[col], errors='coerce')


# Plot original data individually against time
for col in ["Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme"]:
    plt.figure(figsize=(10, 4))
    plt.scatter(df_rainfall["Date"], df_rainfall[col], label=f"{col} (Original)", alpha=0.6)
    plt.xlabel("Date")
    plt.ylabel("Flow (Cumecs)")
    plt.title(f"Mean Daily Flow Before Cleaning - {col}")
    plt.legend()
    plt.grid(True)
    plt.show()


'''------Cleaning rainfall data by removing averages-----'''
for col in ["Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme"]:
    mean_value = df_rainfall[col].mean()
    std_dev = df_rainfall[col].std()
    df_rainfall[col] = remove_outliers(df_rainfall[col], mean_value, std_dev)


# Plot cleaned data individually against time
for col in ["Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme"]:
    plt.figure(figsize=(10, 4))
    plt.scatter(df_rainfall["Date"], df_rainfall[col], label=f"{col} (Cleaned)", alpha=0.6)
    plt.xlabel("Date")
    plt.ylabel("Flow (Cumecs)")
    plt.title(f"Mean Daily Flow After Cleaning - {col}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Interpolate missing values in rainfall data using PCHIP
for col in ["Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme"]:
    mask = df_rainfall[col].notna()
    
    # Ensure Date column has no NaN values by filling them with previous values
    df_rainfall["Date"] = df_rainfall["Date"].ffill()  # Forward fill missing dates
    
    # Convert Date to ordinal (integer format) for interpolation
    x = df_rainfall["Date"][mask].map(pd.Timestamp.toordinal)
    y = df_rainfall[col][mask]
    x_interp = df_rainfall["Date"].map(pd.Timestamp.toordinal)
    
    # Perform PCHIP interpolation
    interpolator = pchip_interpolate(x, y)
    df_rainfall[col] = interpolator(x_interp)



'''------Apply moving averages to Rainfall data'''
df_rainfall.loc[:, ["Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme"]] = (df_rainfall.loc[:, ["Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme"]].rolling(window=window_size, min_periods=1).mean())
df_rainfall = df_rainfall.round(3)


'''--------Merge Water and Rainfall data----'''
df_cleaned = pd.merge(df_rainfall, df_water, on="Date", how="inner")
df_cleaned.columns = df_cleaned.columns.str.strip()
# Creating predictand and predictors
df_cleaned['Skelton Tomorrow'] = df_cleaned['Skelton'].shift(-1)
df_cleaned['Skelton - 1'] = df_cleaned['Skelton'].shift(1)
df_cleaned['Average Rainfall'] = df_cleaned[['Arkengarthdale', 'East Cowton', 'Malham Tarn', 'Snaizeholme']].mean(axis=1)
df_cleaned['Average River flow'] = df_cleaned[['Crakehill', 'Skip Bridge', 'Westwick', 'Skelton']].mean(axis=1)
df_cleaned['Average Rainfall -1 day'] = df_cleaned['Average Rainfall'].shift(1)
df_cleaned['Skelton + average Rain'] = df_cleaned['Skelton'] + df_cleaned['Average Rainfall']
# + skelton today


# Save cleaned data to a new file
df_cleaned.to_csv('Cleaned_Data.txt', sep='\t', index=False)



# Lagging Rainfall Data by 1 to 7 Days
df_lagged_data = df_cleaned.copy()

for lag in range(1, 7):
    for col in ["Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme", "Predictor_1", "Skelton"]:
        df_lagged_data[f"{col}_lag{lag}"] = df_cleaned[col].shift(lag)

df_rainfall = df_rainfall.round(3)
df_lagged_data = df_lagged_data.round(3)
df_lagged_data.to_csv('Lagged_Rainfall_Data.txt', sep='\t', index=False)



# Calculate correlation matrix
correlation_matrix_rain = df_lagged_data.copy().corr()


# Plot Heatmap of Skelton vs Rainfall Locations
correlation_data = df_cleaned
correlation_matrix = correlation_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Finding Predictors")
plt.show()



# Save cleaned data to a new file
df_rainfall.to_csv('Cleaned_Rainfall_Data.txt', sep='\t', index=False)
print("Data cleaning complete. Cleaned file saved as 'Cleaned_Rainfall_Data.csv'.")









# Creates MLP class with matrix multiplication

class MLP:
    def __init__(self,x,y,input_size,output_size, hidden_size, learning_rate, epochs):
        self.x = x
        self.y = y
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Weight initialization with uniform distribution
        self.weights_input_hidden = np.random.uniform(-2/input_size, 2/input_size, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-2/hidden_size, 2/hidden_size, (hidden_size, output_size))
    
        # Bias initialization to go between 1 and 0 for sigmoid activation
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    

    '''----------------------ACTIVATION FUNCTIONS----------------------'''
    #Sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #Sigmoid derivative function
    def sigmoid_derivative(self, node_sum):
        activation = self.sigmoid(node_sum)
        return activation * (1 - activation)
    
    #Tanh function
    def tanh(self, x):
        return np.tanh(x)
    
    #Tanh derivative function
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    #Relu function
    def relu(self, x):
        return np.maximum(0, x)
    
    #Relu derivative function
    def relu_derivative(self, x):
        return np.where(x <= 0, 0,
                        np.where(x > 0, 1, 0))
    
    '''----------------------Error FUNCTIONS----------------------'''
    #Mean Absolute error
    def absolute_error(self, y, y_pred):
        return np.mean(np.abs(y - y_pred))
    
    #Mean Absolute error derivative
    def absolute_error_derivative(self, y, output):
        return np.where(output > y, 1, -1) 

    
    #Mean squared error
    def mean_squared_error(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    #Mean squared error derivative
    def mean_squared_error_derivative(self, y, y_pred):
        return 2 * (y - y_pred) / len(y)
    

    #Root Mean squared error
    def root_mean_squared_error(self, y, y_pred):
        return np.sqrt(np.mean((y - y_pred) ** 2))
    
    #Root Mean squared error derivative
    def root_mean_squared_error_derivative(self, y, y_pred):
        return 2 * (y - y_pred) / len(y)
    
    '''---------------------- FORWARD and BACKWARD PROPAGATION----------------------'''
    def forward(self, x):
        #Input to hidden layer
       
        hidden_layer_sum = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_activation = self.sigmoid(hidden_layer_sum)

        #Hidden to output layer
        output_layer_sum = np.dot(hidden_layer_activation, self.weights_hidden_output) + self.bias_output
        output_layer_activation = self.sigmoid(output_layer_sum)
      
        return output_layer_activation, output_layer_sum, hidden_layer_activation, hidden_layer_sum
    

    def backward(self, y, output_layer_activation, output_layer_sum, hidden_layer_sum):
    # Output node calculation
        output_error = -self.absolute_error_derivative(y, output_layer_activation)
        output_delta = output_error * self.sigmoid_derivative(output_layer_sum)
        

        # Hidden node calculation
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer_sum)
        return output_delta, hidden_delta
    
    '''----------------BACKWARD WEIGHT DECAY-------------'''
    def backward_weight_decay(self, epoch,y, output_layer_activation, output_layer_sum, hidden_layer_sum):
    # Output node calculation
        hidden_omega = (1/(2*(self.weights_hidden_output.size)))*np.sum(self.weights_hidden_output**2)
        if epoch != 0:
            beta = 1/epoch
        else:
            beta = 0
        output_error = -self.absolute_error_derivative(y, output_layer_activation)+hidden_omega*beta
        output_delta = output_error * self.sigmoid_derivative(output_layer_sum)
        

        # Hidden node calculation
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer_sum)
        return output_delta, hidden_delta
    

    

    '''----------------------UPDATE WEIGHTS AND BIASES----------------------'''
    def update_weights(self, x, output_delta, hidden_delta,hidden_layer_activation):
        #update weights and biases
        
        x = x.reshape(1, -1)
        
        self.weights_hidden_output += np.dot(hidden_layer_activation.T,output_delta) * self.learning_rate
        self.weights_input_hidden += np.dot(x.T, hidden_delta) * self.learning_rate
    
        self.bias_output += output_delta * self.learning_rate
        self.bias_hidden += hidden_delta * self.learning_rate

    '''-----------------UPDATE WEIGHTS AND BIASES WITH MOMENTUM----------------------'''
    def update_weights_with_momentum(self, x, output_delta, hidden_delta, hidden_layer_activation):
        x = x.reshape(1, -1)
        
        # Initialise previous weight changes if they don't exist
        if not hasattr(self, 'prev_weight_change_hidden_output'):
            self.prev_weight_change_hidden_output = np.zeros_like(self.weights_hidden_output)
            self.prev_weight_change_input_hidden = np.zeros_like(self.weights_input_hidden)
            self.prev_bias_change_output = np.zeros_like(self.bias_output)
            self.prev_bias_change_hidden = np.zeros_like(self.bias_hidden)
        
        # Momentum parameter
        momentum = 0.9  # Feel free to adjust this value
        
        # Compute the current weight changes (gradients scaled by the learning rate)
        weight_change_hidden_output = np.dot(hidden_layer_activation.T, output_delta) * self.learning_rate
        weight_change_input_hidden = np.dot(x.T, hidden_delta) * self.learning_rate
        bias_change_output = output_delta * self.learning_rate
        bias_change_hidden = hidden_delta * self.learning_rate
        
        # Apply momentum: combine the current weight changes with 0.9 * previous changes
        self.weights_hidden_output += weight_change_hidden_output + momentum * self.prev_weight_change_hidden_output
        self.weights_input_hidden += weight_change_input_hidden + momentum * self.prev_weight_change_input_hidden
        self.bias_output += bias_change_output + momentum * self.prev_bias_change_output
        self.bias_hidden += bias_change_hidden + momentum * self.prev_bias_change_hidden
        
        # Save the current weight changes as the previous changes for the next iteration
        self.prev_weight_change_hidden_output = weight_change_hidden_output
        self.prev_weight_change_input_hidden = weight_change_input_hidden
        self.prev_bias_change_output = bias_change_output
        self.prev_bias_change_hidden = bias_change_hidden

    
    '''---------------------UPDATE WEIGHTS WITH BATCH LEARNING------'''
    def update_weights_batch(self, X_batch, output_delta_batch, hidden_delta_batch, hidden_activation_batch):
    
        batch_size = X_batch.shape[0]

        # Compute batch sum of weight updates
        delta_w_hidden_output = np.dot(hidden_activation_batch.T, output_delta_batch) / batch_size
        delta_w_input_hidden = np.dot(X_batch.T, hidden_delta_batch) / batch_size

        delta_b_output = np.sum(output_delta_batch, axis=0, keepdims=True) / batch_size
        delta_b_hidden = np.sum(hidden_delta_batch, axis=0, keepdims=True) / batch_size

        # Apply updates
        self.weights_hidden_output += self.learning_rate * delta_w_hidden_output
        self.weights_input_hidden += self.learning_rate * delta_w_input_hidden

        self.bias_output += self.learning_rate * delta_b_output
        self.bias_hidden += self.learning_rate * delta_b_hidden

    '''----------------------TRAINING FUNCTION----------------------'''  
    def train(self, x, y):
        self.loss_history = []  # Store loss for analysis

        for epoch in range(self.epochs):  # Loop through epochs
            total_loss = 0  # Track total loss for this epoch

            for i in range(len(x)):  # Iterate over each training sample
                predictors = x[i].reshape(1, -1)  # Ensure correct shape (row vector)
                target = y[i].reshape(1, -1)  # Ensure correct shape (row vector)
            

                # Forward pass
                output, output_sum, hidden_activation, hidden_sum = self.forward(predictors)

                # Compute loss and accumulate it
                loss = self.absolute_error(target, output).sum()
                
                total_loss += loss

                # Backward pass (compute gradients)
                output_delta, hidden_delta = self.backward(target, output,output_sum, hidden_sum)

                # Update weights and biases
                self.update_weights(predictors, output_delta, hidden_delta, hidden_activation)

            # Store and display average loss for the epoch
            avg_loss = total_loss / len(x)

            # Denormalize average loss
            denormalised_loss = denormalise(avg_loss, y_min, y_max)
            print(f"Epoch: {epoch+1}, Denormalised Loss: {denormalised_loss}")

            # Append denormalized loss to history
            self.loss_history.append(denormalised_loss)
        
        return self.loss_history  # Return loss history for analysis
    


    
    
    '''----------------TRAINING FUNCTION WITH MOMENTUM----------------'''
    def train_with_momentum(self, x, y):
        self.loss_history = []  # Store loss for analysis

        for epoch in range(self.epochs):  # Loop through epochs
            total_loss = 0  # Track total loss for this epoch

            for i in range(len(x)):  # Iterate over each training sample
                predictors = x[i].reshape(1, -1)  # Ensure correct shape (row vector)
                target = y[i].reshape(1, -1)  # Ensure correct shape (row vector)

                # Forward pass
                output, output_sum, hidden_activation, hidden_sum = self.forward(predictors)

                # Compute loss and accumulate it
                loss = self.absolute_error(target, output)
                
                total_loss += loss

                # Backward pass (compute gradients)
                output_delta, hidden_delta = self.backward(target, output,output_sum, hidden_sum)

                # Update weights and biases
                self.update_weights_with_momentum(predictors, output_delta, hidden_delta, hidden_activation)

            # Store and display average loss for the epoch
            avg_loss = total_loss / len(x)

            # Denormalize average loss
            denormalised_loss = denormalise(avg_loss, y_min, y_max)
            print(f"Epoch: {epoch+1}, Denormalised Loss: {denormalised_loss}")

            # Append denormalized loss to history
            self.loss_history.append(denormalised_loss)
        
        return self.loss_history  # Return loss history for analysis
    




    '''----------------TRAINING FUNCTION WITH BOLD DRIVER----------------'''
    def train_with_bold_driver(self, x, y):
        self.loss_history = []  # Store loss for analysis
        max_learning_rate = 0.5
        min_learning_rate = 0.01

        learning_rate = self.learning_rate  

        # Initialize bold driver parameters
        increase_factor = 1.1
        decrease_factor = 0.5
        prev_loss = float('inf')

        for epoch in range(self.epochs):  # Loop through epochs
            total_loss = 0  # Track total loss for this epoch

            for i in range(len(x)):  # Iterate over each training sample
                predictors = x[i].reshape(1, -1)  # Ensure correct shape (row vector)
                target = y[i].reshape(1, -1)  # Ensure correct shape (row vector)

                # Forward pass
                output, output_sum, hidden_activation, hidden_sum = self.forward(predictors)

                # Compute loss and accumulate it
                loss = self.absolute_error(target, output).sum()
                
                total_loss += loss

                # Backward pass (compute gradients)
                output_delta, hidden_delta = self.backward(target, output,output_sum, hidden_sum)

                # Update weights and biases
                self.update_weights(predictors, output_delta, hidden_delta, hidden_activation)

            # Store and display average loss for the epoch
            avg_loss = total_loss / len(x)
            # **Apply Bold Driver every 50 epochs**
            if epoch % 250 == 0:
                if avg_loss < prev_loss:  # If loss decreased
                    learning_rate = min(learning_rate * increase_factor, max_learning_rate)  # Increase learning rate
                else:  # If loss increased
                    learning_rate = max(learning_rate * decrease_factor, min_learning_rate)  # Decrease learning rate

                # Update self.learning_rate after modifying it
                self.learning_rate = learning_rate 

            # Update the previous loss for the next iteration
            prev_loss = avg_loss

            # Denormalize average loss
            denormalised_loss = denormalise(avg_loss, y_min, y_max)
            print(f"Epoch: {epoch+1}, Denormalised Loss: {denormalised_loss}")

            # Append denormalized loss to history
            self.loss_history.append(denormalised_loss)
        
        return self.loss_history  # Return loss history for analysis

    



    '''----------------TRAINING FUNCTION WITH MOMENTUM AND BOLD DRIVER----------------'''
    def train_with_bold_driver_momentum(self, x, y):
        self.loss_history = []  # Store loss for analysis
        max_learning_rate = 0.5
        min_learning_rate = 0.01

        learning_rate = self.learning_rate  

        # Initialize bold driver parameters
        increase_factor = 1.1
        decrease_factor = 0.5
        prev_loss = float('inf')

        for epoch in range(self.epochs):  # Loop through epochs
            total_loss = 0  # Track total loss for this epoch

            for i in range(len(x)):  # Iterate over each training sample
                predictors = x[i].reshape(1, -1)  # Ensure correct shape (row vector)
                target = y[i].reshape(1, -1)  # Ensure correct shape (row vector)

                # Forward pass
                output, output_sum, hidden_activation, hidden_sum = self.forward(predictors)

                # Compute loss and accumulate it
                loss = self.absolute_error(target, output).sum()
                
                total_loss += loss

                # Backward pass (compute gradients)
                output_delta, hidden_delta = self.backward(target, output,output_sum, hidden_sum)

                # Update weights and biases
                self.update_weights_with_momentum(predictors, output_delta, hidden_delta, hidden_activation)

            # Store and display average loss for the epoch
            avg_loss = total_loss / len(x)
            # **Apply Bold Driver every 50 epochs**
            if epoch % 250 == 0:
                if avg_loss < prev_loss:  # If loss decreased
                    learning_rate = min(learning_rate * increase_factor, max_learning_rate)  # Increase learning rate
                else:  # If loss increased
                    learning_rate = max(learning_rate * decrease_factor, min_learning_rate)  # Decrease learning rate

                # Update self.learning_rate after modifying it
                self.learning_rate = learning_rate 

            # Update the previous loss for the next iteration
            prev_loss = avg_loss

            # Denormalize average loss
            denormalised_loss = denormalise(avg_loss, y_min, y_max)
            print(f"Epoch: {epoch+1}, Denormalised Loss: {denormalised_loss}")

            # Append denormalized loss to history
            self.loss_history.append(denormalised_loss)
        
        return self.loss_history  # Return loss history for analysis





    '''---------------TRAINING WITH WEIGHT DECAY----------------'''
    def train_weight_decay(self, x, y):
        self.loss_history = []  # Store loss for analysis

        for epoch in range(self.epochs):  # Loop through epochs
            total_loss = 0  # Track total loss for this epoch
            

            for i in range(len(x)):  # Iterate over each training sample
                predictors = x[i].reshape(1, -1)  # Ensure correct shape (row vector)
                target = y[i].reshape(1, -1)  # Ensure correct shape (row vector)

                # Forward pass
                output, output_sum, hidden_activation, hidden_sum = self.forward(predictors)

                # Compute loss and accumulate it
                loss = self.absolute_error(target, output).sum()
                
                total_loss += loss

                # Backward pass (compute gradients)
                output_delta, hidden_delta = self.backward_weight_decay(epoch,target, output,output_sum, hidden_sum)

                # Update weights and biases
                self.update_weights(predictors, output_delta, hidden_delta, hidden_activation)

            # Store and display average loss for the epoch
            avg_loss = total_loss / len(x)

            # Denormalize average loss
            denormalised_loss = denormalise(avg_loss, y_min, y_max)
            print(f"Epoch: {epoch+1}, Denormalised Loss: {denormalised_loss}")

            # Append denormalized loss to history
            self.loss_history.append(denormalised_loss)
        
        return self.loss_history  # Return loss history for analysis 


    '''-------------TRAINING WITH ALL IMPROVEMENTS-----------------'''

    def train_with_bold_driver_momentum_weightdecay(self, x, y):
        self.loss_history = []  # Store loss for analysis
        max_learning_rate = 0.5
        min_learning_rate = 0.01

        learning_rate = self.learning_rate  

        # Initialize bold driver parameters
        increase_factor = 1.1
        decrease_factor = 0.5
        prev_loss = float('inf')

        for epoch in range(self.epochs):  # Loop through epochs
            total_loss = 0  # Track total loss for this epoch

            for i in range(len(x)):  # Iterate over each training sample
                predictors = x[i].reshape(1, -1)  # Ensure correct shape (row vector)
                target = y[i].reshape(1, -1)  # Ensure correct shape (row vector)

                # Forward pass
                output, output_sum, hidden_activation, hidden_sum = self.forward(predictors)

                # Compute loss and accumulate it
                loss = self.absolute_error(target, output).sum()
                
                total_loss += loss

                # Backward pass (compute gradients)
                output_delta, hidden_delta = self.backward_weight_decay(epoch,target, output,output_sum, hidden_sum)

                # Update weights and biases
                self.update_weights_with_momentum(predictors, output_delta, hidden_delta, hidden_activation)

            # Store and display average loss for the epoch
            avg_loss = total_loss / len(x)
            # **Apply Bold Driver every 50 epochs**
            if epoch % 250 == 0:
                if avg_loss < prev_loss:  # If loss decreased
                    learning_rate = min(learning_rate * increase_factor, max_learning_rate)  # Increase learning rate
                else:  # If loss increased
                    learning_rate = max(learning_rate * decrease_factor, min_learning_rate)  # Decrease learning rate

                # Update self.learning_rate after modifying it
                self.learning_rate = learning_rate 

            # Update the previous loss for the next iteration
            prev_loss = avg_loss

            # Denormalize average loss
            denormalised_loss = denormalise(avg_loss, y_min, y_max)
            print(f"Epoch: {epoch+1}, Denormalised Loss: {denormalised_loss}")

            # Append denormalized loss to history
            self.loss_history.append(denormalised_loss)
        
        return self.loss_history  # Return loss history for analysis
    


    '''--------------BATCH LEARNING WITH IMPROVEMENTS-------------------'''

    def train_with_batch_and_improvements(self, x, y, batch_size=32):
        self.loss_history = []  # Store loss for analysis

        # Learning rate parameters
        max_learning_rate = 0.5
        min_learning_rate = 0.01
        learning_rate = self.learning_rate  

        # Bold driver parameters
        increase_factor = 1.1
        decrease_factor = 0.5
        prev_loss = float('inf')

        num_samples = len(x)

        for epoch in range(self.epochs):  # Loop through epochs
            total_loss = 0  # Track total loss for this epoch

            # Shuffle dataset indices
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for batch_start in range(0, num_samples, batch_size):  # Iterate over batches
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]

                # Forward pass
                output, output_sum, hidden_activation, hidden_sum = self.forward(batch_x)

                # Compute loss and accumulate it
                loss = self.absolute_error(batch_y, output).sum()
                total_loss += loss

                # Backward pass (compute gradients)
                output_delta, hidden_delta = self.backward_weight_decay(epoch, batch_y, output, output_sum, hidden_sum)

                # Update weights and biases using batch updates
                self.update_weights_batch(batch_x, output_delta, hidden_delta, hidden_activation)

            # Compute and store average loss for the epoch
            avg_loss = total_loss / (num_samples // batch_size)

            # **Apply Bold Driver every 250 epochs**
            if epoch % 250 == 0:
                if avg_loss < prev_loss:  # If loss decreased
                    learning_rate = min(learning_rate * increase_factor, max_learning_rate)  # Increase learning rate
                else:  # If loss increased
                    learning_rate = max(learning_rate * decrease_factor, min_learning_rate)  # Decrease learning rate
                
                # Update learning rate
                self.learning_rate = learning_rate 

            # Update the previous loss for the next iteration
            prev_loss = avg_loss

            # Denormalize average loss
            denormalised_loss = denormalise(avg_loss, y_min, y_max)
            print(f"Epoch: {epoch+1}, Denormalised Loss: {denormalised_loss}")

            # Append denormalized loss to history
            self.loss_history.append(denormalised_loss)
        
        return self.loss_history  # Return loss history for analysis


        
    def get_weights(self):
        return self.weights_input_hidden, self.weights_hidden_output
        
    def get_loss_history(self):
        return self.loss_history
    
#Regression graph
def plot_regression_results(y_true, y_pred):

    residuals = y_true - y_pred  # Compute residuals
    
    # Compute Pearson correlation coefficient
    correlation, _ = pearsonr(y_true.flatten(), y_pred.flatten())

    plt.figure(figsize=(8, 6))
    
    # Scatter plot for actual vs predicted values
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6, label="Predicted vs Actual")
    
    # Reference line (Perfect Fit y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='dashed', label="Perfect Fit (y=x)")

    # Display correlation coefficient
    plt.text(min(y_pred), max(residuals), f"Pearson r = {correlation:.3f}", fontsize=12, color='black')
    
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression: Predicted vs. Actual")
    plt.legend()
    plt.grid(True)
    plt.show()


        

# Load the cleaned data
df_cleaned = pd.read_csv('Cleaned_Data.txt', sep='\t')

# Drop the Date column and unnecessary columns
drop_cols = ["Date", "Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme", "Crakehill", "Skip Bridge", "Westwick"]
df_cleaned = df_cleaned.drop(columns=[col for col in drop_cols if col in df_cleaned.columns])

# Drop rows with NaN values
df_cleaned = df_cleaned.drop(df.index[-1]).reset_index(drop=True)
df_cleaned = df_cleaned.drop(df.index[0]).reset_index(drop=True)

# Check to see if there are any NaN values
#print(df_cleaned.isnull().any().sum())

# Split the data into predictors and target variable
x = df_cleaned.drop(columns=["Skelton Tomorrow"])
y = df_cleaned["Skelton Tomorrow"]

# Normalize features (Standardization: mean = 0, std = 1)
#x = (x - x.mean()) / X.std()
#y = (y - y.mean()) / y.std()

x_min = x.min()
x_max = x.max()
y_min = y.min()
y_max = y.max()


# Min-Max Standardization
x = 0.8*(x - x_min) / (x_max - x_min)+0.1
y = 0.8*(y - y_min) / (y_max - y_min)+0.1

#denormalise error data
def denormalise(average_loss, y_min, y_max):
    return (average_loss) * (y_max - y_min)+ y_min






# Convert to NumPy arrays
x = x.to_numpy()
y = pd.Series(y).to_numpy().reshape(-1, 1)  # Ensure y is a column vector


# Split data into training (50%), testing (25%), validation (25%)

train_size = int(0.5 * len(x))   # 50% Training
test_size = int(0.25 * len(x))   # 25% Test
val_size = len(x) - (train_size + test_size)  # 25% Validation

x_train, x_test, x_val = np.split(x, [train_size, train_size + test_size])
y_train, y_test, y_val = np.split(y, [train_size, train_size + test_size])



# Perform backpropagation
learning_rate = 0.01
epochs = 1000 # Increased epochs for better training
# Initialize the backpropagation model
model = MLP(x_train, y_train,x_train.shape[1],y_train.shape[1],6, learning_rate, epochs)

# Train the model
output = model.train_with_bold_driver_momentum_weightdecay(x, y)


# Get the weights and loss history
weights = model.get_weights()
loss_history = model.get_loss_history()



# Plot the loss curve
plt.figure(figsize=(8, 6))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.grid(True)
plt.show()


# Plot the predicted vs actual values
y_pred = model.forward(x_test)[0] 





# Denormalize the data
# Calculate min and max for y using training data

y_mean = df_cleaned["Skelton Tomorrow"].mean()
y_std = df_cleaned["Skelton Tomorrow"].std()

# Denormalize y_pred and y_test using training data's min and max
y_pred = (y_pred - 0.1) / 0.8 * (y_max - y_min) + y_min
y_test = (y_test - 0.1) / 0.8 * (y_max - y_min) + y_min
y_train = (y_train - 0.1) / 0.8 * (y_max - y_min) + y_min
y_val = (y_val - 0.1) / 0.8 * (y_max - y_min) + y_min

# Denormalize predictions and actual values before plotting
#y_pred = (y_pred * y_std) + y_mean
#y_test = (y_test.flatten() * y_std) + y_mean


# Plot Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7, color='blue')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.7, color='orange')
plt.xlabel('Sample')
plt.ylabel('Skelton Tomorrow')
plt.title('Predicted vs Actual Values (Scatter Plot)')
plt.legend()
plt.grid(True)
plt.show()

plot_regression_results(y_test, y_pred)
