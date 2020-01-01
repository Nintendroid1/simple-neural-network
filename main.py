from numpy import exp, array, random, dot, hstack, empty, asarray
import sys, csv

#Source code: https://github.com/miloharper/simple-neural-network
class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 4 input connections and 1 output connection.
        # We assign random weights to a 4 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((4, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


#Assigns values found in row (csv data) to 1 and 0 in base array
def assignValues(row):
    status = str(row[0])
    age = int(row[6])
    race = str(row[9])
    ethnicity = str(row[10])
    sex = str(row[11])
    
    print(str(age) + "\t" + str(race) + "\t\t" + str(ethnicity) + "\t\t" + str(sex))
    addInput = [0,0,0,0]
    addOutput = [0]

    if("Arrest" in status):
        addOutput[0] = 1
    #TODO Change inductive biases
    if(age > 30):
        addInput[0] = 1
    if(race == 'BLACK'):
        addInput[1] = 1
    if("NOT" not in ethnicity):
        addInput[2] = 1
    if(sex == 'MALE'):
        addInput[3] = 1

    training_set_inputs.append(addInput)
    training_set_outputs.append(addOutput)

#Parses the csv data
def parseData(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        # base = []
        for row in csv_reader:
            if line_count == 0:
                #Prints all column names
                #print(f'Column names are {", ".join(row)}')

                print(str(row[6]) + "\t" + str(row[9]) + "\t" + str(row[10]) + "\t" + str(row[11]))
                line_count += 1
            else:
                assignValues(row)
                line_count += 1
        # return base

#Tests the neural network based on user-specified input
#TODO Change to file specified
def testNetwork(neural_network):
    # Test the neural network with new situations
    print("Considering new situation Young Black Male: [0, 1, 0, 1] -> ?: ")
    print(neural_network.think(array([0, 1, 0, 1])))

    print("Considering new situation Old White Male: [1, 0, 0, 1] -> ?: ")
    print(neural_network.think(array([1, 0, 0, 1])))

    print("Considering new situation Young Black Female: [0, 1, 0, 0] -> ?: ")
    print(neural_network.think(array([0, 1, 0, 0])))

    print("Considering new situation Young Hispanic Female: [0, 0, 1, 0] -> ?: ")
    print(neural_network.think(array([0, 0, 1, 0])))

    print("Considering new situation Old White Female: [1, 0, 1, 0] -> ?: ")
    print(neural_network.think(array([1, 0, 0, 0])))

if __name__ == "__main__":
    if(len(sys.argv) == 2):
        #Intialise a single neuron neural network.
        neural_network = NeuralNetwork()

        print("Random starting synaptic weights: ")
        print(neural_network.synaptic_weights)

        global training_set_inputs
        global training_set_outputs
        training_set_inputs = []
        training_set_outputs = []

        # Read csv file
        # training_set_inputs = parseData(sys.argv[1])
        parseData(sys.argv[1])
        training_set_inputs = asarray(training_set_inputs)
        training_set_outputs = asarray(training_set_outputs)
        print(training_set_inputs)
        print(training_set_outputs)

        # Train the neural network using a training set.
        # Do it 10,000 times and make small adjustments each time.
        neural_network.train(training_set_inputs, training_set_outputs, 10000)

        print("New synaptic weights after training: ")
        print(neural_network.synaptic_weights)

        testNetwork(neural_network)
    else:
        print("Please input file name for training data for neural network") 
