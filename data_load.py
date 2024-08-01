import toy_data

def LoadData(num):
    data = toy_data.ToyData(num_classes=num)

    return data

def DivideData(data):
    xtrain, ytrain, xtest, ytest = data.load_data()

    return xtrain, ytrain, xtest, ytest




