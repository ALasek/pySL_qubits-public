import os

def exportEigenvalues(params, D, count):
    OutputFileName = "./out/" + params["SimName"] + "/eigs.txt"
    with open(OutputFileName, 'w+') as OutputFile:
        for i in range(0, count):
            OutputFile.write("{:e} \n".format(D[i]))