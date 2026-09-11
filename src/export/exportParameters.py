################################################################################################################################
# Write all parameters used to a file for debugging and simulation reproducability
################################################################################################################################

import os
import pprint

def exportParameters(params):
    OutputFileName = "./out/" + params["SimName"] + "/InputParameters.txt"
    with open(OutputFileName, 'w+') as OutputFile:
        pprint.pprint(params, OutputFile)