import os

def makepath(params):
    if not os.path.isdir("./out"):
        os.mkdir("./out")

    dirPath = "./out/" + params["SimName"]
    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)
    dirPath = dirPath + "/waves"
    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)