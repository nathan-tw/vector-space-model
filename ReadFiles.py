import os

#迭代資料夾
def readDocuments():    
    documentsList = []
    indexList = []
    for dirPath, dirNames, fileNames in os.walk("documents"):
        for f in fileNames:
            filePath = os.path.join(dirPath, f)
            file = open(filePath, 'r')
            documentsList.append(file.read())
            indexList.append(f[:6])
    return indexList, documentsList
