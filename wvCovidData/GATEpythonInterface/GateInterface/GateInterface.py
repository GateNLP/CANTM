import socket
import json
import re
import os
import subprocess
import time
import signal
import pathlib


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class GateInterFace:
    def __init__(self):
        self.PORT = 7899
        self.HOST = "localhost"
        self.maxSendChar = 512
        self.maxRecvChar = 512
        self.loadedPlugins = []
        self.loadedPrs = []
        self.pro = None
        self.logFile = "gpiterface"+str(self.PORT)+".log"
        self.interfaceJavaPath = None

    def init(self, interfaceJavaPath=None):
        if interfaceJavaPath:
            self.interfaceJavaPath = interfaceJavaPath
        else:
            interfaceJavaPath = pathlib.Path(__file__).parent.absolute()
            self.interfaceJavaPath = interfaceJavaPath.parent


        cwd = os.getcwd()
        runScript = os.path.join(self.interfaceJavaPath,'run.sh')
        os.chdir(self.interfaceJavaPath)
        

        if (os.path.isfile(self.logFile)):
            print("logfile existed, try to close previous session")
            try:
                with open(self.logFile,'r') as fin:
                    line = fin.readline().strip()
                    pid = int(line)
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
            except:
                print("can not kill process-"+str(pid)+"please close manully")
            
            
        print(runScript)
        #self.pro = subprocess.Popen(["bash", runScript])
        hostportArg = "-Dexec.args=\""+str(self.PORT)+"\""
        args=["mvn", "exec:java", "-Dexec.mainClass=uk.ac.gate.python.pythonInferface.GateServer",hostportArg]
        #self.pro = subprocess.Popen(["mvn", "exec:java -Dexec.mainClass=uk.ac.gate.python.pythonInferface.GateServer -Dexec.args="7899""])
        self.pro = subprocess.Popen(args)
        with open(self.logFile,'w') as fo:
            fo.write(str(self.pro.pid))
        os.chdir(cwd)
        time.sleep(5)

    def close(self):


        print(self.pro.pid)
        logFile = os.path.join(self.interfaceJavaPath, self.logFile)
        print(logFile)
        os.remove(logFile)
        os.killpg(os.getpgid(self.pro.pid), signal.SIGTERM)
        #try:
        #    os.killpg(os.getpgid(self.pro.pid), signal.SIGTERM)
        #    logFile = os.path.join(self.interfaceJavaPath, self.logFile)
        #    print(logFile)
        #    os.remove(logFile)
        #except:
        #    print("logfile not correctly removed")


    def test(self):
        self._sendDoc2Java('test','this is test sent')

    def _sendDoc2Java(self, jsonKey, jsonValue):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.HOST, self.PORT))
        previ = 0
        i = self.maxSendChar
        eov = False #if value length larger than self.maxChar then send in sepate packages to java
        valueLen = len(jsonValue)

        while(eov == False):
            subValue = jsonValue[previ:i]
            if i < valueLen:
                eov = False
            else:
                eov = True
            jsonSend = {'fromClient':{jsonKey:subValue, 'eov':eov}}
            previ = i
            i += self.maxSendChar
            sock.sendall((json.dumps(jsonSend, cls=MyEncoder)+"\n").encode('utf-8'))
        serverReturn = self._recvDocFromJava(sock)
        sock.close()
        #print(serverReturn)
        return serverReturn

    def _send2Java(self, jsonDict):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #print(self.HOST,self.PORT)
        sock.connect((self.HOST, int(self.PORT)))
        #print(jsonDict)
        for jsonKey in jsonDict:
            #print(jsonKey)
            jsonValue = jsonDict[jsonKey]
            valueLen = len(jsonValue)
            previ = 0
            i = self.maxSendChar
            eov = False #if value length larger than self.maxChar then send in sepate packages to java
            while(eov == False):
                subValue = jsonValue[previ:i]
                if i < valueLen:
                    eov = False
                else:
                    eov = True
                jsonSend = {'fromClient':{jsonKey:subValue, 'eov':False}}
                previ = i
                i += self.maxSendChar
                #print(jsonSend)
                sock.sendall((json.dumps(jsonSend, cls=MyEncoder)+"\n").encode('utf-8'))
        #print('finish')
        jsonSend = {'fromClient':{'eov':True}}
        sock.sendall((json.dumps(jsonSend, cls=MyEncoder)+"\n").encode('utf-8'))
        serverReturn = self._recvDocFromJava(sock)
        sock.close()
        #print(serverReturn)
        return serverReturn




    def _recvDocFromJava(self, sock):
        fullReturn = ""
        eov = False
        fullReturn = {}
        while(eov ==False):
            data_recv = sock.recv(self.maxRecvChar)
            data = json.loads(data_recv)
            eov = data['eov']
            for item in data:
                partReturn = data[item]
                if item not in fullReturn:
                    fullReturn[item] = partReturn
                else:
                    fullReturn[item] += partReturn
            sock.sendall("success\n".encode('utf-8'))
        return fullReturn



    def loadMvnPlugins(self, group, artifact, version):
        jsonDict = {}
        jsonDict['plugin'] = 'maven'
        jsonDict['group'] = group
        jsonDict['artifact'] = artifact
        jsonDict['version'] = version
        response = self._send2Java(jsonDict)
        if response['message'] == 'success':
            self.loadedPlugins.append(response['pluginLoaded'])
        return response


    def loadPRs(self, resourcePath, name, features=None):
        jsonDict = {}
        if features == None:
            jsonDict['loadPR'] = 'withoutFeature'
        else:
            jsonDict['loadPR'] = 'withFeature'
            i = 0
            for featureKey in features:
                keyName = 'prFeatureName'+str(i)
                valueName = 'prFeatureValue'+str(i)
                jsonDict[keyName] = featureKey
                jsonDict[valueName] = features[featureKey]
                i+=1

        jsonDict['resourcePath'] = resourcePath
        jsonDict['name'] = name
        response = self._send2Java(jsonDict)
        if response['message'] == 'success':
            self.loadedPrs.append(response['PRLoaded'])
        return response

    def reinitPRs(self, name):
        jsonDict = {}
        jsonDict['reInitPR'] = 'none'
        jsonDict['name'] = name
        response = self._send2Java(jsonDict)
        if response['message'] != 'success':
            print('error unable to reinit pr', name)
        return response['message']




class AnnotationSet(GateInterFace):
    def __init__(self):
        GateInterFace.__init__(self)
        self.annotationSet = []

    def __len__(self):
        return len(self.annotationSet)

    def __iter__(self):
        for node in self.annotationSet:
            yield node

    def getType(self, annotationType, startidx=None, endidx=None):
        newList = []
        for annotation in self.annotationSet:
            if annotation.type == annotationType:
                if startidx and endidx:
                    if annotation.startNode.offset >= startidx and annotation.endNode.offset <= endidx:
                        newList.append(annotation)
                else:
                    newList.append(annotation)
        subSet = AnnotationSet()
        subSet.annotationSet = newList
        return subSet


    def getbyRange(self, startidx=0, endidx=None):
        newList = []
        for annotation in self.annotationSet:
            if annotation.startNode.offset >= startidx:
                if endidx:
                    if annotation.endNode.offset <= endidx:
                        newList.append(annotation)
                else:
                    newList.append(annotation)
        subSet = AnnotationSet()
        subSet.annotationSet = newList
        return subSet


    def get(self,i):
        return self.annotationSet[i]

    def getbyId(self, annoId):
        returnAnno = None
        for annotation in self.annotationSet:
            if annotation.id == annoId:
                returnAnno = annotation
                break
        return returnAnno

    def append(self, annotation):
        self.annotationSet.append(annotation)
            

    def _getAnnotationFromResponse(self,response):
        annotationResponse = response['annotationSet']
        annotationList = annotationResponse.split('\n, Anno')
        #print('getting annotation from response')
        #print(len(annotationList))
        #print(annotationList[0])
        idPattern = '(?<=tationImpl\: id\=)\d*(?=\; type\=.*\;)'
        typePattern = '(?<=; type\=).*(?=\; features\=)'
        featurePattern = '(?<=; features\=\{)[\S\s]*(?=\}\; start\=.*\;)'
        startNodePattern = '(?<=; start\=NodeImpl\: ).*(?= end\=NodeImpl)'
        endNodePattern = '(?<=; end\=NodeImpl\: ).*' 


        fullPattern = 'AnnotationImpl\: id\=\d*\; type\=.*\; features\=\{.*\}\; start\=NodeImpl\: id\=\d*\; offset\=\d*\; end\=NodeImpl\: id\=\d*\; offset\=\d*'

        for rawAnnotationLine in annotationList:
            #print(rawAnnotationLine)
            #m = re.search(fullPattern, rawAnnotationLine)
            #print(m)
            #if m:
            #    print('match')
            try:
            #if 1:
                #print(rawAnnotationLine)
                currentAnnotation = Annotation()
                currentAnnotation.id = int(re.search(idPattern,rawAnnotationLine).group(0))
                currentAnnotation.type = re.search(typePattern,rawAnnotationLine).group(0)
                #print(len(re.search(featurePattern,rawAnnotationLine).group(0)))
                currentAnnotation._setFeatureFromRawLine(re.search(featurePattern,rawAnnotationLine).group(0))
                #print(currentAnnotation.id)
                #print(currentAnnotation.type)
                #print(currentAnnotation.features)
                currentAnnotation._setStartNode(re.search(startNodePattern,rawAnnotationLine).group(0))
                currentAnnotation._setEndNode(re.search(endNodePattern,rawAnnotationLine).group(0))
                #print(currentAnnotation.startNode.id, currentAnnotation.startNode.offset)
                #print(currentAnnotation.endNode.id, currentAnnotation.endNode.offset)
                self.annotationSet.append(currentAnnotation)
            except:
               #print('bad line, ignore') 
               #print(rawAnnotationLine)
               pass
               #currentAnnotation = Annotation()
               #currentAnnotation.id = int(re.search(idPattern,rawAnnotationLine).group(0))
               #currentAnnotation.type = re.search(typePattern,rawAnnotationLine).group(0)
               #currentAnnotation._setFeatureFromRawLine(re.search(featurePattern,rawAnnotationLine).group(0))
               #print(currentAnnotation.id)
               #print(currentAnnotation.type)
               #print(currentAnnotation.features)
               #currentAnnotation._setStartNode(re.search(startNodePattern,rawAnnotationLine).group(0))
               #currentAnnotation._setEndNode(re.search(endNodePattern,rawAnnotationLine).group(0))
               #print(currentAnnotation.startNode.id, currentAnnotation.startNode.offset)
               #print(currentAnnotation.endNode.id, currentAnnotation.endNode.offset)
               #self.annotationSet.append(currentAnnotation)

class Annotation:
    def __init__(self):
        self.id = None
        self.type = None
        self.features = {}
        self.startNode = None
        self.endNode = None


    def overlap_set(self, compareSet):
        overlap = False
        for compare_annotation in compareSet:
            overlap = self.overlaps(compare_annotation)
            if overlap:
                break
        return overlap

    def matches(self, compareAnno):
        startNodeOffsetMatch = self.startNode.offset == compareAnno.startNode.offset
        endNodeOffsetMatch = self.endNode.offset == compareAnno.endNode.offset
        if startNodeOffsetMatch and endNodeOffsetMatch:
            return True

    def overlaps(self, compareAnno):
        selfStart = self.startNode.offset
        selfEnd = self.endNode.offset-1
        compareStart = compareAnno.startNode.offset
        compareEnd = compareAnno.endNode.offset-1
        if selfStart <= compareStart:
            if selfEnd >= compareStart:
                return True
            else:
                return False
        elif selfStart >= compareStart:
            if compareEnd >= selfStart:
                return True
            else:
                return False


    def _setFeatureFromRawLine(self, rawFeatureLine):
        #print(rawFeatureLine)
        if len(rawFeatureLine) > 0:
            #replace comma in list to |||
            listPattern = '(?<=\=)\[[\w \,]+\]'
            listFeatures = re.findall(listPattern, rawFeatureLine)
            for listFeature in listFeatures:
                newPattern = re.sub(', ',' ||| ', listFeature)
                rawFeatureLine = rawFeatureLine.replace(listFeature, newPattern)
            splittedFeatures = re.split(', ',rawFeatureLine)
            #print(splittedFeatures)
            for splittedFeature in splittedFeatures:
                featureTok = splittedFeature.split('=')
                featureKey = featureTok[0]
                featureValue = featureTok[1]
                if self._isListFeature(featureValue):
                    #self.features[featureKey] = []
                    listValues = featureValue[1:-1].split(' ||| ')
                    self.features[featureKey] = listValues
                else:
                    self.features[featureKey] = featureValue

    def _isListFeature(self, featureValue):
        pattern = '\[.*\]'
        m = re.match(pattern, featureValue)
        if m:
            return True
        else:
            return False



    def _setStartNode(self, rawLine):
        idPattern = '(?<=id\=)\d*(?=\;)'
        offSetPattern = '(?<=offset\=)\d*(?=(\;)|($))'
        nodeId = int(re.search(idPattern, rawLine).group(0))
        offset = int(re.search(offSetPattern, rawLine).group(0))
        #print(nodeId, offset)
        startNode = Node()
        startNode.id = nodeId
        startNode.offset = offset
        self.startNode = startNode

    def _setEndNode(self, rawLine):
        idPattern = '(?<=id\=)\d*(?=\;)'
        offSetPattern = '(?<=offset\=)\d*(?=(\;)|($))'
        nodeId = int(re.search(idPattern, rawLine).group(0))
        offset = int(re.search(offSetPattern, rawLine).group(0))
        #print(nodeId, offset)
        endNode = Node()
        endNode.id = nodeId
        endNode.offset = offset
        self.endNode = endNode


        

class Node:
    def __init__(self):
        self.id = None
        self.offset = None


class GateDocument(GateInterFace):
    def __init__(self):
        GateInterFace.__init__(self)
        self.documentName = None

    def loadDocumentFromURL(self, documentURL):
        documentName = documentURL
        serverReturn = self._sendDoc2Java('loadDocumentFromURL', documentURL)
        if serverReturn['message'] == 'success':
            self.documentName = documentName
        #print(serverReturn)


    def loadDocumentFromFile(self, documentPath):
        documentName = documentPath
        serverReturn = self._sendDoc2Java('loadDocumentFromFile', documentPath)
        #print(serverReturn)
        if serverReturn['message'] == 'success':
            self.documentName = documentName
        #print(serverReturn)

    def getDocumentContent(self):
        jsonDict = {}
        jsonDict['document'] = 'getDocumentContent'
        jsonDict['docName'] = self.documentName
        response = self._send2Java(jsonDict)
        docContent = response['docContent']
        return docContent

    def getAnnotationSetNames(self):
        jsonDict = {}
        jsonDict['document'] = 'getAnnotationSetName'
        jsonDict['docName'] = self.documentName
        response = self._send2Java(jsonDict)
        astName = response['annotationSetName']
        return astName

    def getAnnotations(self, annotationSetName):
        jsonDict = {}
        jsonDict['document'] = 'getAnnotations'
        jsonDict['docName'] = self.documentName
        jsonDict['annotationSetName'] = annotationSetName
        currentAnnotationSet = AnnotationSet()
        #print(jsonDict)
        response = self._send2Java(jsonDict)
        #print(response)
        currentAnnotationSet._getAnnotationFromResponse(response)
        return currentAnnotationSet

    def clearDocument(self):
        jsonDict = {}
        jsonDict['clearDocument'] = self.documentName
        response = self._send2Java(jsonDict)


class GatePipeline(GateInterFace):
    def __init__(self, pipelineName):
        GateInterFace.__init__(self)
        self.pipelineName = pipelineName
        self.corpus = None
        self.prList = []

    def loadPipelineFromFile(self, filePath):
        jsonDict = {}
        jsonDict['pipeline'] = 'loadPipelineFromFile'
        jsonDict['pipelineName'] = self.pipelineName
        jsonDict['filtPath'] = filePath
        response = self._send2Java(jsonDict)
        #print(response)



    def createPipeline(self):
        jsonDict = {}
        jsonDict['pipeline'] = 'createPipeline'
        jsonDict['pipelineName'] = self.pipelineName
        response = self._send2Java(jsonDict)
        #print(response)


    def addPR(self, prName):
        jsonDict = {}
        jsonDict['pipeline'] = 'addPR'
        jsonDict['pipelineName'] = self.pipelineName
        jsonDict['prName'] = prName
        response = self._send2Java(jsonDict)
        #print(response)
        self.prList.append(prName)

    def setCorpus(self, corpus):
        corpusName = corpus.corpusName
        jsonDict = {}
        jsonDict['pipeline'] = 'setCorpus'
        jsonDict['pipelineName'] = self.pipelineName
        jsonDict['corpusName'] = corpusName
        response = self._send2Java(jsonDict)
        #print(response)
        self.corpus = corpus

    def runPipeline(self):
        jsonDict = {}
        jsonDict['pipeline'] = 'runPipeline'
        jsonDict['pipelineName'] = self.pipelineName
        response = self._send2Java(jsonDict)
        #print(response)

    def checkRunTimeParams(self, prName, paramName):
        jsonDict = {}
        jsonDict['pipeline'] = 'checkParams'
        jsonDict['pipelineName'] = self.pipelineName
        jsonDict['resourceName'] = prName
        jsonDict['paramsName'] = paramName
        response = self._send2Java(jsonDict)
        #print(response)
        return response['message']

    def setRunTimeParams(self, prName, paramName, paramValue, paramType):
        jsonDict = {}
        jsonDict['pipeline'] = 'setParams'
        jsonDict['pipelineName'] = self.pipelineName
        jsonDict['resourceName'] = prName
        jsonDict['paramsName'] = paramName
        jsonDict['paramsValue'] = paramValue
        jsonDict['paramsType'] = paramType
        response = self._send2Java(jsonDict)





class GateCorpus(GateInterFace):
    def __init__(self, corpusName):
        GateInterFace.__init__(self)
        self.corpusName = corpusName
        self._createCorpus()
        self.documentList = []

    def _createCorpus(self):
        jsonDict = {}
        jsonDict['corpus'] = 'createCorpus'
        jsonDict['corpusName'] = self.corpusName
        response = self._send2Java(jsonDict)
        #print(response)


    def clearCorpus(self):
        jsonDict = {}
        jsonDict['corpus'] = 'clearCorpus'
        jsonDict['corpusName'] = self.corpusName
        response = self._send2Java(jsonDict)

    def addDocument(self, document):
        documentName = document.documentName
        jsonDict = {}
        jsonDict['corpus'] = 'addDocument'
        jsonDict['corpusName'] = self.corpusName
        jsonDict['documentName'] = documentName
        response = self._send2Java(jsonDict)
        #print(response)
        self.documentList.append(document)



#if __name__ == "__main__":
#    gate= GateInterFace()
#    gate.init('/Users/xingyi/Gate/gateCodes/pythonInferface')
#    #gate.test()
#    document = GateDocument()
#    document.loadDocumentFromURL("https://gate.ac.uk")
#    #document.loadDocumentFromFile("/Users/xingyi/Gate/gateCodes/pythonInferface/ft-airlines-27-jul-2001.xml")
#    #print(document.documentName)
#    #content = document.getDocumentContent()
#    #print(content)
#    #atsName = document.getAnnotationSetNames()
#    #print(atsName)
#    #ats = document.getAnnotations('')
#    #print(len(ats.annotationSet))
#    #print(ats.annotationSet[0])
#    response=gate.loadMvnPlugins("uk.ac.gate.plugins", "annie", "8.5")
#    print(response)
#    prparameter={}
#    response=gate.loadPRs('gate.creole.annotdelete.AnnotationDeletePR')
#    response=gate.loadPRs('gate.creole.tokeniser.DefaultTokeniser')
#    prparameter['grammarURL'] = 'file:////Users/xingyi//Gate/ifpri/JAPE/main.jape'
#    response=gate.loadPRs('gate.creole.Transducer', prparameter)
#    print(response)
#    print(gate.loadedPrs)
#    testPipeLine = GatePipeline('testpipeline')
#    testPipeLine.createPipeline()
#    testPipeLine.addPR('gate.creole.annotdelete.AnnotationDeletePR')
#    testPipeLine.addPR('gate.creole.tokeniser.DefaultTokeniser')
#    testCorpus = GateCorpus('testCorpus')
#    testCorpus.addDocument(document)
#    testPipeLine.setCorpus(testCorpus)
#    testPipeLine.runPipeline()
#    #ats = document.getAnnotations('Original markups')
#    ats = document.getAnnotations('')
#    print(len(ats.annotationSet))
#    print(ats.annotationSet[0])
#    testPipeline2 = GatePipeline('testpipeline2')
#    testPipeLine.loadPipelineFromFile('/Users/xingyi/Gate/ifpri/ifpri.xgapp')
#    gate.close()
#


