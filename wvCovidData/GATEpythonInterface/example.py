from GateInterface import *

#initialise gate
gate= GateInterFace()
# path to gate interface, for my case is /Users/xingyi/Gate/gateCodes/pythonInferface
# this will open java tcp server (on port 7899) to execute GATE operation
#gate.init('/Users/xingyi/Gate/gateCodes/pythonInferface') 
gate.init('./') 

# load document from url
document = GateDocument()
document.loadDocumentFromURL("https://gate.ac.uk")

# load document from local file e.g.
#document.loadDocumentFromFile("/Users/xingyi/Gate/gateCodes/pythonInferface/ft-airlines-27-jul-2001.xml")

# get content from file
content = document.getDocumentContent()
print(content)

# get annotation set name
atsName = document.getAnnotationSetNames()
print(atsName)

#get Original markups annotation set
ats = document.getAnnotations('Original markups')


# load maven plugins
gate.loadMvnPlugins("uk.ac.gate.plugins", "annie", "8.5")

# load PRs 
gate.loadPRs('gate.creole.annotdelete.AnnotationDeletePR', 'anndeletePR')
gate.loadPRs('gate.creole.tokeniser.DefaultTokeniser', 'defToken')

# set initialise parameter for PR
#prparameter['grammarURL'] = 'file:////Users/xingyi//Gate/ifpri/JAPE/main.jape'
#gate.loadPRs('gate.creole.Transducer', prparameter)

# create a pipeline
testPipeLine = GatePipeline('testpipeline')
testPipeLine.createPipeline()

# add PRs to the pipeline
testPipeLine.addPR('anndeletePR')
testPipeLine.addPR('defToken')

# get params
print(testPipeLine.checkRunTimeParams('anndeletePR', 'setsToKeep'))
# set run time params
testPipeLine.setRunTimeParams('anndeletePR', 'keepOriginalMarkupsAS', 'false', 'Boolean')

testPipeLine.setRunTimeParams('anndeletePR', 'setsToKeep', 'Key,Target', 'List')
print(testPipeLine.checkRunTimeParams('anndeletePR', 'setsToKeep'))


# create gate corpus
testCorpus = GateCorpus('testCorpus')

# add document to the corpus
testCorpus.addDocument(document)

# set corpus for pipeline
testPipeLine.setCorpus(testCorpus)

# run pipeline
testPipeLine.runPipeline()

# get default annotation set after run the pipeline, the annotation is stored in local python format
defaultats = document.getAnnotations('')

# load appilication from file
#testPipeline2 = GatePipeline('testpipeline2')
#testPipeLine.loadPipelineFromFile('/Users/xingyi/Gate/ifpri/ifpri.xgapp')

# close the port 
gate.close()

