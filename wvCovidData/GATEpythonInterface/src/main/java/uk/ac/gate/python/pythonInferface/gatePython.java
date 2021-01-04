package uk.ac.gate.python.pythonInferface;

import java.io.*;
import java.net.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.json.JSONObject;

import gate.Annotation;
import gate.AnnotationSet;
import gate.Corpus;
import gate.CorpusController;
import gate.Document;
import gate.Factory;
import gate.FeatureMap;
import gate.Gate;
import gate.ProcessingResource;
import gate.creole.ConditionalSerialAnalyserController;
import gate.creole.Plugin;
import gate.creole.ResourceInstantiationException;
import gate.util.GateException;
import gate.util.persistence.PersistenceManager;

public class gatePython {
	public ConditionalSerialAnalyserController gatePipeline = null;
	public HashMap<String, Document> gateDocList = new HashMap<String, Document>();
	public HashMap<String, AnnotationSet> gateAnnotationSetList = new HashMap<String, AnnotationSet>();
	public HashMap<String, ConditionalSerialAnalyserController> gatePipelines = new HashMap<String, ConditionalSerialAnalyserController>();
	public HashMap<String, ProcessingResource> gatePrs = new HashMap<String, ProcessingResource>();
	public HashMap<String, Corpus> gateCorpus = new HashMap<String, Corpus>();
	
	public HashMap<String, String> processRequest(HashMap<String, String> fullRequest) throws IOException, GateException{
		//String returnResponse = "Nothing";
		HashMap<String, String> returnResponse = new HashMap<String, String>();
		returnResponse.put("message", "Nothing");
		
		
		///////////////load document///////////////////////
		if (fullRequest.containsKey("loadDocumentFromURL")){
			String currentDocumentName = fullRequest.get("loadDocumentFromURL");
			if (gateDocList.containsKey(currentDocumentName)){
				//returnResponse = "existed document";
				returnResponse.put("message", "existed document");
			}
			else{
				Document gateDoc = document.loadDocument(currentDocumentName);
				//System.out.println(gateDoc.getAnnotationSetNames());
				gateDocList.put(currentDocumentName, gateDoc);
				//returnResponse ="success";
				returnResponse.put("message", "success");
			}
		}
		
		if (fullRequest.containsKey("loadDocumentFromFile")){
			String currentDocumentName = fullRequest.get("loadDocumentFromFile");
			if (gateDocList.containsKey(currentDocumentName)){
				//returnResponse = "existed document";
				returnResponse.put("message", "existed document");
			}
			else{
				Document gateDoc = document.loadDocumentFromFile(currentDocumentName);
				//System.out.println(gateDoc.getAnnotationSetNames());
				gateDocList.put(currentDocumentName, gateDoc);
				//returnResponse ="success";
				returnResponse.put("message", "success");
			}
		}
		if (fullRequest.containsKey("clearDocument")){
			String currentDocumentName = fullRequest.get("clearDocument");
			if (gateDocList.containsKey(currentDocumentName)){
				//returnResponse = "existed document";
				Document currentDoc = gateDocList.get(currentDocumentName);
				currentDoc.cleanup();
				gateDocList.remove(currentDocumentName);
				returnResponse.put("message", "success");
			}
			else{
				returnResponse.put("message", "no such document");
			}
		}
		
		
		///////////////get document content////////////////////
		if (fullRequest.containsKey("document")){
			returnResponse = document.documentOperations(fullRequest, gateDocList);
		}
		
		//////////////load plugin/////////////////////
		if (fullRequest.containsKey("plugin")){
			if (fullRequest.get("plugin").equals("maven")){
				String group = fullRequest.get("group");
				String artifact = fullRequest.get("artifact");
				String version = fullRequest.get("version");
				Gate.getCreoleRegister().registerPlugin(new Plugin.Maven(group, artifact, version));
				returnResponse.put("pluginLoaded", artifact);
				returnResponse.put("message", "success");
			}
		}
		
		//////////////load prs///////////////////////////
		if (fullRequest.containsKey("loadPR")){
			ProcessingResource currentPR;
			String prName = fullRequest.get("name");
			String resourcePath = fullRequest.get("resourcePath");
			if (fullRequest.get("loadPR").equals("withFeature")){
				Boolean finishFeatureLoading = false;
				int featureID = 0;
				FeatureMap currentParams = Factory.newFeatureMap();
				do{
					String currentFeatureNameKey = "prFeatureName"+Integer.toString(featureID);
					String currentFeatureValueKey = "prFeatureValue"+Integer.toString(featureID);
					if (fullRequest.containsKey(currentFeatureNameKey)){
						String currentFeatureName = fullRequest.get(currentFeatureNameKey);
						String currentFeatureValue = fullRequest.get(currentFeatureValueKey);
						currentParams.put(currentFeatureName, currentFeatureValue);	
					}
					else{
						finishFeatureLoading = true;
					}
					featureID +=1;
				}while(finishFeatureLoading != true);
				currentPR=(ProcessingResource) Factory.createResource(resourcePath,currentParams);
			}
			else{
				currentPR=(ProcessingResource) Factory.createResource(resourcePath);
			}
			
			gatePrs.put(prName, currentPR);
			//System.out.println("finish loading pr");
			returnResponse.put("PRLoaded", prName);
			returnResponse.put("message", "success");
			//System.out.println(returnResponse);
		}
		
		if (fullRequest.containsKey("reInitPR")){
			ProcessingResource currentPR;
			String prName = fullRequest.get("name");
			currentPR = gatePrs.get(prName);
			currentPR.reInit();
			returnResponse.put("PRReinit", prName);
			returnResponse.put("message", "success");
			//System.out.println(returnResponse);
		}
		
		
		///////////////////pipelines/////////////////////////////////////////
		if (fullRequest.containsKey("pipeline")){
			if (fullRequest.get("pipeline").equals("createPipeline")){
				String pipelineName = fullRequest.get("pipelineName");
				ConditionalSerialAnalyserController currentController = (ConditionalSerialAnalyserController) Factory.createResource("gate.creole.ConditionalSerialAnalyserController");
				gatePipelines.put(pipelineName, currentController);
				//System.out.println("finish creating pipeline");
				returnResponse.put("pipelineCreated", pipelineName);
				returnResponse.put("message", "success");
			}
			
			if (fullRequest.get("pipeline").equals("loadPipelineFromFile")){
				String pipelineName = fullRequest.get("pipelineName");
				String filePath = fullRequest.get("filtPath");
				ConditionalSerialAnalyserController currentController = (ConditionalSerialAnalyserController) PersistenceManager.loadObjectFromFile(new File(filePath));;
				gatePipelines.put(pipelineName, currentController);
				//System.out.println("finish loading pipeline");
				returnResponse.put("pipelineLoaded", pipelineName);
				returnResponse.put("message", "success");
			}
			
			
			if (fullRequest.get("pipeline").equals("addPR")){
				String pipelineName = fullRequest.get("pipelineName");
				String prName = fullRequest.get("prName");
				ConditionalSerialAnalyserController currentpipeline = gatePipelines.get(pipelineName);
				ProcessingResource currentPR = gatePrs.get(prName);
				currentpipeline.add(currentPR);
				gatePipelines.replace(pipelineName, currentpipeline);
				returnResponse.put("pradded", prName);
				returnResponse.put("message", "success");	
			}
			
			if (fullRequest.get("pipeline").equals("setCorpus")){
				String pipelineName = fullRequest.get("pipelineName");
				String corpusName = fullRequest.get("corpusName");
				ConditionalSerialAnalyserController currentpipeline = gatePipelines.get(pipelineName);
				Corpus currentCorpus = gateCorpus.get(corpusName);
				currentpipeline.setCorpus(currentCorpus);
				gatePipelines.replace(pipelineName, currentpipeline);
				returnResponse.put("corpusSetted", corpusName);
				returnResponse.put("message", "success");	
			}
			
			if (fullRequest.get("pipeline").equals("runPipeline")){
				String pipelineName = fullRequest.get("pipelineName");
				ConditionalSerialAnalyserController currentpipeline = gatePipelines.get(pipelineName);
				currentpipeline.execute();
				returnResponse.put("message", "success");	
			}
			
			if (fullRequest.get("pipeline").equals("checkParams")){
				String pipelineName = fullRequest.get("pipelineName");
				String prName = fullRequest.get("resourceName");
				String paramName = fullRequest.get("paramsName");
				
				ConditionalSerialAnalyserController currentpipeline = gatePipelines.get(pipelineName);
				ProcessingResource currentPR = gatePrs.get(prName);
				Object aaa = currentpipeline.getParameterValue(currentPR, paramName);
				//FeatureMap aaa = currentPR.getFeatures();
				//System.out.println("im in server");
				//System.out.println(aaa);
				returnResponse.put("message", aaa.toString());	
			}
			
			if (fullRequest.get("pipeline").equals("setParams")){
				//String pipelineName = fullRequest.get("pipelineName");
				String prName = fullRequest.get("resourceName");
				String paramName = fullRequest.get("paramsName");
				String paramType = fullRequest.get("paramsType");
				String rawValue = fullRequest.get("paramsValue");
			    ProcessingResource currentPR = gatePrs.get(prName);
			    /*
			    System.out.println(prName);
			    System.out.println(paramName);
			    System.out.println(paramType);
			    System.out.println(rawValue);
			    */
				
				
				if(paramType.matches("String")){
					String paramValue = rawValue;
					currentPR.setParameterValue(paramName, paramValue);				
				    returnResponse.put("message", "success");

				}
				else if(paramType.matches("List")) {
					String[] splitedValie = rawValue.split(",");
					List<String> paramValue = new ArrayList<String>();
					for(String current_value : splitedValie) {
						paramValue.add(current_value);
					}
					currentPR.setParameterValue(paramName, paramValue);
					
					System.out.println("success");
				    returnResponse.put("message", "success");

				}
				else if(paramType.matches("Boolean")){
					Boolean paramValue = Boolean.parseBoolean(rawValue);
					currentPR.setParameterValue(paramName, paramValue);				
				    returnResponse.put("message", "success");

				}
				else {
					returnResponse.put("message", "unknown type");
				}
				
			}			
			
		}
		
		//////////////////////corpus//////////////////////////////////////
		if (fullRequest.containsKey("corpus")){
			if (fullRequest.get("corpus").equals("createCorpus")){
				String corpusName = fullRequest.get("corpusName");
				Corpus currentCorpus = Factory.newCorpus(corpusName);
				gateCorpus.put(corpusName, currentCorpus);
				returnResponse.put("corpusAdded", corpusName);
				returnResponse.put("message", "success");
			}
			
			if (fullRequest.get("corpus").equals("addDocument")){
				String corpusName = fullRequest.get("corpusName");
				String docName = fullRequest.get("documentName");
				Corpus currentCorpus = gateCorpus.get(corpusName);
				Document currentDoc = gateDocList.get(docName);
				currentCorpus.add(currentDoc);
				gateCorpus.replace(corpusName, currentCorpus);
				returnResponse.put("documentAdded", docName);
				returnResponse.put("corpus", corpusName);
				returnResponse.put("message", "success");
			}
			
			if (fullRequest.get("corpus").equals("clearCorpus")){
				String corpusName = fullRequest.get("corpusName");
				Corpus currentCorpus = gateCorpus.get(corpusName);
				try {
				    currentCorpus.cleanup();
				    gateCorpus.remove(corpusName);				
     				returnResponse.put("corpus cleared", corpusName);
	    			returnResponse.put("message", "success");
			    }
				catch(Exception e) {
					returnResponse.put("message", "failed");
				}
			}
			
			
			
		}	
		
		
		/////////////////////////////////////////////////////////////////////
		
		//System.out.println(returnResponse);
		return returnResponse;
		
	}
}
