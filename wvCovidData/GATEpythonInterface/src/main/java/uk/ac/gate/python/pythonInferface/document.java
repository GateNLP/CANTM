package uk.ac.gate.python.pythonInferface;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import gate.Gate;
import gate.Factory;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;
import gate.util.GateException;

public class document {
	public static Document loadDocument(String docUrl) throws IOException, GateException{		
		Document doc = Factory.newDocument(new URL(docUrl));
		return doc;
	}
	
	public static Document loadDocumentFromFile(String docUrl) throws IOException, GateException{		
		File inputFile=new File(docUrl);
		Document doc = Factory.newDocument(inputFile.toURI().toURL());
		return doc;
	}
	
	
	public static HashMap<String, String> documentOperations(HashMap<String, String> fullRequest, HashMap<String, Document> gateDocList) throws IOException, GateException{	
		HashMap<String, String> returnResponse = new HashMap<String, String>();
		String operation = fullRequest.get("document");
		if (operation.equals("getDocumentContent")){
			String docName = fullRequest.get("docName");
			String docContent = getDocumentContent(gateDocList.get(docName));
			returnResponse.put("docContent", docContent);
			returnResponse.put("docName", docName);
		}
		
		if (operation.equals("getAnnotationSetName")){
			String docName = fullRequest.get("docName");
			Set<String> annotationSetNames = gateDocList.get(docName).getAnnotationSetNames();
			returnResponse.put("annotationSetName", annotationSetNames.toString());			
		}
		
		if (operation.equals("getAnnotations")){
			String docName = fullRequest.get("docName");
			String annotationSetName = fullRequest.get("annotationSetName");
			List<Annotation> ats = gateDocList.get(docName).getAnnotations(annotationSetName).inDocumentOrder();
			//ats = ats.inDocumentOrder();
			returnResponse.put("annotationSet", ats.toString());								
		}
		
		
		return returnResponse;
	}
	
	public static String getDocumentContent(Document inputDoc) throws IOException, GateException{
		String docContent = inputDoc.getContent().toString();
		return docContent;
	}
}
