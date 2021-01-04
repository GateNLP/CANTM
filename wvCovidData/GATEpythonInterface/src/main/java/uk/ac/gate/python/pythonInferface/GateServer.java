package uk.ac.gate.python.pythonInferface;
import java.io.*;
import java.net.*;
import java.util.Collections;
import java.util.HashMap;

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
import gate.creole.ResourceInstantiationException;
import gate.util.GateException;
import gate.util.persistence.PersistenceManager;
import org.json.*;

public class GateServer {
	public static void main(String[] args) throws Exception{
		String portNumber = args[0];
		GateServer interServer = new GateServer();
		interServer.run(portNumber);
	}
	
	public void run(String portNumber) throws Exception{
		int portid = Integer.parseInt(portNumber);
		String fromClient;
		String toClient;
		System.out.println("initiating Gate ");
		Gate.init();
		ServerSocket srvSock = new ServerSocket(portid);
		System.out.println("listening "+portNumber);
		gatePython gateObject = new gatePython();
		
		
		while(true){
			Socket clientSock = srvSock.accept();
			BufferedReader in = new BufferedReader(new InputStreamReader(clientSock.getInputStream()));
            PrintWriter out = new PrintWriter(clientSock.getOutputStream(),true);
            try{
            	HashMap<String, String> fullRequest = getPythonRequest(in);
            	//System.out.println(fullRequest);
            	
            	HashMap<String, String> returnResponse = gateObject.processRequest(fullRequest);
            	//System.out.println(gateObject.gateDocList);
            	//System.out.println("request received");            	
            	//toClient = "success, this is a very long sentencen, hahahahaha";
            	//toClient = "success";
            	sendPythonClient(in, out, returnResponse, 256);
            	
            }catch(Exception e){
            	//System.out.println(e.getMessage());
           		toClient = "Fail";
           	}                  
		}
	}
	
	public static HashMap<String, String> getPythonRequest(BufferedReader in) throws IOException, GateException{
		String fromClient;
		Boolean eov;
        HashMap<String, String> fullPythonRequest = new HashMap<String, String>();
        
        do{
    		fromClient = in.readLine();
    		JSONObject clientJson = new JSONObject(fromClient);
    		//System.out.println("readNewRequest");
    		//System.out.println(clientJson);
    		JSONObject jsonData = clientJson.getJSONObject("fromClient");
    		
    		eov = jsonData.getBoolean("eov");
    		for (String eachRequest:jsonData.keySet()){
    			if (eachRequest.equals("eov") == false){
    			    String partRequest = jsonData.getString(eachRequest);
    			    String currnetRequest;
    		  	    if (fullPythonRequest.containsKey(eachRequest)){
    				    currnetRequest = fullPythonRequest.get(eachRequest) + partRequest;
    				    fullPythonRequest.replace(eachRequest, currnetRequest);
    			    }
    			    else{
    				    currnetRequest = partRequest;
    			    }
    			    fullPythonRequest.put(eachRequest, currnetRequest);
    			}
    		}    		
    	}while(eov != true);
        //System.out.println("finish recive");
		return fullPythonRequest;
	}
	
	public void sendPythonClient(BufferedReader in, PrintWriter out, HashMap<String, String> responseKeyPair, int maxSendChar) throws IOException{
		Boolean eov = false;
		String partValue;
		JSONObject outJson = new JSONObject();
		String fromClient;
		for (String serverResponseKey:responseKeyPair.keySet()){
			int i = maxSendChar;
			int previ = 0;
			String fullResponse = responseKeyPair.get(serverResponseKey);
			int valueLength = fullResponse.length();
			do{
				if (i > valueLength){
					i=valueLength;
				}
				partValue = fullResponse.substring(previ, i);
				if (i < valueLength){
	        		eov = false;
	        	}
	        	else{
	        		eov = true;
	        	}
				outJson.put(serverResponseKey, partValue);
	        	outJson.put("eov", false);
	        	//System.out.print(outJson.toString());
	        	out.println(outJson.toString());
	        	fromClient = in.readLine();
	            //System.out.print(fromClient); 
	        	previ = i;
	        	i += maxSendChar;
	        	outJson = new JSONObject();
			}while(eov != true);
			
		}
		outJson.put("eov", true);
		out.println(outJson.toString());
		fromClient = in.readLine();
		//System.out.print(fromClient); 
	}
	
	public void sendPythonClient(BufferedReader in, PrintWriter out, String serverResponseKey, String serverResponseValue,int maxSendChar) throws IOException{
		Boolean eov = false;
        String partValue;
        String fromClient;
        int previ = 0;
        int maxChar = maxSendChar;
        int i = maxChar;
        int valueLength = serverResponseValue.length();
        JSONObject outJson = new JSONObject();
        do{
        	if(i>valueLength){
        		i=valueLength;
        	}
        	partValue = serverResponseValue.substring(previ, i);
        	if (i < valueLength){
        		eov = false;
        	}
        	else{
        		eov = true;
        	}
        	outJson.put(serverResponseKey, partValue);
        	outJson.put("eov", eov);
        	//System.out.print(outJson.toString());
        	out.println(outJson.toString());
        	fromClient = in.readLine();
            //System.out.print(fromClient); 
        	previ = i;
        	i += maxChar;
    	}while(eov != true);
	}

}
