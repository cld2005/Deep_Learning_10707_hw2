import java.util.HashMap;
import java.util.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.*;
import java.nio.file.Files;

/**
 * Created by lindichen on 11/7/17.
 */
public class Preprocessing {

    private static HashMap<String, Integer > dict = new HashMap<>();
    private static HashSet<String > dictSet = new HashSet<>();
    private static HashMap<String, Integer > fourGramDict = new HashMap<>();
    public static void main (String [] args){

        String filename = "train.txt";

        try
        {
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            String line;
            while ((line = reader.readLine()) != null)
            {
                String values [] = line.toLowerCase().split(" +");

                for(String v:values){
                    dict.putIfAbsent(v,0);
                    dict.put(v,dict.get(v)+1);
                }
            }
            reader.close();
        }
        catch (Exception e)
        {
            System.err.format("Exception occurred trying to read '%s'.", filename);
            e.printStackTrace();
        }

        ArrayList<Map.Entry<String, Integer>> sortList = new ArrayList(dict.entrySet());

        Collections.sort(sortList, new Comparator<Map.Entry<String, Integer>>(){

            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                if (o2.getValue().compareTo(o1.getValue())==0){
                    return o1.getKey().compareTo(o2.getKey());
                }
                else{
                    return o2.getValue().compareTo(o1.getValue()) ;
                }
            }});

        StringBuilder sb = new StringBuilder();
        File fout = new File("out.txt");

        FileOutputStream fos;
        BufferedWriter bw;
        BufferedWriter bwCount;
        dictSet.add("<end>");
        dictSet.add("<start>");
        dictSet.add("<unk>");
        try{
            fos = new FileOutputStream(fout);
            bw = new BufferedWriter(new OutputStreamWriter(fos));
            for (int i=0;i<8000-3;i++){

                dictSet.add(sortList.get(i).getKey());

                bw.write(sortList.get(i).getKey()+","+sortList.get(i).getValue());
                bw.newLine();
            }
            bw.close();
        }catch (Exception e){
            e.printStackTrace();
        }


        try
        {
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            String line;
            while ((line = reader.readLine()) != null)
            {

                line= "<start> "+line+" <end>";
                String values [] = line.toLowerCase().split(" ");

                if(values.length<4){
                    continue;
                }
                for(int i=0;i<values.length-3;i++){

                    StringBuilder FourGram = new StringBuilder();

                    for(int j=0;j<=3;j++){

                        if (dictSet.contains(values[i+j])){
                            FourGram.append(values[i+j]);
                        }else{
                            FourGram.append("<unk>");
                        }
                        FourGram.append(" ");
                    }

                    FourGram.deleteCharAt(FourGram.length()-1);

                    String fg = FourGram.toString();
                    fourGramDict.putIfAbsent(fg,0);

                    fourGramDict.put(fg,fourGramDict.get(fg)+1);
                }
            }
            reader.close();
        }
        catch (Exception e)
        {
            System.err.format("Exception occurred trying to read '%s'.", filename);
            e.printStackTrace();
        }

        sortList = new ArrayList(fourGramDict.entrySet());

        Collections.sort(sortList, new Comparator<Map.Entry<String, Integer>>(){

            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                if (o2.getValue().compareTo(o1.getValue())==0){
                    return o1.getKey().compareTo(o2.getKey());
                }
                else{
                    return o2.getValue().compareTo(o1.getValue()) ;
                }
            }});

        fout=new File("fourGram50.txt");
        File fourGranCount=new File("fourGramCount.txt");
        FileOutputStream fosCount;
        try{
            fos = new FileOutputStream(fout);
            fosCount = new FileOutputStream(fourGranCount);
            bw = new BufferedWriter(new OutputStreamWriter(fos));
            bwCount = new BufferedWriter(new OutputStreamWriter(fosCount));
            for (int i=0;i<sortList.size();i++){
                if(i<50){
                    bw.write(sortList.get(i).getKey());
                    bw.newLine();
                }

                bwCount.write(String.valueOf(sortList.get(i).getValue()));
                bwCount.newLine();
            }
            bwCount.close();
            bw.close();
        }catch (Exception e){
            e.printStackTrace();
        }

    }
}
