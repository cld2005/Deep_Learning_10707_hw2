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

    private static HashMap<String, Integer> dict = new HashMap<>();
    private static HashMap< Integer, String> indexMap = new HashMap<>();
    private static HashMap<String, Integer> dictMap = new HashMap<>();
    private static HashMap<String, Integer> fourGramDict = new HashMap<>();

    public static void main(String[] args) {

        String filename = "train.txt";

        try {
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            String line;
            while ((line = reader.readLine()) != null) {
                String values[] = line.toLowerCase().split(" +");

                for (String v : values) {
                    dict.putIfAbsent(v, 0);
                    dict.put(v, dict.get(v) + 1);
                }
            }
            reader.close();
        } catch (Exception e) {
            System.err.format("Exception occurred trying to read '%s'.", filename);
            e.printStackTrace();
        }

        ArrayList<Map.Entry<String, Integer>> sortList = new ArrayList(dict.entrySet());

        Collections.sort(sortList, new Comparator<Map.Entry<String, Integer>>() {

            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                if (o2.getValue().compareTo(o1.getValue()) == 0) {
                    return o1.getKey().compareTo(o2.getKey());
                } else {
                    return o2.getValue().compareTo(o1.getValue());
                }
            }
        });

        File fout = new File("dict.txt");
        File foutFGI = new File("FGI.txt");
        FileOutputStream fos;
        FileOutputStream fosFGI;
        BufferedWriter bw;
        BufferedWriter bwCount;
        BufferedWriter bwFGI;
        dictMap.put("<start>", 7998);
        dictMap.put("<unk>", 7999);
        dictMap.put("<end>", 8000);
        indexMap.put(7998,"<start>");
        indexMap.put(7999,"<unk>");
        indexMap.put(8000,"<end>");

        try {
            fos = new FileOutputStream(fout);
            bw = new BufferedWriter(new OutputStreamWriter(fos));
            for (int i = 0; i < 8000 - 3; i++) {

                dictMap.put(sortList.get(i).getKey(), i + 1);
                indexMap.put(i+1,sortList.get(i).getKey());

                bw.write(sortList.get(i).getKey());
                bw.newLine();
            }

            bw.write("<start>");
            bw.newLine();

            bw.write("<unk>");
            bw.newLine();

            bw.write("<end>");
            bw.newLine();
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }


        try {
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            String line;
            while ((line = reader.readLine()) != null) {

                line = "<start> " + line + " <end>";
                String values[] = line.toLowerCase().split(" ");

                if (values.length < 4) {
                    continue;
                }
                for (int i = 0; i < values.length - 3; i++) {

                    StringBuilder FourGram = new StringBuilder();

                    for (int j = 0; j <= 3; j++) {

                        if (dictMap.containsKey(values[i + j])) {
                            FourGram.append(values[i + j]);
                        } else {
                            FourGram.append("<unk>");
                        }
                        FourGram.append(" ");
                    }

                    FourGram.deleteCharAt(FourGram.length() - 1);

                    String fg = FourGram.toString();
                    fourGramDict.putIfAbsent(fg, 0);

                    fourGramDict.put(fg, fourGramDict.get(fg) + 1);
                }
            }
            reader.close();
        } catch (Exception e) {
            System.err.format("Exception occurred trying to read '%s'.", filename);
            e.printStackTrace();
        }

        sortList = new ArrayList(fourGramDict.entrySet());

        Collections.sort(sortList, new Comparator<Map.Entry<String, Integer>>() {

            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                if (o2.getValue().compareTo(o1.getValue()) == 0) {
                    return o1.getKey().compareTo(o2.getKey());
                } else {
                    return o2.getValue().compareTo(o1.getValue());
                }
            }
        });

        fout = new File("fourGram50.txt");
        File fourGranCount = new File("fourGramCount.txt");
        FileOutputStream fosCount;
        try {
            fos = new FileOutputStream(fout);
            fosCount = new FileOutputStream(fourGranCount);
            fosFGI = new FileOutputStream(foutFGI);
            bw = new BufferedWriter(new OutputStreamWriter(fos));
            bwCount = new BufferedWriter(new OutputStreamWriter(fosCount));
            bwFGI = new BufferedWriter(new OutputStreamWriter(fosFGI));
            for (int i = 0; i < sortList.size(); i++) {
                if (i < 50) {
                    bw.write(sortList.get(i).getKey());
                    bw.newLine();
                }

                bwCount.write(String.valueOf(sortList.get(i).getValue()));
                bwCount.newLine();

                String[] values = sortList.get(i).getKey().split(" ");

                if (values.length != 4) {
                    System.out.println("four gram have " + values.length);
                } else {
                    StringBuilder sb = new StringBuilder();
                    for (String v : values) {

                        sb.append(dictMap.get(v));
                        sb.append(",");

                    }

                    sb.deleteCharAt(sb.length()-1);


                    bwFGI.write(sb.toString());
                    bwFGI.newLine();
                }


            }
            bwCount.close();
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
