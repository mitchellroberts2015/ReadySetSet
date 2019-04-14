package com.example.readysetset.algorithm;

import java.util.ArrayList;
import java.util.Arrays;

public class SetFinder {
    static ArrayList<Integer> findSent(ArrayList<Integer[]> inputCards) {
        ArrayList<Integer> retVal = new ArrayList<>();

        for (int i = 0; i < inputCards.size(); i++) {
            for (int j = i+1; j < inputCards.size(); j++) {
                Integer[] c1 = inputCards.get(i);
                Integer[] c2 = inputCards.get(j);
                // Calculate card that would complete the set
                Integer[] matches = new Integer[c1.length];
                for (int k = 0; k < matches.length; k++) {
                    matches[k] = c1[k]==c2[k] ? c1[k] : c1[k]^c2[k];
                }
                for (int k = 0; k < inputCards.size(); k++) {
                    if (Arrays.equals(inputCards.get(k), matches)) {
                        retVal.add(k);
                    }
                }
            }
        }
        return retVal;
    }
}
