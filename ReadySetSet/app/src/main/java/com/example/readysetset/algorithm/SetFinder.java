package com.example.readysetset.algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class SetFinder {
    static List<Set<Integer>> findSet(List<Integer[]> inputCards) {
        List<Set<Integer>> retVal = new ArrayList<>();

        for (int i = 0; i < inputCards.size(); i++) {
            for (int j = i+1; j < inputCards.size(); j++) {
                Integer[] c1 = inputCards.get(i);
                Integer[] c2 = inputCards.get(j);
                // Calculate card that would complete the set
                Integer[] matches = new Integer[c1.length];
                for (int k = 0; k < matches.length; k++) {
                    matches[k] = c1[k]==c2[k] ? c1[k] : (c1[k]^c2[k])%3;
                }
                for (int k = 0; k < inputCards.size(); k++) {
                    if (Arrays.equals(inputCards.get(k), matches)) {
                        Set<Integer> set = new HashSet<>();
                        set.add(i);
                        set.add(j);
                        set.add(k);
                        retVal.add(set);
                    }
                }
            }
        }
        return retVal;
    }
}
