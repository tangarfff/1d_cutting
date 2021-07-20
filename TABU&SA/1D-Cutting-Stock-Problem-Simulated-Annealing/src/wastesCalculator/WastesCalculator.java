package wastesCalculator;

import java.util.*;

import problemData.DataInterface;

public class WastesCalculator implements WastesCalculatorInterface {

    private DataInterface data;

    public WastesCalculator(DataInterface data) {
        this.data = data;
    }

    @Override
    public int calculate(List<Integer> currentSolution) {
        int waste = 0; //花费的钢条

        /* Iterator<Integer> iterator = currentSolution.iterator();
        int usedStack = 0;
        while (iterator.hasNext()) {
            int stackLength = iterator.next();
            if (usedStack - stackLength < 0) {
                waste++;
                usedStack = data.getStockLength() - stackLength;
            } else
                usedStack = usedStack - stackLength;
        }*/
        //Collections.sort(currentSolution);
        //Collections.reverse(currentSolution);
        Iterator<Integer> iterator = currentSolution.iterator();
        int flag;
        Map<Integer, Integer> box_old = new HashMap<Integer, Integer>();
        int[] boxes = new int[currentSolution.size()];
        while (iterator.hasNext()) {
            int stackLength = iterator.next();
            flag = 0;
            for(Map.Entry<Integer, Integer> entry:box_old.entrySet()) {
                if (stackLength + entry.getValue() <= data.getStockLength()) {
                    box_old.replace(entry.getKey(), stackLength + entry.getValue());
                    flag = 1;
                    break;
                }
            }
            if(flag == 1){
                continue;
            }
            else {
                boxes[box_old.size()]+=stackLength;
                box_old.put(box_old.size(),boxes[box_old.size()]);
                waste++;
            }
        }
        return waste;
    }

}
