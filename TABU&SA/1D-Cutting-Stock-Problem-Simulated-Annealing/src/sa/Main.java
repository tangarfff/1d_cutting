package sa;

import heurictic.RandomHeuristic;
import input.Hard28Input;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import neighborhood.NeighborsGenerator;
import neighborhood.RandomNeighborsGenerator;
import problemData.Data;
import wastesCalculator.WastesCalculator;
import wastesCalculator.WastesCalculatorInterface;

public class Main {

    public static void main(String[] args) throws IOException {
        Hard28Input input = new Hard28Input();
        input.read();
        List<Data> dataList = input.getDataList();
        for (Data data : dataList) {
            // https://www.cnblogs.com/SaraMoring/p/12688493.html
            List<Integer> solution = new RandomHeuristic(data).createSolution();
            //int[] s = {4,5,6,7,8};
            //List<Integer> solution = Arrays.stream(s).boxed().collect(Collectors.toList());
            WastesCalculatorInterface calculator = new WastesCalculator(data);
            NeighborsGenerator neighborsGenerator = new RandomNeighborsGenerator();
            SimulatedAnnealing annealing = new SimulatedAnnealing(solution, calculator, neighborsGenerator);
            int result = annealing.calculate(10000);
            System.out.println(data.getName() + ": " + result);
        }

    }

}
