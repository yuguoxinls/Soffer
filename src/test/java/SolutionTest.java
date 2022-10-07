import org.junit.Assert;
import org.junit.Test;

public class SolutionTest {
    Solution solution = new Solution();

    @Test
    public void findRepeatNumberTest(){
        int[] nums = {2, 3, 1, 0, 2, 5, 3};
        Assert.assertEquals(2, solution.findRepeatNumber(nums));
    }

    @Test
    public void replaceSpaceTest(){
        String s = "We are happy.";
        System.out.println(solution.replaceSpace(s));
    }

    @Test
    public void reversePrint(){

    }

    @Test
    public void fib() {
        Assert.assertEquals(1, solution.fib(2));
        Assert.assertEquals(5, solution.fib(5));
        System.out.println(solution.fib(43));
    }

    @Test
    public void numWays() {
        /**
         * 输入：n = 2
         * 输出：2
         * 输入：n = 7
         * 输出：21
         * 输入：n = 0
         * 输出：1
         */
        Assert.assertEquals(2, solution.numWays(2));
        Assert.assertEquals(21, solution.numWays(7));
        Assert.assertEquals(1, solution.numWays(0));
    }

    @Test
    public void minArray() {
        /**
         * 输入：numbers = [3,4,5,1,2]
         *      * 输出：1
         *      * 输入：numbers = [2,2,2,0,1]
         *      * 输出：0
         */
        int[] numbers = {3,4,5,1,2};
        Assert.assertEquals(1, solution.minArray(numbers));
        numbers = new int[]{2, 2, 2, 0, 1};
        Assert.assertEquals(0, solution.minArray(numbers));
    }
}