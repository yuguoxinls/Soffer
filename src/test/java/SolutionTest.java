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

}