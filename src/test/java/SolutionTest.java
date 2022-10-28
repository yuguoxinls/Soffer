import org.junit.Assert;
import org.junit.Test;

import java.awt.*;
import java.util.Arrays;

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

    @Test
    public void exchange() {
        int[] numbers = {};
        System.out.println(Arrays.toString(solution.exchange(numbers)));
    }

    @Test
    public void reverseList() {
        ListNode head = new ListNode(1);
        ListNode node1 = new ListNode(2);
        ListNode node2 = new ListNode(3);
        ListNode node3 = new ListNode(4);
        ListNode node4 = new ListNode(5);
        head.next = node1;
        node1.next = node2;
        node2.next = node3;
        node3.next = node4;
        System.out.println(solution.reverseList(head));
    }

    @Test
    public void spiralOrder() {
        int[][] matrix = {{1,2,3},{4,5,6},{7,8,9}};
        System.out.println(Arrays.toString(solution.spiralOrder(matrix)));
    }

    @Test
    public void majorityElement() {
        int[] nums = {1,2,3,2,2,2,5,4,2};
        Assert.assertEquals(2, solution.majorityElement(nums));
    }

    @Test
    public void getLeastNumbers() {
        int[] arr = {3,2,1};
        System.out.println(Arrays.toString(solution.getLeastNumbers(arr, 2)));
    }

    @Test
    public void maxSubArray(){
        int[] arr = {-2,1,-3,4,-1,2,1,-5,4};
        Assert.assertEquals(6, solution.maxSubArray(arr));
    }

    @Test
    public void firstUniqChar() {
        String s = "aadadaad";
        Assert.assertEquals(' ', solution.firstUniqChar(s));
    }

    @Test
    public void search() {
        int[] arr = {5,7,7,8,8,10};
        Assert.assertEquals(0, solution.search(arr, 1));
    }

    @Test
    public void testMissingNumber() {
        int[] nums = {0,1,3,4,5,6,7,8,9};
        Assert.assertEquals(2, solution.missingNumber(nums));
    }

    @Test
    public void testTwoSum() {
        int[] nums = {2,7,11,15};
        System.out.println(Arrays.toString(solution.twoSum(nums, 9)));
    }

    @Test
    public void testReverseWords() {
        String s = "    ";
        System.out.println(solution.reverseWords(s));
    }

    @Test
    public void testReverseLeftWords() {
        String s = "abcdefg";
        System.out.println(solution.reverseLeftWords(s, 2));
    }

    @Test
    public void testLastRemaining() {
        int n = 10, m = 17;
        System.out.println(solution.lastRemaining(n, m));
    }



    @Test
    public void exist() {
        char[][] board = {{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}};
        String word = "ABCCED";
        System.out.println(solution.exist(board, word));
    }

    @Test
    public void findContinuousSequence2() {
        System.out.println(Arrays.deepToString(solution.findContinuousSequence2(15)));
    }
}