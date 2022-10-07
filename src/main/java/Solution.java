import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

public class Solution {
    /**
     * 找出数组中重复的数字。
     * 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，
     * 但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
     * 输入：[2, 3, 1, 0, 2, 5, 3]
     * 输出：2 或 3
     */
    public int findRepeatNumber(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            /*boolean add = set.add(num);
            if (!add){
                return num;
            }*/
            if (set.contains(num)){ //对数组中的每个元素，如果该元素已经存在于集合中，说明该元素是重复的，直接返回该元素
                return num;
            }
            set.add(num); //将其添加到集合中
        }
        return -1;
    }

    /**
     * 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
     * 输入：s = "We are happy."
     * 输出："We%20are%20happy."
     */
    public String replaceSpace(String s) {
        StringBuilder res = new StringBuilder(); // StringBuilder是一个可变的字符串类，我们可以把它看成是一个容器，处理字符串性能比String好
        for (int i = 0; i < s.length(); i++) {  // 用res存放最后结果，遍历给定字符串，如果是空格，就向res中添加"%20"，不是空格就添加原字符
            char c = s.charAt(i);
            if (c == ' '){
                res.append("%20");
            }else {
                res.append(c);
            }
        }
        return res.toString();
    }

    /**
     * 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
     * 输入：head = [1,3,2]
     * 输出：[2,3,1]
     */
    public int[] reversePrint(ListNode head) {
        Stack<Integer> stack = new Stack<>();
        ListNode handler = head;
        while (handler != null){
            stack.push(handler.val);
            handler = handler.next;
        }
        int[] res = new int[stack.size()];
        for (int i = 0; i < res.length; i++) {
            res[i] = stack.pop();
        }
        return res;
    }

    /**
     * 写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：
        *  F(0) = 0, F(1)= 1
        *  F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
     * 斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。
     * 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
     * 输入：n = 2
     * 输出：1
     * 输入：n = 5
     * 输出：5
     */
    public int fib(int n) {
        // 第一反应想到的是递归，但是随着n的增大，递归需要消耗大量的资源，导致超时
        /*if (n == 0){
            return 0;
        }
        if (n == 1){
            return 1;
        }
        long res = fib(n-1) + fib(n-2);
        if (res >= (1e9+7)){
            return (int) (res % (1e9+7));
        }else {
            return (int) res;
        }*/
        // 采用动态规划，使用3个int类型数据来存储斐波那契数列，这样省去了每次递归计算的资源浪费
        // 以斐波那契数列性质 f(n + 1) = f(n) + f(n - 1)f(n+1)=f(n)+f(n−1) 为转移方程。
        int a = 0, b = 1, sum;
        for (int i = 0; i < n; i++) {
            sum = (a+b) % 1000000007;
            a = b;
            b = sum;
        }
        return a;
    }

    /**
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
     * 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
     * 输入：n = 2
     * 输出：2
     * 输入：n = 7
     * 输出：21
     * 输入：n = 0
     * 输出：1
     */
    public int numWays(int n) {
        // 此类求 多少种可能性 的题目一般都有 递推性质 ，即 f(n) 和 f(n-1)...f(1) 之间是有联系的。
        // 假设跳上一个 n 级的台阶总共有f(n)种跳法，而在完成n级台阶的最后一跳时，只会有两种可能
        //      1. 跳1级台阶，则之前的n-1级台阶有f(n-1)种跳法
        //      2. 跳2级台阶，则之前的n-2级台阶有f(n-2)种跳法
        // 因此，根据分类加法原理，f(n) = f(n-1) + f(n-2)
        // 可见，这是一个斐波那契数列问题：f(0)=1, f(1)=1, f(2)=2
        int a = 1, b = 1, sum;
        for (int i = 0; i < n; i++) {
            sum = (a+b) % 1000000007;
            a = b;
            b = sum;
        }
        return a;
    }

    /**
     * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     * 给你一个可能存在重复元素值的数组numbers，它原来是一个升序排列的数组，并按上述情形进行了一次旋转。请返回旋转数组的最小元素。
     * 例如，数组[3,4,5,1,2] 为 [1,2,3,4,5] 的一次旋转，该数组的最小值为 1。
     * 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
     * 输入：numbers = [3,4,5,1,2]
     * 输出：1
     * 输入：numbers = [2,2,2,0,1]
     * 输出：0
     */
    public int minArray(int[] numbers) {
        for (int i = 0; i < numbers.length-1; i++) {
            if (numbers[i] > numbers[i+1]){
                return numbers[i+1];
            }
        }
        return numbers[0];
    }

    /**
     * 编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数
     * 输入：n = 11 (控制台输入 00000000000000000000000000001011)
     * 输出：3
     * 解释：输入的二进制串 00000000000000000000000000001011中，共有三位为 '1'。
     * 示例 2：
     * 输入：n = 128 (控制台输入 00000000000000000000000010000000)
     * 输出：1
     * 解释：输入的二进制串 00000000000000000000000010000000中，共有一位为 '1'。
     * 示例 3：
     * 输入：n = 4294967293 (控制台输入 11111111111111111111111111111101，部分语言中 n = -3）
     * 输出：31
     * 解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。
     */
    public int hammingWeight(int n) {
        // 根据 与运算 定义，设二进制数字 n ，则有：
        // 若 n&1=0 ，则 n 二进制 最右一位 为 0 ；
        // 若 n&1=1 ，则 n 二进制 最右一位 为 1 。
        // 根据以上特点，考虑以下 循环判断 ：
        // 判断 n 最右一位是否为 1 ，根据结果计数。
        // 将 n 右移一位（本题要求把数字 n 看作无符号数，因此使用 无符号右移 操作）。
        int res = 0;
        while (n != 0){
            res = res + (n&1);
            n >>>= 1;
        }
        return res;
    }

    /**
     * 输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。
     */
    public int[] printNumbers(int n) {
        int count = (int) (Math.pow(10, n) - 1);
        int[] res = new int[count];
        for (int i = 0; i < res.length; i++) {
            res[i] = i+1;
        }
        return res;
    }
}
