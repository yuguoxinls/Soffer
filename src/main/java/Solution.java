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
}
