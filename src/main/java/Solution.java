import java.util.*;

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
    public String replaceSpace(String s) { //TODO 重点看
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
    public int fib(int n) { //TODO 重点看
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
    public int numWays(int n) { //TODO 重点看
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
    public int hammingWeight(int n) { //TODO 重点看
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

    /**
     * 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
     * 返回删除后的链表的头节点。
     * head = [4,5,1,9], val = 5
     * 输出: [4,1,9]
     */
    public ListNode deleteNode(ListNode head, int val) {
        if (head == null){
            return null;
        }
        if (head.val == val){
            return head.next;
        }
        ListNode tmp = head;
        while ((tmp != null)&&(tmp.next != null)){
            if (tmp.next.val == val){
                //
                tmp.next = tmp.next.next;
            }
            tmp = tmp.next;
        }
        return head;
    }
    /*public ListNode deleteNode(ListNode head, int val) {
        ListNode tmp = head;
        if (tmp.val == val){    //判断头部特殊情况
            return head.next;
        }
        while (tmp.next.val!=val){    // 将指针移动到待删除元素的前一个位置
            tmp = tmp.next;
        }
        tmp.next = tmp.next.next;    // 删除
        return head;
    }*/

    /**
     * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数在数组的前半部分，所有偶数在数组的后半部分。
     * 输入：nums = [1,2,3,4]
     * 输出：[1,3,2,4]
     * 注：[3,1,2,4] 也是正确的答案之一。
     */
    /*public int[] exchange(int[] nums) {
        List<Integer> oddList = new ArrayList<>();
        List<Integer> evenList = new ArrayList<>();
        for (int num : nums) {
            if (num%2==0){
                evenList.add(num);
            }else {
                oddList.add(num);
            }
        }
        int len = oddList.size() + evenList.size();
        int[] res = new int[len];
        for (int i = 0; i < oddList.size(); i++) {
            res[i] = oddList.get(i);
        }
        for (int i = 0; i < evenList.size(); i++) {
            res[i + (len-evenList.size())] = evenList.get(i);
        }
        return res;
    }*/
    /**
     * 题解：考虑定义双指针 i , j 分列数组左右两端，循环执行：
     *  指针 i 从左向右寻找偶数；
     *  指针 j 从右向左寻找奇数；
     *  将 偶数 nums[i] 和 奇数 nums[j] 交换。
     */
    public int[] exchange(int[] nums) { //TODO 重点看
        int i = 0;
        int j = nums.length - 1;
        int tmp;
        while (i<j){
            while ((i<j)&&(nums[i]%2!=0)) i++;
            while ((i<j)&&(nums[j]%2==0)) j--;
            tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
        }
        return nums;
    }

    /**
     * 输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。
     * 例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。
     * 给定一个链表: 1->2->3->4->5, 和 k = 2.
     * 返回链表 4->5.
     */
    public ListNode getKthFromEnd(ListNode head, int k) {
        if (head == null){
            return null;
        }
        //1. get length
        int len = 0;
        ListNode tmp = head;
        while (tmp != null){
            len++;
            tmp = tmp.next;
        }
        //2. move head
        int step = len - k;
        while (step != 0){
            head = head.next;
            step--;
        }
        return head;
    }

    /**
     * 定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
     * 输入: 1->2->3->4->5->NULL
     * 输出: 5->4->3->2->1->NULL
     */
    public ListNode reverseList(ListNode head) { //TODO 重点看
        ListNode cur = head, pre = null; //双指针，一个指向头节点，一个指向最后
        while (cur != null){
            ListNode tmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmp;
        }
        return pre;
    }

    /**
     * 输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。
     * 输入：1->2->4, 1->3->4
     * 输出：1->1->2->3->4->4
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) { //TODO 重点看
        // 显然要定义两个指针来遍历两个链表，每次取两个指针中的最小值，放到合并后的链表中
        // 这里的重点是：合并后的链表没有第一个节点，导致无法将获得的第一个节点存到合并后的链表中
        // 这时可以定义一个伪头节点，作为合并后链表的第一个节点
        ListNode dum = new ListNode(0);
        ListNode cur = dum;
        while (l1 != null && l2 != null){
            if (l1.val <= l2.val){
                cur.next = l1;
                cur = cur.next;
                l1 = l1.next;
            }else {
                cur.next = l2;
                cur = cur.next;
                l2 = l2.next;
            }
        }
        if (l2 == null){
            cur.next = l1;
        }
        if (l1 == null){
            cur.next = l2;
        }
        return dum.next;
    }

    /**
     * 请完成一个函数，输入一个二叉树，该函数输出它的镜像。
     * 输入：root = [4,2,7,1,3,6,9]
     * 输出：[4,7,2,9,6,3,1]
     */
    //TODO 重点看
    public TreeNode mirrorTree(TreeNode root) { // mirrorTree会对当前的二叉树镜像
        // 看成一个递归问题，交换每个节点的左 / 右子节点，即可生成二叉树的镜像
        if (root == null) return null; // 特例
        TreeNode tmp = root.left;
        root.left = mirrorTree(root.right); // 把根节点的右子树镜像，交给左子树
        root.right = mirrorTree(tmp);  // 把根节点的左子树镜像，交给右子树
        return root; // 返回根节点
    }

    /**
     * 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
     * 输入：root = [1,2,2,3,4,4,3]
     * 输出：true
     * 输入：root = [1,2,2,null,3,null,3]
     * 输出：false
     */
    public boolean isSymmetric(TreeNode root) { //TODO 重点看
        // 题解：二叉树对称，当且仅当其左右子树对称
            // L.val == R.val
            // L.left.val == R.right.val
            // L.right.val == R.left.val
            // 上述三步，用一个函数来表示，专门来判断每对节点是否对称
        return root == null || recur(root.left, root.right);
    }
    private boolean recur(TreeNode L, TreeNode R) {
        if (L == null && R == null) return true;
        if (L == null || R == null || L.val != R.val) return false;
        return recur(L.left, R.right) && recur(L.right, R.left);
    }

    /**
     * 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
     * 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
     * 输出：[1,2,3,6,9,8,7,4,5]
     */
    public int[] spiralOrder(int[][] matrix) { //TODO 重点中的重点！！！
        // 题解：按照题目要求，打印有四个方向
            //1. 从左到右
            //2. 从上到下
            //3. 从右到左
            //4. 从下到上
        // 这样就完成了一圈的打印，因此可定义4个变量，来表示边界
        // 可以使用 x++ 和 ++x 的不同来提高效率
        if (matrix.length == 0) return new int[0]; //特殊情况
        int l = 0, t = 0, r = matrix[0].length-1, b = matrix.length-1;
        int[] res = new int[(b+1) * (r+1)];
        int x = 0; //表示res数组的索引
        while (true){
            // left to right
            for (int i = l; i <= r; i++) {
                res[x++] = matrix[t][i]; // 先执行表达式，最后对x进行自增
//                x = x + 1;
            }
//            t = t + 1;
            if (++t > b) break; // 先自增，再执行表达式
            // top to bottom
            for (int i = t; i <= b; i++) {
                res[x++] = matrix[i][r];
//                x = x + 1;
            }
//            r = r - 1;
            if (l > --r) break;
            // right to left
            for (int i = r; i >= l; i--) {
                res[x++] = matrix[b][i];
//                x = x + 1;
            }
//            b = b - 1;
            if (t > --b) break;
            // bottom to top
            for (int i = b; i >= t; i--) {
                res[x++] = matrix[i][l];
//                x = x + 1;
            }
//            l = l + 1;
            if (++l > r) break;
        }
        return res;

    }


}
