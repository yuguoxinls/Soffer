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
    public String replaceSpace(String s) { //TODO 重点看 +1
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
    public int fib(int n) { //TODO 重点看 +1
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
        // 采用 **动态规划**，使用3个int类型数据来存储斐波那契数列，这样省去了每次递归计算的资源浪费
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
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
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
    public boolean isSymmetric(TreeNode root) {
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

    /**
     * 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
     * 例如: 给定二叉树: [3,9,20,null,null,15,7],
     * 返回其层次遍历结果：
     * [
     *   [3],
     *   [9,20],
     *   [15,7]
     * ]
     */
    /*public List<List<Integer>> f(TreeNode root) {
        if (root == null) return new ArrayList<>();
        List<List<Integer>> ans = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            List<Integer> tmp = new ArrayList<>(); // 存放每一层节点的值
            int queueSize = queue.size();
            while (queueSize > 0){
                TreeNode node = queue.poll();
                tmp.add(node.val);
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
                queueSize--;
            }
            ans.add(tmp);
        }
        return ans;
    }*/
    public List<List<Integer>> levelOrder(TreeNode root) { // TODO 看！！！！！！
        // 题目要求的二叉树的 从上至下 打印（即按层打印），又称为二叉树的 广度优先搜索（BFS）。BFS 通常借助 队列 的先入先出特性来实现。
        Queue<TreeNode> queue = new LinkedList<>(); // 用于存放待打印的节点
        List<List<Integer>> res = new ArrayList<>(); // 用于存放最后结果
        if (root != null) queue.add(root); // 先把根节点放进去
        while (!queue.isEmpty()){
            List<Integer> tmp = new ArrayList<>(); //用于存放当前层的所有节点值
            for (int i = queue.size(); i >0 ; i--) { // 由于queue中存放了待打印的节点，因此其长度为当前层所需要的数量，也就是循环这么多次，向tmp中添加这么多次数的值
                TreeNode node = queue.poll(); // 每一次从队列中取出节点
                tmp.add(node.val); // 获取值，放到tmp中
                if (node.left != null) queue.add(node.left); // 当前节点的值获取完毕，把他的左右节点添加到队列中，待后续循环处理
                if (node.right != null) queue.add(node.right);
            } // for循环结束后，tmp中存放了当前层的所有节点的值
            res.add(tmp);
        }
        return res;
    }

    /**
     * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
     * 输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
     * 输出: 2
     */
    /**
     * 本题常见的三种解法：
     * 哈希表统计法： 遍历数组 nums ，用 HashMap 统计各数字的数量，即可找出 众数 。此方法时间和空间复杂度均为 O(N)。
     * 数组排序法： 将数组 nums 排序，数组中点的元素 一定为众数。
     * 摩尔投票法： 核心理念为 票数正负抵消 。此方法时间和空间复杂度分别为 O(N)和 O(1)，为本题的最佳解法。
     */
    public int majorityElement(int[] nums) {
        // 方法一：hashMap by myself
        /*int target = nums.length/2;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)){
                Integer value = map.get(num);
                map.put(num, ++value);
            }else {
                map.put(num, 1);
            }
        }
        Set<Integer> set = map.keySet();
        for (Integer key : set) {
            Integer targetValue = map.get(key);
            if (targetValue > target) return key;
        }
        return -1;*/
        // 方法二：对数组排序
        /*for (int i = 0; i < nums.length-1; i++) {
            for (int j = 0; j < nums.length - 1; j++) {
                if (nums[j] > nums[j+1]){
                    int tmp = nums[j];
                    nums[j] = nums[j+1];
                    nums[j+1] = tmp;
                }
            }
        } // 冒泡排序效率太低，数组很大的话，时间空间复杂度大，可更换其他排序方法
        return nums[nums.length/2];*/
        // 方法三：摩尔投票法
        int x = 0, votes = 0; //TODO 太妙了！！！
        for(int num : nums){
            if(votes == 0) x = num;
            votes += num == x ? 1 : -1; // 如果num == x，votes就自增1；否则自减1
        }
        return x;
    }

    /**
     * 输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
     */
    public int[] getLeastNumbers(int[] arr, int k) { // TODO 待优化，使用堆的思想(官方)或者基于快排的数组划分(K神)
        for (int i = 0; i < arr.length-1; i++) {
            for (int j = 0; j < arr.length - 1; j++) {
                if (arr[j] > arr[j+1]){
                    int tmp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = tmp;
                }
            }
        }
        return Arrays.copyOfRange(arr, 0, k);
    }

    /**
     * 输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
     * 输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
     * 输出: 6
     * 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
     */
    public int maxSubArray(int[] nums) { //TODO 动态规划
        /**
         * 动态规划
         * 1.状态，即子问题。
         * dp[i] 代表以元素 nums[i] 为结尾的连续子数组最大和。
         * 2.转移策略，自带剪枝。
         * 若 dp[i−1]≤0 ，说明 dp[i−1] 对 dp[i] 产生负贡献，即 dp[i−1]+nums[i] 还不如 nums[i] 本身大。
         * 3.状态转移方程，根据前两步抽象而来。
         * 当 dp[i−1]>0 时：执行 dp[i] = dp[i-1] + nums[i]；
         * 当 dp[i−1]≤0 时：执行 dp[i] = nums[i] ；
         * 4.设计dp数组，保存子问题的解，避免重复计算
         * 5.实现代码
         * 整个动态规划，最难的就是定义状态。一旦状态定义出来，表明你已经抽象出了子问题，可以肢解原来的大问题了。
         */
        /*int res = nums[0];
        for (int i = 1; i < nums.length; i++) { // 看不懂，哭了
            nums[i] = nums[i] + Math.max(nums[i-1], 0);
            res = Math.max(res, nums[i]);
        }
        return res;*/
        List<Integer> res = new ArrayList<>();
        res.add(nums[0]);
        for (int i = 1; i < nums.length; i++) {
            if (res.get(i-1) < 0){
                res.add(nums[i]);
            }else {
                res.add(res.get(i-1) + nums[i]);
            }
        }
        int max = res.get(0);
        for (Integer data : res) {
            if (data > max) max = data;
        }
        return max;
    }

    /**
     * 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。
     * 输入：s = "abaccdeff"
     * 输出：'b'
     * 输入：s = ""
     * 输出：' '
     */
    public char firstUniqChar(String s) {
        Map<Character, Integer> map = new LinkedHashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (map.containsKey(c)){
                Integer value = map.get(c);
                map.put(c, ++value);
            }else {
                map.put(c, 1);
            }
        }
        if (s.isEmpty()) return ' ';
        Set<Character> characters = map.keySet();
        for (Character character : characters) {
            if (map.get(character) == 1) return character;
        }
        return ' ';
    }

    /**
     * 输入两个链表，找出它们的第一个公共节点。
     */
    ListNode getIntersectionNode(ListNode headA, ListNode headB) { //TODO
        /**
         * 设「第一个公共节点」为 node「链表 headA」的节点数量为 a「链表 headB」的节点数量为 b「两链表的公共尾部」的节点数量为 c，则有：
         * 头节点 headA 到 node 前，共有 a−c个节点；
         * 头节点 headB 到 node 前，共有 b−c个节点；
         * 考虑构建两个节点指针 A, B 分别指向两链表头节点 headA , headB ，做如下操作：
         * 指针 A 先遍历完链表 headA ，再开始遍历链表 headB ，当走到 node 时，共走步数为：a+(b−c)
         * 指针 B 先遍历完链表 headB ，再开始遍历链表 headA ，当走到 node 时，共走步数为：b+(a−c)
         * 此时指针 A , B 重合，并有两种情况：
         *  若两链表 有 公共尾部 (即 c>0) ：指针 A , B 同时指向「第一个公共节点」node 。
         *  若两链表 无 公共尾部 (即 c=0) ：指针 A , B 同时指向 null。
         * 因此返回 A 即可。
         */
        ListNode A = headA, B = headB;
        while (A != B){
            A = A != null ? A.next : headB;
            B = B != null ? B.next : headA;
        }
        return A;
    }

    /**
     * 统计一个数字在排序数组中出现的次数。
     * 输入: nums = [5,7,7,8,8,10], target = 8
     * 输出: 2
     * 输入: nums = [5,7,7,8,8,10], target = 6
     * 输出: 0
     */
    public int search(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        int mid = 0;
        while (l <= r){
            mid = (l+r)/2;
            if (target == nums[mid]) {
                break;
            }else if (target > nums[mid]){
                l = mid + 1;
            }else {
                r = mid - 1;
            }

        }
        int count = 0;
        for (int i = mid; i < nums.length; i++) {
            if (nums[i] == target) {
                count++;
            }else {
                break;
            }
        }
        for (int i = mid - 1; i >= 0; i--) {
            if (nums[i] == target) {
                count++;
            }else {
                break;
            }
        }
        return count;
    }

    /**
     * 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围 0～n-1之内。
     * 在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
     * 输入: [0,1,3]
     * 输出: 2
     * 输入: [0,1,2,3,4,5,6,7,9]
     * 输出: 8
     */
    public int missingNumber(int[] nums) { //TODO 需要判断特殊情况，缺失的是第一项或者最后一项
        /*if (nums[0] == 1){
            //代表缺失的是第一项
            return 0;
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i) return i;
        }
        return nums.length; //执行到这，说明前面条件都不满足，因此缺失的是最后一项*/
        // 尝试使用二分法，自己写的这个效率低
        /*if (nums[0] == 1) return 0;
        int l = 0;
        int r = nums.length-1;
        while (l<=r){
            int mid = (l+r)/2;
            if (nums[mid] != mid){
                if (nums[mid - 1] == mid - 1) return mid;
                r = mid - 1;
            }else {
                l = mid + 1;
            }
        }
        return nums.length;*/
        //TODO K神二分法
        int l = 0;
        int r = nums.length-1;
        while (l<=r){
            int mid = (l+r)/2;
            if (nums[mid] == mid) l = mid + 1;
            else r = mid - 1;
        }
        return l; // 最后返回 l 太惊艳了！这个条件自己像是想不出来的，要举个例子试一下
    }

    /**
     * 给定一棵二叉搜索树，请找出其中第 k 大的节点的值。
     * 输入: root = [3,1,4,null,2], k = 1
     *    3
     *   / \
     *  1   4
     *   \
     *    2
     * 输出: 4
     * 输入: root = [5,3,6,2,4,null,null,1], k = 3
     *        5
     *       / \
     *      3   6
     *     / \
     *    2   4
     *   /
     *  1
     * 输出: 4
     */
    /*int res, k;
    public int kthLargest(TreeNode root, int k) { // TODO K神解答，看不懂。。。
        *//**
         * 二叉搜索树的中序遍历得到的是升序序列：左、根、右，算法如下：
         * // 打印中序遍历
         * void dfs(TreeNode root) {
         *     if(root == null) return;
         *     dfs(root.left); // 左
         *     System.out.println(root.val); // 根
         *     dfs(root.right); // 右
         * }
         * 为适应本题，可改造上述算法：
         *  1. 顺序改为右、根、左
         *  2. 递归遍历时计数，统计当前节点的序号
         *  3. 递归到第 k个节点时，应记录结果 res ；
         *  4. 记录结果后，后续的遍历即失去意义，应提前终止（即返回）。
         *//*
        this.k = k;
        dfs(root);
        return res;
    }
    void dfs(TreeNode root){
        if (root == null) return;
        dfs(root.right);
        if (k == 0) return;
        if (--k == 0) res = root.val;
        dfs(root.left);
    }*/
    List<Integer> res = new ArrayList<>();
    public int kthLargest(TreeNode root, int k) {
        inOrder(root);
        return res.get(k - 1);
    }
    public void inOrder(TreeNode node){
        if (node == null) return;
        inOrder(node.right);
        res.add(node.val);
        inOrder(node.left);
    }

    /**
     * 输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。
     * 给定二叉树 [3,9,20,null,null,15,7]，
     *     3
     *    / \
     *   9  20
     *     /  \
     *    15   7
     * 返回它的最大深度 3 。
     */
    public int maxDepth(TreeNode root) { //TODO 二叉树的遍历
        // 二叉树的遍历分为DFS：先序，中序，后序；BFS；按层遍历
        // DFS常使用递归或栈；BFS常使用队列
        // 方法一：DFS, 采用递归，分析子问题：树的深度等于MAX(左子树深度，右子树深度)+1 (更好一些)
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
        // 方法二：BFS，使用队列，树的深度等于层数
        /*if (root == null) return 0;
        List<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int res = 0;
        while (!queue.isEmpty()){
            List<TreeNode> tmp = new LinkedList<>();
            for (TreeNode treeNode : queue) {
                if (treeNode.left != null) tmp.add(treeNode.left);
                if (treeNode.right != null) tmp.add(treeNode.right);
            }
            queue = tmp;
            res++;
        }
        return res;*/
    }

    /**
     * 输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中 **任意节点** 的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。
     * 给定二叉树 [3,9,20,null,null,15,7]
     *     3
     *    / \
     *   9  20
     *     /  \
     *    15   7
     * 返回 true 。
     * 给定二叉树 [1,2,2,3,3,null,null,4,4]
     *        1
     *       / \
     *      2   2
     *     / \
     *    3   3
     *   / \
     *  4   4
     * 返回 false 。
     */
    public boolean isBalanced(TreeNode root) {
        if (root == null) return true;
        return Math.abs(maxDepth(root.left) - maxDepth(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right);
    }

    /**
     * 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
     * 输入：nums = [2,7,11,15], target = 9
     * 输出：[2,7] 或者 [7,2]
     * 输入：nums = [10,26,30,31,47,60], target = 40
     * 输出：[10,30] 或者 [30,10]
     */
    public int[] twoSum(int[] nums, int target) {
        /*if (nums.length == 0 || nums.length == 1) return new int[0];
        int[] res = new int[2];
        boolean flag = false;
        for (int i = 0; i < nums.length - 1; i++) {
            int l = i + 1, r = nums.length -1, newTarget = target - nums[i];
            while (l <= r){
                int mid = (l+r)/2;
                if (nums[mid] > newTarget){
                    r = mid - 1;
                }else if (nums[mid] < newTarget){
                    l = mid + 1;
                }else {
                    res[0] = nums[i];
                    res[1] = newTarget;
                    flag = true;
                    break;
                }
            }
            if (flag) break;
        }
        if (res[0] + res[1] != target) return new int[0];
        return res;*/ // 能做出来，但太耗时
        //TODO K神：双指针(对撞指针)
        int l = 0, r = nums.length - 1;
        while (l < r){
            if (nums[l] + nums[r] > target) r--; //最大的加最小的都比target大，所以最大的数舍弃；
            else if (nums[l] + nums[r] < target) l++; //最小的加最大的都比target小，所以最小的舍弃
            else return new int[] { nums[l], nums[r] };
        }
        return new int[0];
    }

    /**
     * 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
     * 序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
     * 示例 1：
     * 输入：target = 9
     * 输出：[[2,3,4],[4,5]]
     * 示例 2：
     * 输入：target = 15
     * 输出：[[1,2,3,4,5],[4,5,6],[7,8]]
     */
    public int[][] findContinuousSequence(int target) {
        //todo k神 滑动窗口 看不懂
        /**
         * 算法流程：
         * 初始化： 左边界 i=1，右边界 j=2，元素和 s=3，结果列表 res；
         * 循环： 当 i≥j 时跳出；
         *  当 s>target 时： 向右移动左边界 i=i+1，并更新元素和 s；
         *  当 s<target 时： 向右移动右边界 j=j+1，并更新元素和 s；
         *  当 s=target 时： 记录连续整数序列，并向右移动左边界 i=i+1；
         * 返回值： 返回结果列表 res；
         */
        int i = 1, j = 2, s = 3;
        List<int[]> res = new ArrayList<>();
        while (i < j){
            if (s == target){
                int[] ans = new int[j - i +1];
                for (int k = i; k <= j; k++) {
                    ans[k-i] = k;
                }
                res.add(ans);
            }
            if (s >= target){
                s -= i;
                i++;
            }else {
                j++;
                s += j;
            }
        }
        return res.toArray(new int[0][]);
    }

    /**
     * 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。
     * 例如输入字符串"I am a student. "，则输出"student. a am I"。
     * 示例 1：
     * 输入: "the sky is blue"
     * 输出: "blue is sky the"
     * 示例 2：
     * 输入: "  hello world!  "
     * 输出: "world! hello"
     * 解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
     * 示例 3：
     * 输入: "a good   example"
     * 输出: "example good a"
     * 解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
     */
    public String reverseWords(String s) {
        /*String[] split = s.split(" ");
        if (split.length == 0) return "";
        StringBuilder res = new StringBuilder(split[split.length - 1]);
        for (int i = split.length - 2; i >= 0; i--) {
            if (!split[i].equals("")) res.append(" ").append(split[i]);
        }
        return res.toString();*/
        //todo K神，双指针, 但是测试效果不如上边的方法
        s = s.trim(); //去掉首尾空格
        int j = s.length() - 1, i = j;
        StringBuilder res = new StringBuilder();
        while (i >= 0){
            while (i >= 0 && s.charAt(i) != ' ') i--; //从最后一个字母开始，找到第一个空格
            res.append(s, i+1, j+1).append(" ");
            while (i>=0 && s.charAt(i) == ' ') i--; //跳过单词间的所有空格，找到下一个单词
            j = i;
        }
        return res.toString().trim();
    }

    /**
     * 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。
     * 比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。
     * 示例 1：
     * 输入: s = "abcdefg", k = 2
     * 输出: "cdefgab"
     * 示例 2：
     * 输入: s = "lrloseumgh", k = 6
     * 输出: "umghlrlose"
     */
    public String reverseLeftWords(String s, int n) {
        // 方法一：切片
//        return s.substring(n) + s.substring(0, n);
        // 方法二：遍历拼接
        /*StringBuilder res = new StringBuilder();
        for (int i = n; i < s.length(); i++) {
            res.append(s.charAt(i));
        }
        for (int i = 0; i < n; i++) {
            res.append(s.charAt(i));
        }
        return res.toString();*/
        // 方法三：字符串拼接（不使用StringBuilder）
        String res = "";
        for (int i = n; i < s.length(); i++) {
            res = res + s.charAt(i);
        }
        for (int i = 0; i < n; i++) {
            res = res + s.charAt(i);
        }
        return res; // 三个方法，效率逐渐变差
    }

    /**
     * 从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。
     * 2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A不能视为 14。
     * 示例 1:
     * 输入: [1,2,3,4,5]
     * 输出: True
     * 示例 2:
     * 输入: [0,0,1,2,5]
     * 输出: True
     */
    public boolean isStraight(int[] nums) {
        /**
         * 解题思路：
         * 根据题意，此 5 张牌是顺子的充分条件 如下：
         *  除大小王外，所有牌无重复 ；
         *  设此 5 张牌中最大的牌为 max ，最小的牌为 min（大小王除外），则需满足：max−min<5
         */
        // 方法一：利用set元素不重复
        /*Set<Integer> repeat = new HashSet<>();
        int max = 0;
        int min = 14;
        for (int num : nums) {
            if (num == 0) continue;
            if (repeat.contains(num)) return false;
            max = Math.max(max, num);
            min = Math.min(min, num);
            repeat.add(num);
        }
        return max - min < 5;*/
        // 方法二：排序加遍历，先对数组进行排序，排序后相同元素相邻
        int joker = 0; //大小王的数量
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) joker++; //统计大小王的数量
            else if (nums[i] == nums[i+1]) return false; // 如果有除了大小王外的相同元素，直接返回false
        }
        return nums[4] - nums[joker] < 5; // 排序后，max = nums[4]， min = nums[joker]

    }

    /**
     * 0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。
     * 例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。
     * 示例 1：
     * 输入: n = 5, m = 3
     * 输出: 3
     * 示例 2：
     * 输入: n = 10, m = 17
     * 输出: 2
     */
    public int lastRemaining(int n, int m) {
        /*List<Integer> tmp = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            tmp.add(i);
        }
        int j = 0;
        while (tmp.size() > 1){
            j = j + (m - 1);
            if (j > tmp.size() - 1) j = j % tmp.size();
            tmp.remove(j);
        }
        return tmp.get(0);*/ // 效率太低
        //todo k神 动态规划
        int x = 0;
        for (int i = 2; i <= n; i++) {
            x = (x + m) % i;
        }
        return x;
    }

    /**
     * 写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。
     * 输入: a = 1, b = 1
     * 输出: 2
     */
    public int add(int a, int b) { //todo 看！！！
        /**
         * 显然该题要使用位运算，定义两数之和为s，则
         *  s = a和b的非进位和 + 进位
         *  非进位和：a ^ b (a b异或)
         *  进位：(a & b) << 1 (a 与 b 然后左移一位)
         * 由于题目要求不能出现“+”，因此 s = a和b的非进位和 + 进位 中的“+”可用递归来替代
         */
        if (b == 0) return a;
        return add(a ^ b, (a & b) << 1);
    }

    /**
     * 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
     * 百度百科中最近公共祖先的定义为：
     *  “对于有根树T的两个结点p、q，最近公共祖先表示为一个结点x，满足x是p、q的祖先且x的深度尽可能大（一个节点也可以是它自己的祖先）。”
     * 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
     * 输出: 6
     * 解释: 节点 2 和节点 8 的最近公共祖先是 6。
     * 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
     * 输出: 2
     * 解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) { //todo
        /**
         * 祖先的定义：节点root是节点q的祖先。当且仅当q在root的左/右子树，或者q == root
         * 最近的公共祖先：root是p和q的公共祖先，若root.left和root.right都不是p和q的公共祖先，则root是最近的公共祖先
         * 因此，若root是p和q的公共祖先，则有以下3种情况：
         *  1. p和q分别在root的左右子树中；
         *  2. p == root，且q在root的左/右子树中；
         *  3. q == root，且p在root的左/右子树中；
         */
        // 方法一：遍历
        /*while (root != null){
            if (root.val < p.val && root.val < q.val){
                // 说明p和q都在root的右子树中
                root = root.right;
            }else if (root.val > p.val && root.val > q.val){
                //说明p和q都在root的左子树中
                root = root.left;
            }else break;
        }
        return root;*/
        // 方法二：递归
        if (root.val < p.val && root.val < q.val) {
            // 说明p和q都在root的右子树中
            return lowestCommonAncestor(root.right, p, q);
        }
        if (root.val > p.val && root.val > q.val){
            return lowestCommonAncestor(root.left, p, q);
        }
        return root;
    }

    /**
     * 上题还有变体：将二叉搜索树换为了二叉树，此时没有节点值之间的大小关系了
     */
    public TreeNode lowestCommonAncestorWithCommonTree(TreeNode root, TreeNode p, TreeNode q) { //todo
        if (root == null) return null;
        if (p == root || q == root) return root; // 如果pq有一个是根节点，则最近公共祖先一定是root
        TreeNode left = lowestCommonAncestorWithCommonTree(root.left, p, q); // 在root的左子树中寻找p q，将其定义为left
        TreeNode right = lowestCommonAncestorWithCommonTree(root.right, p, q); // 在root的右子树中寻找p q，将其定义为right
        if (left == null) return right; // 说明左子树中没找到 那么就在right中
        else if (right == null) return left; // 同理，右子树没找到，在left中
        else return root; // left和right都不为空，说明p和q在root异侧，则root是其最近公共祖先
    }


}
