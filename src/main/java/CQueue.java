import java.util.Stack;

/**
 * 用两个栈实现一个队列。队列的声明如下，
 * 请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。
 * (若队列中没有元素，deleteHead操作返回 -1 )
 */
class CQueue {
    Stack<Integer> s1;
    Stack<Integer> s2;
    public CQueue() {
        s1 = new Stack<>();
        s2 = new Stack<>();
    }
    
    public void appendTail(int value) {
        s1.push(value);
    }
    
    public int deleteHead() {
       /* if (s1.size() == 0){
            return -1;
        }
        while (s1.size() != 1){
            s2.push(s1.pop());
        }
        int res = s1.pop();
        while (s2.size() != 0){
            s1.push(s2.pop());
        }
        return res;*/
        if (!s2.isEmpty()){
            return s2.pop();
        }
        if (s1.isEmpty()){
            return -1;
        }
        while (!s1.isEmpty()){
            s2.push(s1.pop());
        }
        return s2.pop();
    }
}