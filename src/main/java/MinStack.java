import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

/**
 * 定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。
 * MinStack minStack = new MinStack();
 * minStack.push(-2);
 * minStack.push(0);
 * minStack.push(-3);
 * minStack.min();   --> 返回 -3.
 * minStack.pop();
 * minStack.top();      --> 返回 0.
 * minStack.min();   --> 返回 -2.
 */
/*
class MinStack {

    private List<Integer> stack;

    */
/** initialize your data structure here. *//*

    public MinStack() {
        stack = new ArrayList<>();
    }
    
    public void push(int x) {
        stack.add(x);
//        if (x < min) min = x;
    }
    
    public void pop() {
        stack.remove(stack.size() - 1);
    }
    
    public int top() {
        return stack.get(stack.size() - 1);
    }
    
    public int min() {
        Integer min = stack.get(0);
        for (Integer data : stack) {
            if (data < min) min = data;
        }
        return min;
    }
}*/
//TODO 官方题解
class MinStack {
    // 本题的重点在于要求 min 函数的时间复杂度为O(1)，简单来想的话，取最小值是要遍历的，时间复杂度为O(N)
    // 官方用了辅助栈的方法，定义一个辅助栈来存储最小值
    Deque<Integer> stack;
    Deque<Integer> minStack;

    /** initialize your data structure here. */
    public MinStack() {
        stack = new LinkedList<>();
        minStack = new LinkedList<>();
        minStack.push(Integer.MAX_VALUE); // 保证之后存到该栈中的元素都是最小值
    }

    public void push(int x) {
        stack.push(x);
        if (minStack.peek() < x){
            minStack.push(minStack.peek());
        }
        if (minStack.peek() >= x){
            minStack.push(x);
        }
//        minStack.push(Math.min(minStack.peek(), x));
    }

    public void pop() {
        stack.pop();
        minStack.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int min() {
        return minStack.peek();
    }
}
