import org.junit.Assert;
import org.junit.Test;

/**
 * * MinStack minStack = new MinStack();
 * * minStack.push(-2);
 * * minStack.push(0);
 * * minStack.push(-3);
 * * minStack.min();   --> 返回 -3.
 * * minStack.pop();
 * * minStack.top();      --> 返回 0.
 * * minStack.min();   --> 返回 -2.
 */
public class MinStackTest {
    MinStack minStack = new MinStack();
    @Test
    public void test(){
        minStack.push(2);
        minStack.push(0);
        minStack.push(3);
        minStack.push(0);
        Assert.assertEquals(0, minStack.min());
        minStack.pop();
        Assert.assertEquals(0, minStack.min());
        minStack.pop();
        Assert.assertEquals(0, minStack.min());
        minStack.pop();
        Assert.assertEquals(2, minStack.min());
    }

}