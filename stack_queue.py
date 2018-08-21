#4.1 括号匹配
class ValidParentheses:


    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        left=('(','[','{')
        right=(')',']','}')

        stack=[]
        for ele in s:
            if ele in right:
                if (len(stack)>0 and
                   left.index(stack[-1])==right.index(ele)):
                    stack.pop()
                else:
                    return False
            else:
                stack.append(ele)
        return len(stack)==0


    def run(self):
        s="()"
        print(self.isValid(s))

#4.1.2 最长有效括号
class LongestValidParentheses:

    def longest(self,s):
        '''
        :type s:str
        :param s:
        :return:
        '''
        last=-1 #最后一个未匹配的右括号的下标
        maxLen=0
        left=[] #作为栈，储存未匹配的左括号的下标
        for i in range(len(s)):
            if s[i]=='(':
                left.append(i)
            else:
                if len(left)>0:
                    #匹配成功
                    left.pop()
                    if len(left)==0:
                        #如果左括号已经全部匹配
                        #有效区间为最后未匹配的右括号之后到当前位置
                        maxlen=max(maxLen,i-last)
                    else:
                        #如果还有左括号未匹配
                        #有效区间为未匹配的左括号之后到当前位置
                        maxLen=max(maxLen,i-left[-1])
                else:
                    #匹配失败
                    last=i
        return maxLen


class  LargestRectangleInHistogram:

    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        heights.append(0)
        res=0
        i=0
        stack=[]
        while i<len(heights):
            if len(stack)==0 or heights[i]>heights[stack[-1]]:
                stack.append(i)
                i+=1
            else:
                tmp=stack[-1]
                stack.pop()
                res=max(res,
                       heights[tmp]*(i if len(stack)==0 else i-stack[-1]-1)
                       )
        return res

    #暴力求解
    def brust(self,heights):
        '''
        :type heights:list
        :param heights:
        :return:
        '''
        res = 0
        for i in range(len(heights)):
            left = i - 1
            right = i + 1
            while left > -1 and heights[left] >= heights[i]: #注意边界
                left -= 1
            while right < len(heights) and heights[right] >= heights[i]: #注意边界
                right += 1
            res = max(res, heights[i] * (right - left - 1))
        return res

    def run(self):
        h=[2,1,2]
        print(self.brust(h))

#逆波兰表达式求值
class EvalRPN:
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        oper=("+","-","*","/")
        num_stack=[]
        for element in tokens:
            if element not in oper:
                num_stack.append(int(element))
            else:
                tmp=0
                if element=='+':
                    num2=num_stack.pop()
                    num1=num_stack.pop()
                    tmp=num1+num2
                elif element=='-':
                    num2=num_stack.pop()
                    num1=num_stack.pop()
                    tmp=num1-num2
                elif element=='*':
                    num2=num_stack.pop()
                    num1=num_stack.pop()
                    tmp=num1*num2
                else:
                    num2=num_stack.pop()
                    num1=num_stack.pop()
                    tmp=num1/num2
                num_stack.append(tmp)
        return num_stack.pop()


    def run(self):
        s=["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
        print(self.evalRPN(s))





def main():
    sol=EvalRPN()
    sol.run()

if __name__ == '__main__':
    main()
